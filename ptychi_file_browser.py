import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import tifffile
from PIL import Image
import time

from scan_watcher_thread import ScanWatcherThread

import pyqtgraph as pg
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QColor, QBrush


class PtychiReconBrowser(QtWidgets.QMainWindow):
    def __init__(self, ui_path, parent=None):
        super().__init__(parent)

        # Load UI
        uic.loadUi(ui_path, self)

        # ---- Basic UI setup ----
        # self.graphics_scene = QtWidgets.QGraphicsScene(self)
        # self.graphicsView_1.setScene(self.graphics_scene)


        self.base_path: Path | None = None
        self.viewChoice = ''
        self.res_m = 1.
        self.runtable_df = None
        self.scan_watcher = None
        self.scan_goodness = 'unknown'
        self.file_load_path = None

        self._scan_row_items = {}     # scan_name -> QTreeWidgetItem
        self._seen_scans = set()      # just scan names
        self._seen_param_folders = {} # scan_name -> set(param_folder_paths)
        self._seen_recon_files = {}   # scan_name -> {param_folder_path -> set(recon_file_paths)}

        self._set_scan_watcher_ui('gray')
        self._setup_tree()
        self._connect_signals()
        self._setup_pyqtgraph_view()
        self.on_base_path_entered()

        self.treeWidget_fileStructure.setContextMenuPolicy(Qt.CustomContextMenu)
        self.treeWidget_fileStructure.customContextMenuRequested.connect(self.on_tree_right_click)



    def _setup_tree(self):
        """Initial tree configuration."""
        self.treeWidget_fileStructure.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection
        )
        self.treeWidget_fileStructure.setUniformRowHeights(True)
        self.treeWidget_fileStructure.setRootIsDecorated(False)
        self.treeWidget_fileStructure.viewport().setMouseTracking(True)

        # Set initial column widths (pixels)
        self.treeWidget_fileStructure.setColumnWidth(0, 50)  # Scan column
        self.treeWidget_fileStructure.setColumnWidth(1, 85)  # Param folder
        self.treeWidget_fileStructure.setColumnWidth(2, 50)  # Recon file
        self.treeWidget_fileStructure.setColumnWidth(3, 100)  # Sample name

        # header = self.treeWidget_fileStructure.header()
        # header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)


    def _connect_signals(self):
        """Wire UI signals (empty handlers for now)."""
        self.lineEdit_basePath.returnPressed.connect(self.on_base_path_entered)
        self.pushButton_browseBasePath.clicked.connect(self.on_browse_base_path)
        self.pushButton_populateTree.clicked.connect(self.populate_tree_with_scans)
        self.treeWidget_fileStructure.itemClicked.connect(self.on_tree_item_clicked)
        self.treeWidget_fileStructure.currentItemChanged.connect(self.on_tree_selection_changed)
        self.pushButton_stopScanUpdate.clicked.connect(self.on_stop_scan_update)
        self.pushButton_updateScanGoodness.clicked.connect(self.on_update_scan_goodness)


    def _setup_pyqtgraph_view(self):
        """
        Embed pyqtgraph ImageView into the placeholder widget.
        """
        self.pg_view = pg.ImageView()
        self.graphicsView_1_layout = QtWidgets.QVBoxLayout(self.graphicsView_1)
        self.graphicsView_1_layout.setContentsMargins(0, 0, 0, 0)
        self.graphicsView_1_layout.addWidget(self.pg_view)


    # ------------------------------------------------------------------
    # util/misc
    # ------------------------------------------------------------------

    def set_item_tooltip(self, item, column):
        """
        Set tooltip for a single cell to match its visible text.
        """
        text = item.text(column)
        if text:
            item.setData(column, Qt.ToolTipRole, text)


    def add_to_tree(self, tree_row, tree_idx, folder_in):
        tree_row.setText(tree_idx, folder_in.name)
        tree_row.setData(tree_idx, Qt.UserRole, folder_in)
        tree_row.setData(tree_idx, Qt.ToolTipRole, folder_in.name)


    # ------------------------------------------------------------------
    # scan goodness
    # ------------------------------------------------------------------

    def on_update_scan_goodness(self):
        item = self.treeWidget_fileStructure.currentItem()
        if item is None:
            return

        scan_path = item.data(0, Qt.UserRole)
        if not isinstance(scan_path, Path):
            return

        self.write_scan_goodness(scan_path)
        item.setData(0, Qt.UserRole + 1, self.scan_goodness)
        self.apply_scan_goodness_style(item, item.data(0, Qt.UserRole+1))


    def write_scan_goodness(self, scan_path: Path):
        """
        Create / remove scan goodness files based on radio button state.
        """
        good_file = scan_path / "scan_is_good.txt"
        reanalyze_file = scan_path / "scan_should_be_reanalyzed.txt"
        bad_file = scan_path / "scan_is_bad.txt"

        if self.radioButton_good.isChecked():
            self.scan_goodness = 'good'
            good_file.write_text("good\n")
            if reanalyze_file.exists():
                reanalyze_file.unlink()
            if bad_file.exists():
                bad_file.unlink()

        elif self.radioButton_reanalyze.isChecked():
            self.scan_goodness = 'reanalyze'
            reanalyze_file.write_text("reanalyze\n")
            if good_file.exists():
                good_file.unlink()
            if bad_file.exists():
                bad_file.unlink()

        elif self.radioButton_bad.isChecked():
            self.scan_goodness = 'bad'
            bad_file.write_text("bad\n")
            if good_file.exists():
                good_file.unlink()
            if reanalyze_file.exists():
                reanalyze_file.unlink()

        else:  # unknown
            self.scan_goodness = 'unknown'
            if good_file.exists():
                good_file.unlink()
            if bad_file.exists():
                bad_file.unlink()


    def update_scan_goodness_ui(self, goodness: str):
        """
        Check the scan folder and update the radio buttons.
        """
        # Default: unknown
        self.radioButton_good.setChecked(True if goodness == 'good' else False)
        self.radioButton_reanalyze.setChecked(True if goodness == 'reanalyze' else False)
        self.radioButton_bad.setChecked(True if goodness == 'bad' else False)
        self.radioButton_unknown.setChecked(True if goodness == 'unknown' else False)


    def apply_scan_goodness_style(self, row_item: QtWidgets.QTreeWidgetItem, goodness: str):
        if goodness == "good":
            color = QColor(198,239,206)  # light green
            for col in range(self.treeWidget_fileStructure.columnCount()):
                row_item.setBackground(col, color)

        elif goodness == "reanalyze":
            color = QColor(255,235,156)  # light yellow
            for col in range(self.treeWidget_fileStructure.columnCount()):
                row_item.setBackground(col, color)

        elif goodness == "bad":
            color = QColor(255,199,206)  # light red
            for col in range(self.treeWidget_fileStructure.columnCount()):
                row_item.setBackground(col, color)

        else:
            for col in range(self.treeWidget_fileStructure.columnCount()):
                row_item.setBackground(col, QBrush())


    # ------------------------------------------------------------------
    # thread logic
    # ------------------------------------------------------------------

    def start_scan_watcher(self):
        if self.base_path is None:
            return

        # Stop existing watcher
        if self.scan_watcher is not None:
            self.scan_watcher.stop()
            self.scan_watcher.wait()

        # self.scan_watcher = ScanWatcherThread(self.base_path, 
        #                                       seen_scans=self._seen_scans,
        #                                       seen_param_folders=self._seen_param_folders,
        #                                       seen_recon_files=self._seen_recon_files)
        # self.scan_watcher.scan_found.connect(self.on_scan_found)
        # self.scan_watcher.param_folder_found.connect(self.on_param_folder_found)
        # self.scan_watcher.recon_file_found.connect(self.on_recon_file_found)
        # self.scan_watcher.start()
        
        self.scan_watcher = ScanWatcherThread(Path(self.base_path), poll_interval=2.0)
        self.scan_watcher.scan_found.connect(self.on_scan_found)
        self.scan_watcher.finished_adding_scans.connect(self.on_finished_adding_scans)
        self.scan_watcher.start()


        self._set_scan_watcher_ui('running')

        
    def on_scan_found(self, scan_path: Path):
        print(f"New scan detected: {scan_path.name}")

        if scan_path.name in self._scan_row_items:
            self._refresh_scan_row(scan_path)   # existing → update in place
        else:
            self._add_scan_row(scan_path)       # new → add row


        
    def on_finished_adding_scans(self):
        self.treeWidget_fileStructure.setSortingEnabled(True)
        self.treeWidget_fileStructure.sortByColumn(0, Qt.AscendingOrder)
        self.treeWidget_fileStructure.setSortingEnabled(False)


    def on_param_folder_found(self, param_path: Path):
        # Update the row for this scan
        scan_name = param_path.parent.name
        self._add_param_row(self._scan_row_items.get(scan_name), scan_name, param_path)
        

    def on_recon_file_found(self, param_path: Path, recon_file: Path):
        scan_name = param_path.parent.name
        self._add_recon_row(self._scan_row_items.get(scan_name), scan_name, param_path, recon_file)


    def on_stop_scan_update(self):
        if self.scan_watcher is not None:
            self.scan_watcher.stop()
            self.scan_watcher.wait()
            self.scan_watcher = None
            self._set_scan_watcher_ui('stopped')
            print("Scan update stopped.")
        else:
            self.start_scan_watcher()


    def closeEvent(self, event):
        if getattr(self, "scan_watcher", None) is not None:
            self.scan_watcher.stop()
            self.scan_watcher.wait()
            self._set_scan_watcher_ui('gray')
        event.accept()


    def _set_scan_watcher_ui(self, status: str):
        if status == 'running':
            self.pushButton_stopScanUpdate.setEnabled(True)
            self.pushButton_stopScanUpdate.setText("Stop\nUpdating Scans")
            self.pushButton_stopScanUpdate.setStyleSheet("background-color: red; color: white;")
        elif status == 'gray':
            self.pushButton_stopScanUpdate.setEnabled(False)
            self.pushButton_stopScanUpdate.setText("Stopped\nUpdating Scans")
            self.pushButton_stopScanUpdate.setStyleSheet("background-color: lightgray; color: black;")
        elif status == 'stopped':
            self.pushButton_stopScanUpdate.setEnabled(True)
            self.pushButton_stopScanUpdate.setText("Start\nUpdating Scans")
            self.pushButton_stopScanUpdate.setStyleSheet("background-color: green; color: white;")


    # ------------------------------------------------------------------
    # file searching util
    # ------------------------------------------------------------------

    def find_recent_folder(self, dir_in):
        it = list(os.scandir(dir_in))
        names = {e.name for e in it}
        self.scan_goodness = (
            'good' if 'scan_is_good.txt' in names else
            'reanalyze' if 'scan_should_be_reanalyzed.txt' in names else
            'bad' if 'scan_is_bad.txt' in names else
            'unknown'
        )

        subdirs = [e for e in it if e.is_dir(follow_symlinks=False)]

        if not subdirs:
            return None, None
        if len(subdirs) == 1:
            temp = Path(subdirs[0].path)
            return temp, [temp,]

        return Path(max(subdirs, key=lambda e: e.stat().st_mtime).path), [Path(e.path) for e in subdirs]
    

    def get_latest_recon_file(self, dir_in, get_all_instead=False):
        """
        Find the latest recon file in dir_in.

        Priority:
        1) recon_Niter<integer>*.h5 with largest integer
        2) most recently modified .h5 file
        """
        best_niter = None
        best_niter_entry = None
        h5_entries = []

        with os.scandir(dir_in) as it:
            for e in it:
                try:
                    if not e.name.endswith(".h5"):
                        continue

                    h5_entries.append(e)

                    if e.name.startswith("recon_Niter"):
                        # fast-path integer parse
                        suffix = e.name[11:-3]  # after "recon_Niter", before ".h5"
                        token = suffix.split("_", 1)[0]
                        if token.isdigit():
                            niter = int(token)
                            if best_niter is None or niter > best_niter:
                                best_niter = niter
                                best_niter_entry = e
                except:
                    continue

        if get_all_instead:
            return [Path(e.path) for e in h5_entries]

        # Prefer largest Niter
        if best_niter_entry is not None:
            return Path(best_niter_entry.path), [Path(e.path) for e in h5_entries]

        if not h5_entries:
            return None, None

        # Only now do we touch stat()
        if len(h5_entries) == 1:
            temp = Path(h5_entries[0].path)
            return temp, [temp,]

        return Path(max(h5_entries, key=lambda e: e.stat().st_mtime).path), [Path(e.path) for e in h5_entries]



    # ------------------------------------------------------------------
    # read runtable
    # ------------------------------------------------------------------
    
    def load_runtable(self):
        """
        Load the runtable_full_*.csv corresponding to the current base_path.

        Returns
        -------
        pd.DataFrame or None
        """
        if self.base_path is None:
            return None

        parent = self.base_path.parent
        csv_name = f"runtable_full_{parent.name}.csv"
        csv_path = parent / csv_name

        if not csv_path.exists():
            return None

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"Failed to load runtable: {exc}")
            return None

        # Normalize run column to int if possible
        if "run" in df.columns:
            df["run"] = pd.to_numeric(df["run"], errors="coerce").astype("Int64")

        return df
    

    def get_sample_name_for_scan(self, scan_path: Path):
        """
        Return sample name for a scan folder S#### if available.
        """
        if self.runtable_df is None:
            return None

        try:
            scan_num = int(scan_path.name[1:])
        except ValueError:
            return None

        if "run" not in self.runtable_df.columns:
            return None
        if "sample_name" not in self.runtable_df.columns:
            return None

        match = self.runtable_df[self.runtable_df["run"] == scan_num]
        if match.empty:
            return None

        value = match.iloc[0]["sample_name"]
        return str(value) if pd.notna(value) else None


    # ------------------------------------------------------------------
    # data loading
    # ------------------------------------------------------------------
    

    def load_data_from_file(self, file_path: Path):
        """
        loading data from a file.

        Parameters
        ----------
        file_path : Path

        Returns
        -------
        data : 2d numpy array
        """
        extension = self.comboBox_imageChoice.currentText()
        if extension == 'recon_NiterXXX.h5':
            self.file_load_path = file_path
        elif extension in ('dp_sum.tiff', 'init_probe_mag.tiff'):
            self.file_load_path = file_path.parent / extension
        else:
            base = extension.rsplit("Niter", 1)[0]
            suffix = file_path.stem.split("recon_", 1)[1]
            self.file_load_path = file_path.parent / f"{base}{suffix}{Path(extension).suffix}"
            # self.file_load_path = file_path.parent / (extension.split('.')[0][:-3] + '%d.%s' %(int(file_path.stem.split('Niter')[1]), extension.split('.')[1]))

        if self.file_load_path.suffix in ('.h5', '.hdf5'):
            with h5py.File(self.file_load_path, 'r') as f:
                obj = np.angle(f['object'][0][()])
                self.res_m = float(f['obj_pixel_size_m'][()])

                
        if self.file_load_path.suffix in ('.tiff',):
            with tifffile.TiffFile(self.file_load_path) as tif:
                obj = tif.asarray()
                if 'pixel_size' in tif.imagej_metadata.keys():
                    self.res_m = 1e-6 * tif.imagej_metadata['pixel_size']
                    
                elif 'xspacing' in tif.imagej_metadata.keys():
                    self.res_m = 1e-6 * tif.imagej_metadata['xspacing']

                if len(obj.shape) == 3:
                    obj = np.mean(obj, 2).T

        if self.file_load_path.suffix in ('.png',):
            obj = Image.open(self.file_load_path).convert("L")  # L = grayscale
            obj = np.array(obj, dtype=np.float32).T


        return obj


    # ------------------------------------------------------------------
    # plotting
    # ------------------------------------------------------------------

    def set_plot_label(self, scan: str, sample_name: str, file_path: str):
        full_text = f"{scan}, {sample_name}\n{file_path}"

        # Elide long lines
        elide_width = 500  # pixels, adjust to fit your layout
        metrics = self.label_plot_1.fontMetrics()
        lines = full_text.split("\n")
        elided_lines = [metrics.elidedText(line, Qt.ElideMiddle, elide_width) for line in lines]
        elided_text = "\n".join(elided_lines)

        self.label_plot_1.setText(elided_text)
        self.label_plot_1.setToolTip(full_text)


    def display_data(self, data: np.ndarray, scan: int, sample_name: str):
        """
        Display a 2D numpy array using pyqtgraph.
        """
        if data.ndim != 2:
            raise ValueError("display_data expects a 2D numpy array")

        # Update title label
        # self.label_plot_1.setText("%s, %s\n%s" % (scan, sample_name, self.file_load_path))
        self.set_plot_label(scan, sample_name, str(self.file_load_path))


        # Convert to float32 for pyqtgraph
        data = data.astype(np.float32)

        # Display
        if self.checkBox_logCmap.isChecked():
            self.pg_view.setImage(np.log10(np.clip(np.abs(data), a_min=np.finfo(float).eps, a_max=None)), autoLevels=True)

        self.pg_view.setImage(data, autoLevels=True)

        # Enable colorbar
        if not hasattr(self, "_colorbar_added"):
            self.pg_view.ui.histogram.show()
            self._colorbar_added = True


    # New handler
    def on_tree_selection_changed(self, current: QtWidgets.QTreeWidgetItem, previous: QtWidgets.QTreeWidgetItem):
        if current is not None:
            # Determine the column you want to act on; e.g., column 2 for recon file
            self.on_tree_item_clicked(current, 2)

    
    def on_tree_item_clicked(self, item, column):
        """
        Handle clicks anywhere on a scan row.
        """
    
        # Try recon file first
        file_path = item.data(2, Qt.UserRole)

        if not isinstance(file_path, Path):
            return

        data = self.load_data_from_file(file_path)
        if data is None:
            return

        self.display_data(data, item.text(0), item.text(3))
        self.update_scan_goodness_ui(item.data(0, Qt.UserRole + 1))


    def on_tree_right_click(self, pos):
        item = self.treeWidget_fileStructure.itemAt(pos)
        if item is None:
            return

        # Get the column under the mouse
        index = self.treeWidget_fileStructure.indexAt(pos)
        column = index.column()
        scan_name = item.text(0)

        if column not in (0, 1, 2):
            return

        menu = QtWidgets.QMenu()

        if column == 0:
            # trigger an update of the file scan
            action = menu.addAction("Refresh scan")
            action.triggered.connect(
                lambda checked, p=(self.base_path / scan_name): self._refresh_scan_row(p)
            )

        if column == 1:  # column 1 stores param folder
            # Add all param folders for this scan to the menu
            for param_path in sorted(self._seen_param_folders[scan_name]):
                action = menu.addAction(param_path.name)
                # Use a lambda to capture param_path
                action.triggered.connect(lambda checked, p=param_path, i=item: self._switch_param_folder(i, scan_name, p))

        elif column == 2:  # column 1 stores param folder
            param_name = item.text(1)

            # Add all param folders for this scan to the menu
            # for recon_file in sorted(self._seen_recon_files[scan_name][param_name]):
            for recon_file in sorted(self._seen_recon_files[scan_name][param_name], key=self._recon_sort_key):
                action = menu.addAction(recon_file.name)
                # Use a lambda to capture recon_file
                action.triggered.connect(lambda checked, r=recon_file, i=item: self._switch_recon_file(i, r))

        menu.exec_(self.treeWidget_fileStructure.viewport().mapToGlobal(pos))
        self.on_tree_item_clicked(item, 1)


    def _recon_sort_key(self, p: Path):
        name = p.stem

        if "_idx" in name:
            # recon_idx15_Niter20_60 → 15
            return int(name.split("_idx")[1].split("_")[0])

        # recon_Niter200 → 200
        return int(name.split("Niter")[-1].split("_")[0])


    def _switch_param_folder(self, row_item, scan_name, param_path):
        """
        Update column 1 to a new param folder and refresh the latest recon in column 2.
        """
        # Update column 1
        self.add_to_tree(row_item, 1, param_path)

        # Get latest recon in this param folder
        latest_recon, all_recon = self.get_latest_recon_file(param_path)

        if latest_recon is not None:
            # Store all recon files
            for recon_file in all_recon:
                if recon_file not in self._seen_recon_files[scan_name][param_path.name]:
                    self._seen_recon_files[scan_name][param_path.name].add(recon_file)

            # Update column 2
            self._add_recon_row(row_item, scan_name, param_path, latest_recon)

        else:
            row_item.setText(2, "—")
            row_item.setData(2, Qt.ToolTipRole, "No recon file found")


    def _switch_recon_file(self, row_item, recon_file):
        # Update the displayed data
        self.add_to_tree(row_item, 2, recon_file)


    # ------------------------------------------------------------------
    # scan tree
    # ------------------------------------------------------------------

    def iter_scan_folders(self):
        if self.base_path is None or not self.base_path.exists():
            return
        
        with os.scandir(self.base_path) as it:
            for entry in it:

                name = entry.name
                if len(name) == 5 and name.startswith("S") and name[1:].isdigit():
                    yield Path(entry.path)


    def populate_tree_with_scans(self):
        """
        Depth-first population of the tree.
        """
        self.treeWidget_fileStructure.clear()

        if self.base_path is None:
            return
        
        t0 = time.time()
        self.treeWidget_fileStructure.setUpdatesEnabled(False)
        for i, scan_path in enumerate(self.iter_scan_folders()):
            if ((i+1) % 30 == 0):
                self.treeWidget_fileStructure.setUpdatesEnabled(True)
            if i == 5:
                self.treeWidget_fileStructure.resizeColumnToContents(0)
                self.treeWidget_fileStructure.resizeColumnToContents(2)

            print(f"Processing {scan_path.name}")

            self._add_scan_row(scan_path)

            # Keep UI responsive
            if ((i+1) % 30 == 0):
                QtWidgets.QApplication.processEvents()

            if (i+1) % 30 == 0:
                self.treeWidget_fileStructure.setUpdatesEnabled(False)

        self.treeWidget_fileStructure.setUpdatesEnabled(True)
        QtWidgets.QApplication.processEvents()

        self.treeWidget_fileStructure.resizeColumnToContents(0)
        self.treeWidget_fileStructure.resizeColumnToContents(2)
        # self.treeWidget_fileStructure.setUpdatesEnabled(False)
        print(time.time() - t0, 's')

        self.start_scan_watcher()


    def _add_param_folder(self, scan_name: str, param_path: Path):
        """Register a param folder under a scan."""
        self._seen_param_folders[scan_name].add(param_path)
        self._seen_recon_files[scan_name].setdefault(param_path.name, set())


    def _add_scan_row(self, scan_path: Path):
        """
        Add one row to the tree for a single scan and track all param folders and recon files.
        """
        # Create the tree row
        row_item = QtWidgets.QTreeWidgetItem(self.treeWidget_fileStructure)
        self._scan_row_items[scan_path.name] = row_item
        self._populate_scan_row(row_item, scan_path)


    def _refresh_scan_row(self, scan_path: Path):
        """
        Replace an existing scan row with a fresh one, then sort the tree.
        """
        # Remove existing row if it exists
        row_item = self._scan_row_items.pop(scan_path.name, None)
        if row_item is not None:
            index = self.treeWidget_fileStructure.indexOfTopLevelItem(row_item)
            if index != -1:
                self.treeWidget_fileStructure.takeTopLevelItem(index)

        # Add a new row
        self._add_scan_row(scan_path)

        # Optionally sort by scan name (or keep your custom order)
        self.treeWidget_fileStructure.sortItems(0, Qt.AscendingOrder)

        # Make sure the new row is selected
        row_item = self._scan_row_items.get(scan_path.name)
        if row_item is not None:
            self.treeWidget_fileStructure.setCurrentItem(row_item)


    def _populate_scan_row(self, row_item: QtWidgets.QTreeWidgetItem, scan_path: Path):
        self.add_to_tree(row_item, 0, scan_path)

        # Initialize nested storage
        self._seen_scans.add(scan_path.name)
        self._seen_param_folders.setdefault(scan_path.name, set())
        self._seen_recon_files.setdefault(scan_path.name, {})

        # ---- Find all param folders ----
        recent_param, all_param = self.find_recent_folder(scan_path)
        row_item.setData(0, Qt.UserRole + 1, self.scan_goodness)  # store it
        self.apply_scan_goodness_style(row_item, self.scan_goodness)

        if recent_param is not None:
            # Store all param folders
            for param_path in all_param:
                if param_path is not None:
                    self._add_param_folder(scan_path.name, param_path)

                    # Store all recon files except most recent
                    if param_path is not recent_param:
                        all_recon = self.get_latest_recon_file(param_path, get_all_instead=True)
                        if all_recon is not None:
                            for recon_file in all_recon:
                                self._seen_recon_files[scan_path.name][param_path.name].add(recon_file)

            # Display the recent param folder (column 1)
            self._add_param_row(row_item, scan_path.name, recent_param)
        else:
            row_item.setText(1, "—")
            row_item.setText(2, "—")
            row_item.setData(1, Qt.ToolTipRole, "No parameter folder found")
            row_item.setData(2, Qt.ToolTipRole, "No recon file found")

        # Column 3: sample name
        sample_name = self.get_sample_name_for_scan(scan_path)
        if sample_name is not None:
            row_item.setText(3, sample_name)
            row_item.setData(3, Qt.ToolTipRole, sample_name)
        else:
            row_item.setText(3, "—")
            row_item.setData(3, Qt.ToolTipRole, "Sample name not found")


    def _add_param_row(self, row_item: QtWidgets.QTreeWidgetItem, scan_name: str, param_path: Path):
        """
        Populate column 1 for param folder and handle latest recon in column 2.
        """
        # Column 1: param folder
        self.add_to_tree(row_item, 1, param_path)

        # ---- Find all recon files in this param folder ----
        latest_recon, all_recon = self.get_latest_recon_file(param_path)

        # Column 2: latest recon
        if latest_recon is not None:

            # Store all recon files
            for recon_file in all_recon:
                self._seen_recon_files[scan_name][param_path.name].add(recon_file)

            self._add_recon_row(row_item, scan_name, param_path, latest_recon)
        else:
            row_item.setText(2, "—")
            row_item.setData(2, Qt.ToolTipRole, "No recon file found")


    def _add_recon_row(self, row_item: QtWidgets.QTreeWidgetItem, scan_name: str, param_path: Path, recon_file: Path):
        """
        Populate column 2 with a recon file.
        """
        self.add_to_tree(row_item, 2, recon_file)


    # ------------------------------------------------------------------
    # handling base path
    # ------------------------------------------------------------------

    def set_base_path(self, path: Path):
        """
        Validate and set the base path.
        This is the single source of truth for updating self.base_path.
        """
        path = path.expanduser().resolve()

        if not path.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid path",
                f"Path does not exist:\n{path}",
            )
            return False

        if not path.is_dir():
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid path",
                f"Path is not a directory:\n{path}",
            )
            return False

        self.base_path = path
        self.lineEdit_basePath.setText(str(path))

        self.runtable_df = self.load_runtable()

        self._set_scan_watcher_ui('gray')
        return True
    

    def on_base_path_entered(self):
        text = self.lineEdit_basePath.text().strip()
        if not text:
            return

        self.set_base_path(Path(text))


    def on_browse_base_path(self):
        start_dir = (
            str(self.base_path)
            if self.base_path is not None
            else str(Path.home())
        )

        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select base path",
            start_dir,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )

        if not directory:
            return

        self.set_base_path(Path(directory))


# ------------------------------------------------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)

    ui_path = Path(__file__).parent / "ptychi_file_browser.ui"
    if not ui_path.exists():
        raise FileNotFoundError(f"UI file not found: {ui_path}")

    window = PtychiReconBrowser(ui_path)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
