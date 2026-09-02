import sys
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import tifffile
from PIL import Image
import time

from scan_watcher_thread import ScanWatcherThread
from pg_image_tools import ImagePlotWidget

TREE_CACHE_FILENAME = "ptychi_file_browser_tree_cache.csv"

# ---- log csv (beamline run log found next to the base path) ----
LOG_CSV_SCAN_COL = "scan"
LOG_CSV_SAMPLE_COL = "sample_name"
LOG_CSV_STEP_COL = "scan_step_size"
SCAN_BAD_LIST_FILENAME = "scan_bad_list.csv"
SCAN_BAD_SCAN_COL = "scan"
SCAN_BAD_FLAG_COL = "is bad"
RUNTABLE_BAD_HEADER = "bad"
LOG_CSV_VIEW_COLS = ["scan", "completed", "date", "time", "ExpTime", "n_pos", "phi", "scan_type"]

# scan_step_size motor -> (displayed unit, multiplier from the unit in the csv).
# Everything not listed here is a real motor, stored in mm and shown in microns.
LOG_CSV_STEP_UNITS = {
    "time": ("ms", 1e3),   # csv is seconds
    "phi": ("deg", 1.0),   # csv is already degrees
}
LOG_CSV_STEP_DEFAULT_UNIT = ("um", 1e3)

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtCore import Qt, QSettings, QEvent
from PyQt5.QtGui import QImage, QPixmap, QColor, QBrush

GOODNESS_COLORS = {
    'good': QColor(198, 239, 206),       # light green
    'reanalyze': QColor(255, 235, 156),  # light yellow
    'bad': QColor(255, 199, 206),        # light red
}

# Runtable row with a recon file but no scan goodness set yet: paler than 'good'
RECON_EXISTS_COLOR = QColor(232, 248, 237)



class RuntableWindow(QtWidgets.QDialog):
    """
    Non-blocking runtable viewer. A plain QDialog, except that it is a real
    window rather than a modal popup and it reports its own closing so the
    'bad' checkboxes can be written out.
    """

    def __init__(self, parent, on_close):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window)
        self._on_close = on_close

    def closeEvent(self, event):
        self._on_close()
        super().closeEvent(event)


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

        # ---- log csv state ----
        self.log_csv_path = None       # Path to the chosen csv
        self.log_csv_df = None         # full DataFrame, only used by the viewer
        self.log_csv_ok = False        # the flag gating every log csv feature
        self._log_sample_by_scan = {}  # "S0042" -> sample name
        self._log_bad_by_scan = {}     # "S0042" -> bool, mirrors scan_bad_list.csv
        self._log_csv_stat = None      # (st_mtime, st_size) guard
        self.runtable_window = None    # non-blocking viewer, kept alive on self
        self._runtable_updating = False  # guard against itemChanged while rebuilding

        self.treeWidget_fileStructure.installEventFilter(self)

        self._initialize_empty_data_containers()
        self._set_scan_watcher_ui('gray')
        self._setup_tree()
        self._connect_signals()
        self._setup_pyqtgraph_view()
        self.restore_window_size()
        self.load_base_path()
        self.on_base_path_entered()
        self._set_log_csv_ui()

        self.treeWidget_fileStructure.setContextMenuPolicy(Qt.CustomContextMenu)
        self.treeWidget_fileStructure.customContextMenuRequested.connect(self.on_tree_right_click)


    # ------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------

    def _initialize_empty_data_containers(self):
        self._scan_row_items = {}     # scan_name -> QTreeWidgetItem
        self._seen_scans = set()      # just scan names
        self._seen_param_folders = {} # scan_name -> set(param_folder_paths)
        self._seen_recon_files = {}   # scan_name -> {param_folder_path -> set(recon_file_paths)}


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
        self.pushButton_populateTree.clicked.connect(self.on_populate_tree_clicked)
        self.pushButton_saveTree.clicked.connect(self.on_save_tree_clicked)
        self.pushButton_loadTree.clicked.connect(self.on_load_tree_clicked)
        self.treeWidget_fileStructure.itemClicked.connect(self.on_tree_item_clicked)
        self.treeWidget_fileStructure.currentItemChanged.connect(self.on_tree_selection_changed)
        self.pushButton_stopScanUpdate.clicked.connect(self.on_stop_scan_update)
        self.pushButton_updateScanGoodness.clicked.connect(self.on_update_scan_goodness)
        self.toolButton_tips.clicked.connect(self.show_secret_features)
        self.pushButton_addScan.clicked.connect(self.on_add_scan_clicked)
        self.pushButton_viewRuntable.clicked.connect(self.show_runtable_window)


    def _setup_pyqtgraph_view(self):
        """
        Embed the reusable ImagePlotWidget into the placeholder widget and
        register the browser-specific right-click menu actions.

        Everything generic (measurement, cursor readout, lineout + resolution
        metric, filters, zoom) lives in pg_image_tools; only the actions that
        reach into this app's tree/file state are wired up here.
        """
        self.plot = ImagePlotWidget(
            self.graphicsView_1,
            info_label=self.label_plot_info,
            title_label=self.label_plot_1,
            transpose_checkbox=self.checkBox_transpose,
            log_checkbox=self.checkBox_logCmap,
        )
        # App-specific menu entries, above the widget's own block
        self.plot.add_menu_action(
            "Copy Param Folder Path", self._copy_current_param_path, at_top=True
        )
        self.plot.add_menu_action(
            "Copy Absolute File Path", self._copy_current_abs_file_path, at_top=True
        )
        self.plot.add_menu_separator(at_top=True)

        # "Full Probe Zoom" toggle — when checked, skip the square crop zoom
        self._full_probe_zoom_action = self.plot.add_menu_action(
            "Full Probe Zoom", checkable=True,
            after=self.plot.action_auto_reset_zoom,
        )


    def _copy_current_param_path(self):
        item = self.treeWidget_fileStructure.currentItem()
        if item is None:
            return
        param_path = Path(item.data(1, Qt.UserRole)).name
        QApplication.clipboard().setText(str(param_path))

    def _copy_current_abs_file_path(self):
        if self.file_load_path is None:
            return
        QApplication.clipboard().setText(str(self.file_load_path))


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


    def save_base_path(self):
        settings = QSettings("temp", "PtychiFileBrowser")
        settings.setValue("last_base_path", str(self.base_path))


    def load_base_path(self):
        settings = QSettings("temp", "PtychiFileBrowser")
        last_path = settings.value("last_base_path", defaultValue="/mnt/micdata2/")
        if last_path:
            self.lineEdit_basePath.setText(last_path)
            
        
    def restore_window_size(self):
        settings = QSettings("temp", "PtychiFileBrowser")

        geom = settings.value("window_geometry")
        if geom is not None:
            self.restoreGeometry(geom)

        state = settings.value("window_state")
        if state is not None:
            self.restoreState(state)


    def show_secret_features(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Hidden Features")

        layout = QtWidgets.QVBoxLayout(dlg)

        edit = QtWidgets.QTextEdit()
        edit.setReadOnly(True)
        edit.setPlainText(f"""
        Hidden Features / Shortcuts
                          
        Populating
        --- Set Base Path and click Populate tree to start
        --- Populate tree is slow, so afterwards try to add each scan manually or use scan auto updater

        Tree Navigation
        --- Up / Down    → switch scan number
        --- Left / Right → switch recon file
        --- , / .        → switch parameter folder
        --- k / l        → switch file type (for example: object_ph or probe_mag)
        --- Right click column 0 → refresh scan
        --- Right click column 1 → switch parameter folder
        --- Right click column 2 → switch recon file
                          
        Plot
        --- Click on two points to get distance and lineout
        --- Open lineout viewer with right-click on plot
        ----- Lineout viewer can be used to find the 25%-75% resolution
        ----- Drag the blue and green lines to the boundaries of a hard edge, and the right-click menu will calculate it
        --- By default, probe viewer is centered on mode 0, and the right-click menu can turn this off
        --- Right-click menu can copy parameter folder string (Ndp256...)
        --- Right-click menu can change default zoom behavior ("View All" resets the zoom)

        Scan goodness
        --- Row color shows scan goodness, tracked as txt file in scan folder
        --- Green is a good ptycho recon, yellow marks a scan to reanalyze, red is bad data
        
        Sample names
        --- Taken from the log csv first, then the sources below, then a dash
        --- Pulled from file 'runtable_full_{self.base_path.parent.name}.csv'
        --- File must be located in parent of base path
        --- i.e. {self.base_path.parent}
        --- Searches for scan number in column 'run', and returns corresponding string from 'sample_name'

        Log CSV / Runtable
        --- On Populate tree or Load tree, searches parent of base path for .csv files
        --- i.e. {self.base_path.parent}
        --- If more than one is found, a popup asks which to use, and the choice is remembered
        --- Must have a 'scan' column of four-digit run numbers, i.e. S0001, or all of this is skipped
        --- Duplicate scan numbers use the last row in the file
        --- Column 'sample_name' feeds the sample name column, before the runtable and file name guesses
        --- Re-read when a brand new scan is added, so scans measured after startup still get info
        --- File is only ever opened read only, so it never blocks whatever is writing to it
        --- "View Runtable" opens a filtered, color-coded table in its own window
        ----- Shows scan, completed, date, time, ExpTime, n_pos, phi, scan_type
        ----- Then one column per entry in 'scan_step_size', motors in microns, time in ms, phi in deg
        ----- Row color starts from scan goodness, then pale green if a recon file exists in the tree
        ----- or yellow if it does not, and finally red if 'completed' is no
        ----- Far right "bad" checkbox is ticked for every red row, and ticking one turns its row red
        ----- Ticks are saved to '{SCAN_BAD_LIST_FILENAME}' next to the log csv, as columns 'scan' and 'is bad'
        ----- That file is written on populate, on refresh, and when either window is closed
        ----- An 'is bad' of yes forces a row red on reload, so untick a row to take the mark back off

        Scan Auto Updater
        --- Every 10.0 s, checks a file {self.base_path / "recon_completed.csv"}
        --- Looks for new rows since last check, in the form S0001, i.e. four-digit run number
        --- Refreshes every scan number it finds
        --- Code to add to ptychi reconstruction script just after ptychi handles the reconstruction:
        import csv
        # append row to log file
        with open(os.path.join(data_main_dir, 'ptychi_recons', 'recon_completed.csv'), "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["S%04d" % scan_num])

        Deleting
        !!!These are irreversible, handle with caution!!!
        --- Right-click menu of each scan has several delete options, they all pop up a confirmation dialog
        ----- "Delete all reconstructions" deletes the entire scan folder
        ----- "Delete currently selected reconstruction" deletes entire parameter folder (e.g. starts with Ndp256)
        ----- "Delete intermediate reconstructions" deletes all files associated with intermediate iterations
        ------- Only targets current parameter folder, and will list every file before deletion
        --- Tips menu has "Delete intermediate reconstructions tool" button
        ----- Loops through every parameter folder within scan range, deleting all intermediate iteration files
        """.strip())

        layout.addWidget(edit)

        btn_row = QtWidgets.QHBoxLayout()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        fontSize_label = QtWidgets.QLabel("Font Size:")
        self.comboBox_fontSize = QtWidgets.QComboBox()
        self.comboBox_fontSize.addItems([str(s) for s in [6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24]])
        self.comboBox_fontSize.setCurrentText("10")
        self.comboBox_fontSize.currentTextChanged.connect(self.on_font_size_changed)
        del_tool_btn = QtWidgets.QPushButton("Delete intermediate reconstructions tool")
        del_tool_btn.setStyleSheet("background-color: red; color: white;")
        del_tool_btn.clicked.connect(lambda: (dlg.accept(), self.show_delete_intermediate_tool()))
        btn_row.addWidget(close_btn)
        btn_row.addWidget(fontSize_label)
        btn_row.addWidget(self.comboBox_fontSize)
        btn_row.addWidget(del_tool_btn)
        layout.addLayout(btn_row)

        dlg.resize(700, 400)
        dlg.exec()


    def on_font_size_changed(self, size_str):
        """
        Change the font size of every widget in the entire GUI (all open windows/dialogs).
        """
        try:
            size = int(size_str)
        except ValueError:
            return

        app = QApplication.instance()

        app_font = app.font()
        app_font.setPointSize(size)
        app.setFont(app_font)

        for widget in app.allWidgets():
            widget_font = widget.font()
            widget_font.setPointSize(size)
            widget.setFont(widget_font)


    def on_add_scan_clicked(self):
        """
        Show a dialog to add one scan or a range of scans.
        """
        # Find the maximum scan number from _seen_scans
        max_scan_num = 0
        for scan_name in self._seen_scans:
            if len(scan_name) == 5 and scan_name.startswith("S") and scan_name[1:].isdigit():
                max_scan_num = max(max_scan_num, int(scan_name[1:]))

        # Create dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Add Scan")
        layout = QtWidgets.QVBoxLayout(dlg)

        # --- First scan combo ---
        layout.addWidget(QtWidgets.QLabel("Select or enter scan name:"))
        combo_first = QtWidgets.QComboBox()
        combo_first.setEditable(True)
        for idx in range(1, 11):
            combo_first.addItem(f"S{max_scan_num + idx:04d}")
        layout.addWidget(combo_first)

        # --- Range row: checkbox + second combo ---
        range_row = QtWidgets.QHBoxLayout()
        chk_range = QtWidgets.QCheckBox("Enter range")
        combo_last = QtWidgets.QComboBox()
        combo_last.setEditable(True)
        for idx in range(1, 11):
            combo_last.addItem(f"S{max_scan_num + idx:04d}")
        combo_last.setEnabled(False)
        range_row.addWidget(chk_range)
        range_row.addWidget(combo_last)
        layout.addLayout(range_row)

        chk_range.toggled.connect(combo_last.setEnabled)

        # --- OK / Cancel ---
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dlg.accept)
        button_box.rejected.connect(dlg.reject)
        layout.addWidget(button_box)

        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return None

        def parse_scan(text):
            """Return scan number int if valid S#### format, else None."""
            t = text.strip()
            if len(t) == 5 and t.startswith("S") and t[1:].isdigit():
                return int(t[1:])
            return None

        first_num = parse_scan(combo_first.currentText())
        if first_num is None:
            QtWidgets.QMessageBox.warning(self, "Invalid Format",
                f"Invalid scan name: {combo_first.currentText()}\nExpected format: S#### (e.g., S0042)")
            return None

        if chk_range.isChecked():
            last_num = parse_scan(combo_last.currentText())
            if last_num is None:
                QtWidgets.QMessageBox.warning(self, "Invalid Format",
                    f"Invalid scan name: {combo_last.currentText()}\nExpected format: S#### (e.g., S0042)")
                return None
            scan_nums = range(min(first_num, last_num), max(first_num, last_num) + 1)
        else:
            scan_nums = [first_num]

        # Only pay for a re-read if at least one of these is a brand new row
        if any(f"S{num:04d}" not in self._scan_row_items for num in scan_nums):
            self.refresh_log_csv_for_new_scan()

        missing = []
        for num in scan_nums:
            scan_name = f"S{num:04d}"
            scan_path = self.base_path / scan_name
            if scan_path.exists():
                self._add_scan_row(scan_path)
                print(f"Added scan: {scan_name}")
            else:
                missing.append(str(scan_path))

        self.treeWidget_fileStructure.sortItems(0, Qt.AscendingOrder)

        if missing:
            QtWidgets.QMessageBox.warning(self, "Scan Not Found",
                "The following scan folders were not found:\n" + "\n".join(missing))

        return None


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
        color = GOODNESS_COLORS.get(goodness, QBrush())
        for col in range(self.treeWidget_fileStructure.columnCount()):
            row_item.setBackground(col, color)


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
        
        self.scan_watcher = ScanWatcherThread(Path(self.base_path), poll_interval=10.0)
        self.scan_watcher.scan_found.connect(self.on_scan_found)
        self.scan_watcher.finished_adding_scans.connect(self.on_finished_adding_scans)
        self.scan_watcher.start()


        self._set_scan_watcher_ui('running')

        
    def on_scan_found(self, scan_path: Path):
        print(f"New scan detected: {scan_path.name}")

        if scan_path.name in self._scan_row_items:
            self._refresh_scan_row(scan_path)   # existing → update in place
        else:
            self.refresh_log_csv_for_new_scan()
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
        settings = QSettings("temp", "PtychiFileBrowser")

        self.save_scan_bad_list()

        settings.setValue("window_geometry", self.saveGeometry())
        settings.setValue("window_state", self.saveState())

        if getattr(self, "scan_watcher", None) is not None:
            self.scan_watcher.stop()
            self.scan_watcher.wait()

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
        Tries the log csv, then the runtable, then falls back to parsing a
        filename directly.
        """
        try:
            scan_num = int(scan_path.name[1:])
        except ValueError:
            return None

        # --- log csv lookup ---
        log_name = self.get_log_csv_sample_name(scan_path.name)
        if log_name:
            return log_name

        # --- runtable lookup ---
        if self.runtable_df is not None:
            if "run" in self.runtable_df.columns and "sample_name" in self.runtable_df.columns:
                match = self.runtable_df[self.runtable_df["run"] == scan_num]
                if not match.empty:
                    value = match.iloc[0]["sample_name"]
                    if pd.notna(value):
                        return str(value)

        # --- filesystem fallback ---
        base_dir = self.base_path.parent
        run_dir = base_dir / f"ptycho/{scan_num:03d}"
        if not run_dir.is_dir():
            run_dir = base_dir / f"ptycho/S{scan_num:04d}"
        if not run_dir.is_dir():
            return None

        hits = list(run_dir.glob("*00001.h5"))
        if not hits:
            return None

        stem = hits[0].stem
        parts = stem.split("_")
        for i, p in enumerate(parts):
            if p.endswith(f"{scan_num:03d}"):
                sample_name = ("_".join(parts[:i]) + "_" + p[:-3]) if i > 0 else p[:-3]
                print(sample_name)
                return sample_name or None

        return None


    # ------------------------------------------------------------------
    # log csv
    # ------------------------------------------------------------------

    def _set_log_csv_ui(self):
        """Enable the runtable viewer only when a usable log csv is loaded."""
        self.pushButton_viewRuntable.setEnabled(bool(self.log_csv_ok))


    def _clear_log_csv_state(self):
        self.log_csv_path = None
        self.log_csv_df = None
        self.log_csv_ok = False
        self._log_sample_by_scan = {}
        self._log_bad_by_scan = {}
        self._log_csv_stat = None


    def _log_csv_stat_tuple(self, path: Path):
        """(mtime, size) of the csv, or None if it cannot be stat'ed."""
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (st.st_mtime, st.st_size)


    def discover_log_csv(self):
        """
        Find the log csv in the parent of base_path.

        Reuses the remembered choice when it still applies, otherwise asks
        the user if more than one candidate exists.

        Returns
        -------
        Path or None
        """
        if self.base_path is None:
            return None

        settings = QSettings("temp", "PtychiFileBrowser")
        remembered_base = settings.value("log_csv_base_path")
        remembered_csv = settings.value("log_csv_path")
        if remembered_base == str(self.base_path) and remembered_csv:
            remembered_csv = Path(remembered_csv)
            if remembered_csv.exists():
                return remembered_csv

        try:
            # Skip the bad list we write ourselves - it has a 'scan' column
            # too, so it would otherwise pass as a log csv
            candidates = sorted(
                p for p in self.base_path.parent.glob("*.csv")
                if p.name not in (SCAN_BAD_LIST_FILENAME, TREE_CACHE_FILENAME)
            )
        except OSError as exc:
            print(f"Failed to search for log csv: {exc}")
            return None

        if not candidates:
            return None

        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            chosen = self._ask_which_log_csv(candidates)
            if chosen is None:
                return None

        settings.setValue("log_csv_base_path", str(self.base_path))
        settings.setValue("log_csv_path", str(chosen))
        return chosen


    def _ask_which_log_csv(self, candidates):
        """Popup asking which of several csv files to use. None if cancelled."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Select Log CSV")
        layout = QtWidgets.QVBoxLayout(dlg)

        layout.addWidget(QtWidgets.QLabel(
            f"Multiple csv files found in:\n{self.base_path.parent}\n\nSelect the run log to use:"
        ))

        combo = QtWidgets.QComboBox()
        for path in candidates:
            combo.addItem(path.name)
        layout.addWidget(combo)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dlg.accept)
        button_box.rejected.connect(dlg.reject)
        layout.addWidget(button_box)

        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return None

        return candidates[combo.currentIndex()]


    def _rebuild_scan_lookup(self, df):
        """
        Build {scan_name -> sample name or None} from a log csv DataFrame.

        Rows are walked in file order so a duplicate scan number keeps the
        last (highest row number) entry.
        """
        lookup = {}
        scans = df[LOG_CSV_SCAN_COL]
        samples = df[LOG_CSV_SAMPLE_COL] if LOG_CSV_SAMPLE_COL in df.columns else None

        for i in range(len(df)):
            scan_key = str(scans.iloc[i]).strip()
            if not scan_key:
                continue

            sample = None
            if samples is not None:
                value = samples.iloc[i]
                if pd.notna(value):
                    sample = str(value).strip() or None

            lookup[scan_key] = sample

        return lookup


    def load_log_csv(self):
        """
        Locate and fully read the log csv, setting self.log_csv_ok.

        Any failure (no file, unreadable, no 'scan' column) leaves the flag
        False so every log csv feature is skipped. The file is only opened
        read-only and closed immediately, so it is never locked.

        Returns
        -------
        bool
        """
        self._clear_log_csv_state()

        if self.base_path is None:
            self._set_log_csv_ui()
            return False

        csv_path = self.discover_log_csv()
        if csv_path is None:
            self._set_log_csv_ui()
            return False

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"Failed to load log csv: {exc}")
            self._set_log_csv_ui()
            return False

        if LOG_CSV_SCAN_COL not in df.columns:
            print(f"Log csv {csv_path.name} has no '{LOG_CSV_SCAN_COL}' column, skipping")
            self._set_log_csv_ui()
            return False

        self.log_csv_path = csv_path
        self.log_csv_df = df
        self._log_sample_by_scan = self._rebuild_scan_lookup(df)
        self._log_csv_stat = self._log_csv_stat_tuple(csv_path)
        self.log_csv_ok = True

        self.load_scan_bad_list()

        self._set_log_csv_ui()
        print(f"Loaded log csv from {csv_path}")
        return True


    def get_log_csv_sample_name(self, scan_name: str):
        """Sample name for an S#### scan from the log csv, or None."""
        if not self.log_csv_ok:
            return None
        return self._log_sample_by_scan.get(scan_name)


    # ------------------------------------------------------------------
    # scan bad list
    # ------------------------------------------------------------------

    @property
    def scan_bad_list_path(self):
        """Path of scan_bad_list.csv, alongside the log csv."""
        if self.log_csv_path is None:
            return None
        return self.log_csv_path.parent / SCAN_BAD_LIST_FILENAME


    def load_scan_bad_list(self):
        """
        Read scan_bad_list.csv into self._log_bad_by_scan. A missing or
        unreadable file just leaves the mapping empty; it gets created by
        the next save.
        """
        self._log_bad_by_scan = {}

        path = self.scan_bad_list_path
        if path is None or not path.exists():
            return

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"Failed to load {SCAN_BAD_LIST_FILENAME}: {exc}")
            return

        if SCAN_BAD_SCAN_COL not in df.columns or SCAN_BAD_FLAG_COL not in df.columns:
            print(f"{SCAN_BAD_LIST_FILENAME} has unexpected columns, ignoring")
            return

        for scan, flag in zip(df[SCAN_BAD_SCAN_COL], df[SCAN_BAD_FLAG_COL]):
            scan_key = str(scan).strip()
            if scan_key:
                self._log_bad_by_scan[scan_key] = str(flag).strip().lower() == "yes"


    def _log_completed_by_scan(self):
        """{scan -> completed} from the full log csv frame, last duplicate wins."""
        completed = {}
        df = self.log_csv_df
        if df is None or "completed" not in df.columns:
            return completed

        for scan, value in zip(df[LOG_CSV_SCAN_COL], df["completed"]):
            scan_key = str(scan).strip()
            if scan_key:
                completed[scan_key] = None if pd.isna(value) else value
        return completed


    def _refresh_bad_state_from_colors(self):
        """
        Re-derive 'is bad' for every scan in the log csv from its current
        runtable color, so the saved list always mirrors the checkboxes.
        """
        completed = self._log_completed_by_scan()
        for scan_key in self._log_sample_by_scan:
            color = self._runtable_row_color(
                scan_key,
                completed.get(scan_key),
                marked_bad=self._log_bad_by_scan.get(scan_key, False),
            )
            self._log_bad_by_scan[scan_key] = color == GOODNESS_COLORS['bad']


    def save_scan_bad_list(self):
        """
        Write scan_bad_list.csv next to the log csv, creating it if it does
        not exist yet. Called after the tree is built, when the runtable is
        refreshed or closed, and when the main window closes.
        """
        if not self.log_csv_ok:
            return

        path = self.scan_bad_list_path
        if path is None:
            return

        self._refresh_bad_state_from_colors()

        rows = [
            {SCAN_BAD_SCAN_COL: scan_key,
             SCAN_BAD_FLAG_COL: "yes" if self._log_bad_by_scan.get(scan_key) else "no"}
            for scan_key in self._log_sample_by_scan
        ]

        try:
            pd.DataFrame(rows, columns=[SCAN_BAD_SCAN_COL, SCAN_BAD_FLAG_COL]).to_csv(
                path, index=False
            )
        except Exception as exc:
            print(f"Failed to save {SCAN_BAD_LIST_FILENAME}: {exc}")
            return

        print(f"Saved scan bad list to {path}")


    def refresh_log_csv_for_new_scan(self):
        """
        Cheap re-read used when a brand new scan row is added, so scans
        measured after the csv was last read still pick up their info.

        Stats the file first and does no I/O at all when it has not changed.
        Otherwise only the scan and sample_name columns are parsed; the full
        frame behind the runtable viewer is refreshed only if that window is
        currently open.
        """
        if not self.log_csv_ok or self.log_csv_path is None:
            return

        stat = self._log_csv_stat_tuple(self.log_csv_path)
        if stat is None or stat == self._log_csv_stat:
            return

        viewer_open = self.runtable_window is not None and self.runtable_window.isVisible()

        try:
            if viewer_open:
                df = pd.read_csv(self.log_csv_path)
            else:
                df = pd.read_csv(
                    self.log_csv_path,
                    usecols=lambda c: c in (LOG_CSV_SCAN_COL, LOG_CSV_SAMPLE_COL),
                )
        except Exception as exc:
            print(f"Failed to re-read log csv: {exc}")
            return

        if LOG_CSV_SCAN_COL not in df.columns:
            return

        self._log_sample_by_scan = self._rebuild_scan_lookup(df)
        self._log_csv_stat = stat

        if viewer_open:
            self.log_csv_df = df
            self._rebuild_runtable_table()


    def reload_log_csv_full(self):
        """Full re-read of the already-chosen log csv (viewer Refresh button)."""
        if not self.log_csv_ok or self.log_csv_path is None:
            return

        try:
            df = pd.read_csv(self.log_csv_path)
        except Exception as exc:
            print(f"Failed to reload log csv: {exc}")
            return

        if LOG_CSV_SCAN_COL not in df.columns:
            return

        self.log_csv_df = df
        self._log_sample_by_scan = self._rebuild_scan_lookup(df)
        self._log_csv_stat = self._log_csv_stat_tuple(self.log_csv_path)


    # ------------------------------------------------------------------
    # runtable viewer
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_scan_step_size(text):
        """
        Parse a scan_step_size entry into {name: value in display units}.

        The entry is a space separated list of (name, step size) pairs, e.g.
        "X 0.000500 Z 0.000500" -> {"X": 0.5, "Z": 0.5}. Motors are stored in
        mm and shown in microns; the exceptions in LOG_CSV_STEP_UNITS ('time'
        in seconds shown as ms, 'phi' already in degrees) get their own
        conversion. Malformed pairs are skipped rather than raising.
        """
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return {}

        tokens = str(text).split()
        steps = {}
        for i in range(0, len(tokens) - 1, 2):
            name = tokens[i]
            _, scale = LOG_CSV_STEP_UNITS.get(name, LOG_CSV_STEP_DEFAULT_UNIT)
            try:
                steps[name] = float(tokens[i + 1]) * scale
            except ValueError:
                continue
        return steps


    @staticmethod
    def _step_column_header(name):
        """Column header for one scan_step_size entry, e.g. 'X (um)'."""
        unit, _ = LOG_CSV_STEP_UNITS.get(name, LOG_CSV_STEP_DEFAULT_UNIT)
        return f"{name} ({unit})"


    def _runtable_columns(self, df):
        """
        Return (headers, step_motors) for the viewer.

        headers is the subset of LOG_CSV_VIEW_COLS present in the file, in
        that order, followed by one column per name found in scan_step_size
        (first-seen order).
        """
        headers = [c for c in LOG_CSV_VIEW_COLS if c in df.columns]

        step_motors = []
        if LOG_CSV_STEP_COL in df.columns:
            for value in df[LOG_CSV_STEP_COL]:
                for motor in self._parse_scan_step_size(value):
                    if motor not in step_motors:
                        step_motors.append(motor)

        return headers + [self._step_column_header(m) for m in step_motors], step_motors


    def _runtable_row_color(self, scan_key, completed_value, marked_bad=False):
        """
        Row color for the runtable, applied in increasing precedence:
        scan goodness, then whether a recon file exists, then not completed,
        and finally the manual 'bad' checkbox / scan_bad_list.csv. Note that
        an 'is bad' of no never clears a color the other rules produced.
        """
        row_item = self._scan_row_items.get(scan_key)

        color = None
        if row_item is not None:
            goodness = row_item.data(0, Qt.UserRole + 1)
            color = GOODNESS_COLORS.get(goodness)

        # No recon file in the tree (including scans absent from the tree)
        if row_item is None or not isinstance(row_item.data(2, Qt.UserRole), Path):
            color = GOODNESS_COLORS['reanalyze']
        elif color is None:
            # Analyzed, but no scan goodness set yet
            color = RECON_EXISTS_COLOR

        if completed_value is not None and str(completed_value).strip().lower() == "no":
            color = GOODNESS_COLORS['bad']

        if marked_bad:
            color = GOODNESS_COLORS['bad']

        return color


    def _rebuild_runtable_table(self):
        """Refill the runtable viewer's table from self.log_csv_df."""
        table = self.tableWidget_runtable
        df = self.log_csv_df

        self.label_runtablePath.setText(str(self.log_csv_path or ""))
        self.label_runtablePath.setToolTip(str(self.log_csv_path or ""))

        self._runtable_updating = True
        try:
            table.setSortingEnabled(False)
            table.clear()

            if df is None:
                table.setRowCount(0)
                table.setColumnCount(0)
                return

            headers, step_motors = self._runtable_columns(df)
            plain_cols = len(headers) - len(step_motors)
            bad_col = len(headers)

            table.setColumnCount(len(headers) + 1)
            table.setHorizontalHeaderLabels(headers + [RUNTABLE_BAD_HEADER])
            table.setRowCount(len(df))

            completed = df["completed"] if "completed" in df.columns else None
            scans = df[LOG_CSV_SCAN_COL]

            for row in range(len(df)):
                for col in range(plain_cols):
                    value = df[headers[col]].iloc[row]
                    table.setItem(row, col, self._make_runtable_item(value))

                if step_motors:
                    steps = self._parse_scan_step_size(df[LOG_CSV_STEP_COL].iloc[row])
                    for offset, motor in enumerate(step_motors):
                        table.setItem(row, plain_cols + offset,
                                      self._make_runtable_item(steps.get(motor)))

                scan_key = str(scans.iloc[row]).strip()
                completed_value = None if completed is None else completed.iloc[row]
                if completed_value is not None and pd.isna(completed_value):
                    completed_value = None

                color = self._runtable_row_color(
                    scan_key,
                    completed_value,
                    marked_bad=self._log_bad_by_scan.get(scan_key, False),
                )
                is_bad = color == GOODNESS_COLORS['bad']
                self._log_bad_by_scan[scan_key] = is_bad

                table.setItem(row, bad_col,
                              self._make_runtable_bad_item(scan_key, completed_value, is_bad))
                self._color_runtable_row(row, color)

            table.setSortingEnabled(True)
            table.resizeColumnsToContents()
        finally:
            self._runtable_updating = False


    def _color_runtable_row(self, row, color):
        """Paint every cell of one runtable row, including the checkbox cell."""
        table = self.tableWidget_runtable
        brush = color if color is not None else QBrush()
        for col in range(table.columnCount()):
            item = table.item(row, col)
            if item is not None:
                item.setBackground(brush)


    def _make_runtable_bad_item(self, scan_key, completed_value, is_bad):
        """
        The 'bad' cell: a bare checkbox. The scan and its completed value ride
        along on the item so a toggle still works after the table is sorted.
        """
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if is_bad else Qt.Unchecked)
        item.setTextAlignment(Qt.AlignCenter)
        item.setData(Qt.UserRole, scan_key)
        item.setData(Qt.UserRole + 1,
                     None if completed_value is None else str(completed_value))
        return item


    def _on_runtable_item_changed(self, item):
        """
        Handle a manual toggle of the 'bad' checkbox: recolor the row and
        remember the choice. Checking always turns the row red; unchecking
        only clears it if no other rule was making it red, in which case the
        box snaps back so it always agrees with the color.
        """
        if self._runtable_updating:
            return
        if item.column() != self.tableWidget_runtable.columnCount() - 1:
            return

        scan_key = item.data(Qt.UserRole)
        checked = item.checkState() == Qt.Checked

        color = self._runtable_row_color(
            scan_key, item.data(Qt.UserRole + 1), marked_bad=checked
        )
        is_bad = color == GOODNESS_COLORS['bad']
        self._log_bad_by_scan[scan_key] = is_bad

        self._runtable_updating = True
        try:
            if is_bad != checked:
                item.setCheckState(Qt.Checked if is_bad else Qt.Unchecked)
            self._color_runtable_row(item.row(), color)
        finally:
            self._runtable_updating = False


    @staticmethod
    def _make_runtable_item(value):
        """
        Table cell for one value. Numbers go in as numbers so that sorting
        by a numeric column is numeric rather than lexicographic.
        """
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return item

        if isinstance(value, (int, float, np.integer, np.floating)):
            item.setData(Qt.EditRole, float(value))
        else:
            item.setData(Qt.EditRole, str(value))

        return item


    def show_runtable_window(self):
        """
        Open the runtable viewer: a non-blocking, color-coded view of the
        log csv. The window is created once and reused so it can refresh
        itself while open.
        """
        if not self.log_csv_ok:
            return

        if self.runtable_window is None:
            dlg = RuntableWindow(self, self.save_scan_bad_list)
            dlg.setWindowTitle("Runtable")

            layout = QtWidgets.QVBoxLayout(dlg)

            self.label_runtablePath = QtWidgets.QLabel()
            self.label_runtablePath.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(self.label_runtablePath)

            self.tableWidget_runtable = QtWidgets.QTableWidget()
            self.tableWidget_runtable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.tableWidget_runtable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.tableWidget_runtable.verticalHeader().setVisible(False)
            self.tableWidget_runtable.itemChanged.connect(self._on_runtable_item_changed)
            layout.addWidget(self.tableWidget_runtable)

            btn_row = QtWidgets.QHBoxLayout()
            close_btn = QtWidgets.QPushButton("Close")
            close_btn.clicked.connect(dlg.close)
            refresh_btn = QtWidgets.QPushButton("Refresh")
            refresh_btn.clicked.connect(self.on_runtable_refresh)
            btn_row.addWidget(close_btn)
            btn_row.addWidget(refresh_btn)
            btn_row.addStretch()
            layout.addLayout(btn_row)

            dlg.resize(1000, 600)
            self.runtable_window = dlg

        self._rebuild_runtable_table()
        self.runtable_window.show()
        self.runtable_window.raise_()
        self.runtable_window.activateWindow()


    def on_runtable_refresh(self):
        """Persist the current checkboxes, then re-read the log csv and rebuild."""
        self.save_scan_bad_list()
        self.reload_log_csv_full()
        self._rebuild_runtable_table()


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
        self._positions_px = None  # reset on every load

        if extension == 'recon_NiterXXX.h5':
            self.file_load_path = file_path

        elif extension in ('recon_NiterXXX_ph.h5', 'recon_NiterXXX_mag.h5', 'recon_NiterXXX_pos.h5'):
            variant = extension[len('recon_NiterXXX'):]   # e.g. "_ph.h5"
            self.file_load_path = file_path
            # self.file_load_path = file_path.parent / f"{file_path.stem}{variant}"

        elif extension in ('dp_sum.tiff', 'init_probe_mag.tiff'):
            self.file_load_path = file_path.parent / extension

        elif extension == 'init_positions.png':
            self.file_load_path = file_path.parent / extension

        else:
            base = extension.rsplit("Niter", 1)[0]
            suffix = file_path.stem.split("recon_", 1)[1]
            self.file_load_path = file_path.parent / f"{base}{suffix}{Path(extension).suffix}"

        if not self.file_load_path.exists():
            return None

        if self.file_load_path.suffix in ('.h5', '.hdf5'):
            with h5py.File(self.file_load_path, 'r') as f:
                if extension == 'recon_NiterXXX_mag.h5':
                    obj = np.abs(f['object'][0][()]).T
                else:
                    obj = np.angle(f['object'][0][()]).T
                self.res_m = float(f['obj_pixel_size_m'][()])
                if extension == 'recon_NiterXXX_pos.h5' and 'positions_px' in f:
                    self._positions_px = f['positions_px'][()]

        elif self.file_load_path.suffix in ('.tiff',):
            with tifffile.TiffFile(self.file_load_path) as tif:
                obj = tif.asarray()
                if 'pixel_size' in tif.imagej_metadata.keys():
                    self.res_m = 1e-6 * tif.imagej_metadata['pixel_size']

                elif 'xspacing' in tif.imagej_metadata.keys():
                    self.res_m = 1e-6 * tif.imagej_metadata['xspacing']

                if len(obj.shape) == 3:
                    obj = np.mean(obj, 2).T

                if 'object_' in self.file_load_path.stem:
                    obj = obj.T

        elif self.file_load_path.suffix in ('.png',):
            obj = Image.open(self.file_load_path).convert("L")  # L = grayscale
            obj = np.array(obj, dtype=np.float32).T

        return obj


    # ------------------------------------------------------------------
    # plotting
    # ------------------------------------------------------------------

    def display_data(self, data: np.ndarray, scan: int, sample_name: str):
        """
        Hand a 2D numpy array to the plot widget.

        Everything below this point (transpose, filters, log scale, overlays,
        readouts) is the plot widget's business — see pg_image_tools.
        """
        if data.ndim != 2:
            raise ValueError("display_data expects a 2D numpy array")

        self.plot.set_title(f"{scan}, {sample_name}\n{self.file_load_path}")
        self.plot.set_image(data, pixel_size_m=self.res_m)


    # New handler
    def on_tree_selection_changed(self, current: QtWidgets.QTreeWidgetItem, previous: QtWidgets.QTreeWidgetItem):
        if current is not None:
            # Determine the column you want to act on; e.g., column 2 for recon file
            self.on_tree_item_clicked(current, 2)

    
    def on_tree_item_clicked(self, item, column):
        """
        Handle clicks anywhere on a scan row.
        """
    
        # Try recon file first; fall back to param folder (col 1) for non-recon choices
        file_path = item.data(2, Qt.UserRole)

        if not isinstance(file_path, Path):
            param_path = item.data(1, Qt.UserRole)
            if not isinstance(param_path, Path) or not param_path.exists():
                return
            # Synthetic path: parent = param_path; stem is irrelevant for tiff/png choices
            file_path = param_path / "_"

        data = self.load_data_from_file(file_path)
        if data is None:
            return

        self.display_data(data, item.text(0), item.text(3))
        self._apply_probe_zoom()
        self._update_positions_overlay()
        self.update_scan_goodness_ui(item.data(0, Qt.UserRole + 1))


    def _apply_probe_zoom(self):
        """For probe images (wide+short), zoom to the leftmost H×H square."""
        if self._full_probe_zoom_action.isChecked():
            return
        extension = self.comboBox_imageChoice.currentText()
        if extension != 'init_probe_mag.tiff' and not extension.startswith('probe_mag'):
            return
        # Leftmost square: x in [0, H], y in [0, H]
        self.plot.zoom_to_left_square()


    def _update_positions_overlay(self):
        """Show scan positions as a red scatter overlay (only for _pos choice)."""
        pos = getattr(self, '_positions_px', None)
        if pos is None or len(pos) == 0:
            self.plot.clear_scatter_overlay()
            return
        # positions_px is (N, 2) in (row, col) = (y, x) convention stored as
        # offsets from center; shift to image center in pyqtgraph (axis 0 = x, axis 1 = y)
        nx, ny = self.plot.displayed_shape or (0, 0)
        self.plot.set_scatter_overlay(x=pos[:, 1] + nx / 2, y=pos[:, 0] + ny / 2)


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
            remove_action = menu.addAction("Remove scan from list")
            remove_action.triggered.connect(
                lambda checked, n=scan_name, i=item: self._remove_scan_row(n, i)
            )

            # "Delete Reconstructions" submenu with red text via QWidgetAction
            delete_menu = QtWidgets.QMenu(menu)
            delete_menu.setStyleSheet(
                "QMenu::item { color: darkred; }"
                "QMenu::item:selected { background-color: #8b0000; color: white; }"
            )
            scan_path = self.base_path / scan_name
            del_all_action = delete_menu.addAction(f"Delete all reconstructions for '{scan_name}'")
            del_all_action.triggered.connect(
                lambda checked, n=scan_name, p=scan_path, i=item: self._delete_scan_folder(n, p, i)
            )
            param_path = item.data(1, Qt.UserRole)
            if isinstance(param_path, Path):
                del_current_action = delete_menu.addAction(
                    f"Delete currently selected reconstruction: '{param_path.name}'"
                )
                del_current_action.triggered.connect(
                    lambda checked, n=scan_name, p=param_path, i=item: self._delete_param_folder(n, p, i)
                )
                del_inter_action = delete_menu.addAction(
                    f"Delete intermediate reconstructions: '{param_path.name}'"
                )
                del_inter_action.triggered.connect(
                    lambda checked, n=scan_name, p=param_path: self._delete_intermediate_recons(n, p)
                )

            red_label = QtWidgets.QLabel("Delete Reconstructions Dialogs")
            red_label.setStyleSheet("color: red; font-weight: bold; padding: 2px 40px 2px 20px;")
            red_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            delete_widget_action = QtWidgets.QWidgetAction(menu)
            delete_widget_action.setDefaultWidget(red_label)
            delete_widget_action.setMenu(delete_menu)
            menu.addSeparator()
            menu.addAction(delete_widget_action)

        elif column == 1:  # column 1 stores param folder
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


    def _switch_recon_with_arrows(self, item, key):
        scan_name = item.text(0)

        param_path = item.data(1, Qt.UserRole)
        current_recon = item.data(2, Qt.UserRole)

        if not isinstance(param_path, Path) or not isinstance(current_recon, Path):
            return

        recon_list = sorted(
            self._seen_recon_files[scan_name][param_path.name],
            key=self._recon_sort_key
        )

        if not recon_list:
            return

        try:
            idx = recon_list.index(current_recon)
        except ValueError:
            return

        if key == Qt.Key_Right:
            new_idx = min(idx + 1, len(recon_list) - 1)
        else:  # left
            new_idx = max(idx - 1, 0)

        if new_idx == idx:
            return

        new_recon = recon_list[new_idx]

        # reuse your existing switch logic
        self._switch_recon_file(item, new_recon)


    def _switch_param_folder_with_keys(self, item, key):
        scan_name = item.text(0)
        current_param = item.data(1, Qt.UserRole)
        
        if not isinstance(current_param, Path):
            return
        
        # Get sorted list of param folders for this scan
        param_list = sorted(self._seen_param_folders[scan_name])
        
        if not param_list:
            return
        
        try:
            idx = param_list.index(current_param)
        except ValueError:
            return
        
        if key == Qt.Key_Period:  # . key goes forward
            new_idx = min(idx + 1, len(param_list) - 1)
        else:  # Qt.Key_Comma goes backward
            new_idx = max(idx - 1, 0)
        
        if new_idx == idx:
            return
        
        new_param = param_list[new_idx]
        
        # Reuse existing switch logic
        self._switch_param_folder(item, scan_name, new_param)


    def _switch_image_choice_with_keys(self, key):
        """K = move comboBox up one entry, L = move down one entry."""
        n = self.comboBox_imageChoice.count()
        if n == 0:
            return
        idx = self.comboBox_imageChoice.currentIndex()
        if key == Qt.Key_K:
            new_idx = max(idx - 1, 0)
        else:  # Key_L
            new_idx = min(idx + 1, n - 1)
        if new_idx != idx:
            self.comboBox_imageChoice.setCurrentIndex(new_idx)


    def eventFilter(self, obj, event):
        if obj is self.treeWidget_fileStructure and event.type() == QEvent.KeyPress:

            if event.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Comma, Qt.Key_Period, Qt.Key_K, Qt.Key_L):

                item = self.treeWidget_fileStructure.currentItem()
                if item is None:
                    return False

                # Handle left/right arrow keys for recon files (column 2)
                if event.key() in (Qt.Key_Left, Qt.Key_Right):
                    self._switch_recon_with_arrows(item, event.key())
                    self.on_tree_item_clicked(item, 2)
                    return True  # handled → stop default behavior

                # Handle comma/period keys for param folders (column 1)
                elif event.key() in (Qt.Key_Comma, Qt.Key_Period):
                    self._switch_param_folder_with_keys(item, event.key())
                    self.on_tree_item_clicked(item, 1)
                    return True  # handled → stop default behavior

                # Handle k/l keys for comboBox image choice
                elif event.key() in (Qt.Key_K, Qt.Key_L):
                    self._switch_image_choice_with_keys(event.key())
                    self.on_tree_item_clicked(item, 2)
                    return True  # handled → stop default behavior

        return super().eventFilter(obj, event)


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


    def on_populate_tree_clicked(self):
        """
        Handler for pushButton_populateTree. If a tree cache already exists
        for base_path, ask the user whether to load it (fast) or re-walk
        the filesystem (slow, but authoritative).
        """
        if self.base_path is not None and (self.base_path / TREE_CACHE_FILENAME).exists():
            dlg = QtWidgets.QMessageBox(self)
            dlg.setWindowTitle("Populate Tree")
            dlg.setText("Do you want to load the cached files, or read the file structure directly?")
            load_btn = dlg.addButton("Load cached files", QtWidgets.QMessageBox.AcceptRole)
            read_btn = dlg.addButton("Read file structure", QtWidgets.QMessageBox.DestructiveRole)
            dlg.setDefaultButton(load_btn)
            dlg.exec_()

            clicked = dlg.clickedButton()
            if clicked is load_btn:
                self.load_tree_from_csv()
                return
            elif clicked is not read_btn:
                return  # dialog dismissed without a choice

        self.populate_tree_with_scans()


    def populate_tree_with_scans(self):
        """
        Depth-first population of the tree.
        """
        self._initialize_empty_data_containers()
        self.treeWidget_fileStructure.clear()

        if self.base_path is None:
            return

        self.load_log_csv()

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
        self.save_tree_to_csv()
        self.save_scan_bad_list()
        print(time.time() - t0, 's')

        self._set_scan_watcher_ui('stopped')


    # ------------------------------------------------------------------
    # tree cache (save/load)
    # ------------------------------------------------------------------

    def on_save_tree_clicked(self):
        if not self._seen_scans:
            return
        self.save_tree_to_csv()

    def on_load_tree_clicked(self):
        self.load_tree_from_csv()

    def save_tree_to_csv(self):
        """
        Flatten the current tree state to a long-format CSV (one row per
        scan/param-folder/recon-file triple) in base_path, so it can be
        reloaded later without re-walking the filesystem.
        """
        if self.base_path is None:
            return

        rows = []
        for scan_name in sorted(self._seen_scans):
            row_item = self._scan_row_items.get(scan_name)
            if row_item is None:
                continue

            scan_goodness = row_item.data(0, Qt.UserRole + 1) or 'unknown'
            sample_name = row_item.text(3)
            current_param = row_item.data(1, Qt.UserRole)
            current_recon = row_item.data(2, Qt.UserRole)

            param_folders = sorted(self._seen_param_folders.get(scan_name, set()))

            if not param_folders:
                rows.append({
                    'scan_name': scan_name,
                    'scan_goodness': scan_goodness,
                    'sample_name': sample_name,
                    'param_folder': '',
                    'is_current_param': 0,
                    'recon_file': '',
                    'is_current_recon': 0,
                })
                continue

            for param_path in param_folders:
                is_current_param = int(param_path == current_param)
                recon_files = sorted(
                    self._seen_recon_files.get(scan_name, {}).get(param_path.name, set()),
                    key=self._recon_sort_key,
                )

                if not recon_files:
                    rows.append({
                        'scan_name': scan_name,
                        'scan_goodness': scan_goodness,
                        'sample_name': sample_name,
                        'param_folder': param_path.name,
                        'is_current_param': is_current_param,
                        'recon_file': '',
                        'is_current_recon': 0,
                    })
                    continue

                for recon_file in recon_files:
                    rows.append({
                        'scan_name': scan_name,
                        'scan_goodness': scan_goodness,
                        'sample_name': sample_name,
                        'param_folder': param_path.name,
                        'is_current_param': is_current_param,
                        'recon_file': recon_file.name,
                        'is_current_recon': int(is_current_param and recon_file == current_recon),
                    })

        csv_path = self.base_path / TREE_CACHE_FILENAME
        pd.DataFrame(rows, columns=[
            'scan_name', 'scan_goodness', 'sample_name',
            'param_folder', 'is_current_param',
            'recon_file', 'is_current_recon',
        ]).to_csv(csv_path, index=False)
        print(f"Saved tree cache to {csv_path}")

    def load_tree_from_csv(self):
        """
        Rebuild the tree and its bookkeeping dicts purely from the CSV
        cache — no filesystem access beyond reading the CSV itself.
        """
        if self.base_path is None:
            return

        csv_path = self.base_path / TREE_CACHE_FILENAME
        if not csv_path.exists():
            QtWidgets.QMessageBox.warning(
                self, "No tree cache found",
                f"No tree cache file found at:\n{csv_path}\n\nUse 'Populate tree' or 'Save Tree' first.",
            )
            return

        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Failed to load tree cache", str(exc))
            return

        self.load_log_csv()

        self._initialize_empty_data_containers()
        self.treeWidget_fileStructure.clear()
        self.treeWidget_fileStructure.setUpdatesEnabled(False)

        for scan_name, scan_rows in df.groupby('scan_name', sort=True):
            scan_path = self.base_path / scan_name
            self._seen_scans.add(scan_name)
            self._seen_param_folders.setdefault(scan_name, set())
            self._seen_recon_files.setdefault(scan_name, {})

            first = scan_rows.iloc[0]
            scan_goodness = first['scan_goodness'] or 'unknown'
            # Prefer the log csv over whatever was cached when the tree was saved
            sample_name = self.get_log_csv_sample_name(scan_name) or first['sample_name']

            row_item = QtWidgets.QTreeWidgetItem(self.treeWidget_fileStructure)
            self._scan_row_items[scan_name] = row_item
            self.add_to_tree(row_item, 0, scan_path)
            row_item.setData(0, Qt.UserRole + 1, scan_goodness)
            self.apply_scan_goodness_style(row_item, scan_goodness)

            current_param_path = None
            current_recon_path = None

            for _, r in scan_rows.iterrows():
                if not r['param_folder']:
                    continue
                param_path = scan_path / r['param_folder']
                self._seen_param_folders[scan_name].add(param_path)
                self._seen_recon_files[scan_name].setdefault(r['param_folder'], set())

                if r['is_current_param'] == '1':
                    current_param_path = param_path

                if r['recon_file']:
                    recon_path = param_path / r['recon_file']
                    self._seen_recon_files[scan_name][r['param_folder']].add(recon_path)
                    if r['is_current_recon'] == '1':
                        current_recon_path = recon_path

            if current_param_path is not None:
                self.add_to_tree(row_item, 1, current_param_path)
            else:
                row_item.setText(1, "—")
                row_item.setData(1, Qt.ToolTipRole, "No parameter folder found")

            if current_recon_path is not None:
                self.add_to_tree(row_item, 2, current_recon_path)
            else:
                row_item.setText(2, "—")
                row_item.setData(2, Qt.ToolTipRole, "No recon file found")

            if sample_name:
                row_item.setText(3, sample_name)
                row_item.setData(3, Qt.ToolTipRole, sample_name)
            else:
                row_item.setText(3, "—")
                row_item.setData(3, Qt.ToolTipRole, "Sample name not found")

        self.treeWidget_fileStructure.setUpdatesEnabled(True)
        self.treeWidget_fileStructure.resizeColumnToContents(0)
        self.treeWidget_fileStructure.resizeColumnToContents(2)
        self.treeWidget_fileStructure.sortItems(0, Qt.AscendingOrder)
        self.save_scan_bad_list()
        self._set_scan_watcher_ui('stopped')
        print(f"Loaded tree cache from {csv_path}")


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


    def _confirm_delete(self, header_text: str, paths: list, relative_to: Path = None) -> bool:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Confirm Deletion")
        layout = QtWidgets.QVBoxLayout(dlg)

        layout.addWidget(QtWidgets.QLabel(header_text))

        file_list = QtWidgets.QTextEdit()
        file_list.setReadOnly(True)
        file_list.setMaximumHeight(400)
        file_list.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        file_list.setMinimumWidth(800)
        lines = [
            str(p.relative_to(relative_to)) if relative_to else p.name
            for p in paths
        ]
        file_list.setPlainText("\n".join(lines))
        layout.addWidget(file_list)

        btn_row = QtWidgets.QHBoxLayout()

        delete_btn = QtWidgets.QPushButton("Delete")
        delete_btn.setStyleSheet("background-color: red; color: white; padding: 4px 16px;")
        delete_btn.setAutoDefault(False)
        delete_btn.setDefault(False)
        delete_btn.clicked.connect(dlg.accept)

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setStyleSheet("background-color: green; color: white; padding: 4px 16px;")
        cancel_btn.setDefault(True)
        cancel_btn.setAutoDefault(True)
        cancel_btn.clicked.connect(dlg.reject)

        btn_row.addWidget(delete_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        cancel_btn.setFocus()
        return dlg.exec_() == QtWidgets.QDialog.Accepted

    def _build_intermediate_delete_list(self, param_path: Path) -> list:
        """
        Return a list of Paths for all intermediate reconstructions in param_path.
        The highest-iteration recon_NiterN.h5 is kept; all others and their
        derived files (comboBox items containing XXX but not recon_Niter) are deleted.
        """
        # Step 1: find all recon_NiterN.h5 files and their iteration numbers
        niter_map = {}  # int -> Path
        for f in param_path.glob("recon_Niter*.h5"):
            token = f.stem[len("recon_Niter"):].split("_", 1)[0]
            if token.isdigit():
                niter_map[int(token)] = f

        if not niter_map:
            return []

        max_niter = max(niter_map)
        niters_to_delete = [n for n in niter_map if n != max_niter]
        delete_paths = [niter_map[n] for n in niters_to_delete]

        # Step 2: derived files from comboBox options containing XXX
        for i in range(self.comboBox_imageChoice.count()):
            option = self.comboBox_imageChoice.itemText(i)
            if "recon_Niter" in option:
                continue  # base recon files — already handled above
            if "XXX" not in option:
                continue
            for n in niters_to_delete:
                full_path = param_path / option.replace("XXX", str(n))
                if full_path.exists():
                    delete_paths.append(full_path)

        return delete_paths

    def _delete_intermediate_recons(self, scan_name: str, param_path: Path):
        delete_paths = self._build_intermediate_delete_list(param_path)
        if not delete_paths:
            QtWidgets.QMessageBox.information(self, "Nothing to delete",
                "No intermediate reconstructions found.")
            return
        if not self._confirm_delete(f"Delete files:\n{param_path}", delete_paths):
            return
        for p in delete_paths:
            if p.exists():
                p.unlink()
        # Remove deleted recon files from tracker
        deleted_names = {p for p in delete_paths}
        current = self._seen_recon_files[scan_name].get(param_path.name, set())
        self._seen_recon_files[scan_name][param_path.name] = current - deleted_names

    def _collect_all_intermediate_delete_paths(self, first_num: int, last_num: int) -> list:
        """Collect intermediate recon files for every param folder across a scan range."""
        all_paths = []
        for n in range(min(first_num, last_num), max(first_num, last_num) + 1):
            scan_path = self.base_path / f"S{n:04d}"
            if not scan_path.exists():
                continue
            try:
                subdirs = sorted(d for d in scan_path.iterdir() if d.is_dir())
            except PermissionError:
                continue
            for param_path in subdirs:
                all_paths.extend(self._build_intermediate_delete_list(param_path))
        return all_paths

    def show_delete_intermediate_tool(self):
        """Open a dialog to delete intermediate reconstructions across a scan range."""
        max_scan_num = 0
        for scan_name in self._seen_scans:
            if len(scan_name) == 5 and scan_name.startswith("S") and scan_name[1:].isdigit():
                max_scan_num = max(max_scan_num, int(scan_name[1:]))

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Delete Intermediate Reconstructions")
        layout = QtWidgets.QVBoxLayout(dlg)

        layout.addWidget(QtWidgets.QLabel("First scan in range:"))
        combo_first = QtWidgets.QComboBox()
        combo_first.setEditable(True)
        for idx in range(1, 11):
            combo_first.addItem(f"S{max_scan_num + idx:04d}")
        layout.addWidget(combo_first)

        layout.addWidget(QtWidgets.QLabel("Last scan in range:"))
        combo_last = QtWidgets.QComboBox()
        combo_last.setEditable(True)
        for idx in range(1, 11):
            combo_last.addItem(f"S{max_scan_num + idx:04d}")
        layout.addWidget(combo_last)

        btn_row = QtWidgets.QHBoxLayout()
        confirm_btn = QtWidgets.QPushButton("Confirm files")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(confirm_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        def parse_scan(text):
            t = text.strip()
            if len(t) == 5 and t.startswith("S") and t[1:].isdigit():
                return int(t[1:])
            return None

        def on_confirm():
            first_text = combo_first.currentText().strip()
            last_text = combo_last.currentText().strip()
            first_num = parse_scan(first_text)
            last_num = parse_scan(last_text)
            if first_num is None:
                QtWidgets.QMessageBox.warning(dlg, "Invalid Format",
                    f"Invalid scan name: {first_text}\nExpected: S#### (e.g., S0042)")
                return
            if last_num is None:
                QtWidgets.QMessageBox.warning(dlg, "Invalid Format",
                    f"Invalid scan name: {last_text}\nExpected: S#### (e.g., S0042)")
                return

            all_paths = self._collect_all_intermediate_delete_paths(first_num, last_num)
            if not all_paths:
                QtWidgets.QMessageBox.information(dlg, "Nothing to delete",
                    "No intermediate reconstructions found in the given range.")
                return

            header = f"Delete files range:\n{first_text}\n{last_text}"
            if not self._confirm_delete(header, all_paths, relative_to=self.base_path):
                return

            for p in all_paths:
                if p.exists():
                    p.unlink()

            # Update trackers
            deleted_set = set(all_paths)
            lo, hi = min(first_num, last_num), max(first_num, last_num)
            for n in range(lo, hi + 1):
                sname = f"S{n:04d}"
                if sname in self._seen_recon_files:
                    for pname in self._seen_recon_files[sname]:
                        self._seen_recon_files[sname][pname] -= deleted_set

            dlg.accept()

        confirm_btn.clicked.connect(on_confirm)
        dlg.resize(380, 180)
        dlg.exec()

    def _delete_scan_folder(self, scan_name: str, scan_path: Path, item: QtWidgets.QTreeWidgetItem):
        if not self._confirm_delete(f"Delete files:\n{scan_path.parent}", [scan_path]):
            return
        shutil.rmtree(scan_path)
        self._remove_scan_row(scan_name, item)

    def _delete_param_folder(self, scan_name: str, param_path: Path, item: QtWidgets.QTreeWidgetItem):
        if not self._confirm_delete(f"Delete files:\n{param_path.parent}", [param_path]):
            return
        shutil.rmtree(param_path)
        # Clean up trackers
        self._seen_param_folders[scan_name].discard(param_path)
        self._seen_recon_files[scan_name].pop(param_path.name, None)
        # Update the tree row
        remaining = sorted(self._seen_param_folders[scan_name])
        if remaining:
            self._switch_param_folder(item, scan_name, remaining[-1])
        else:
            item.setText(1, "—")
            item.setData(1, Qt.UserRole, None)
            item.setData(1, Qt.ToolTipRole, "No parameter folder found")
            item.setText(2, "—")
            item.setData(2, Qt.UserRole, None)
            item.setData(2, Qt.ToolTipRole, "No recon file found")

    def _remove_scan_row(self, scan_name: str, row_item: QtWidgets.QTreeWidgetItem):
        # Col 0: remove from tree widget
        index = self.treeWidget_fileStructure.indexOfTopLevelItem(row_item)
        if index != -1:
            self.treeWidget_fileStructure.takeTopLevelItem(index)
        # Col 0: scan_row_items + seen_scans
        self._scan_row_items.pop(scan_name, None)
        self._seen_scans.discard(scan_name)
        # Col 1: param folders
        self._seen_param_folders.pop(scan_name, None)
        # Col 2: recon files
        self._seen_recon_files.pop(scan_name, None)

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

        # The log csv lives next to the base path, so the old one no longer
        # applies. It is only re-read on populate/load tree, not here.
        self._clear_log_csv_state()
        self._set_log_csv_ui()

        self._set_scan_watcher_ui('gray')
        self.save_base_path()

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
