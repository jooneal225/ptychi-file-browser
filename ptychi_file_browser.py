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
from scipy.ndimage import map_coordinates, median_filter, gaussian_filter

from scan_watcher_thread import ScanWatcherThread

import pyqtgraph as pg
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtCore import Qt, QSettings, QEvent
from PyQt5.QtGui import QImage, QPixmap, QColor, QBrush, QFont



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

        self.treeWidget_fileStructure.installEventFilter(self)

        self._initialize_empty_data_containers()
        self._set_scan_watcher_ui('gray')
        self._setup_tree()
        self._connect_signals()
        self._setup_pyqtgraph_view()
        self.restore_window_size()
        self.load_base_path()
        self.on_base_path_entered()

        self.treeWidget_fileStructure.setContextMenuPolicy(Qt.CustomContextMenu)
        self.treeWidget_fileStructure.customContextMenuRequested.connect(self.on_tree_right_click)


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
        self.pushButton_populateTree.clicked.connect(self.populate_tree_with_scans)
        self.treeWidget_fileStructure.itemClicked.connect(self.on_tree_item_clicked)
        self.treeWidget_fileStructure.currentItemChanged.connect(self.on_tree_selection_changed)
        self.pushButton_stopScanUpdate.clicked.connect(self.on_stop_scan_update)
        self.pushButton_updateScanGoodness.clicked.connect(self.on_update_scan_goodness)
        self.toolButton_tips.clicked.connect(self.show_secret_features)
        self.pushButton_addScan.clicked.connect(self.on_add_scan_clicked)


    def _setup_pyqtgraph_view(self):
        """
        Embed pyqtgraph ImageView into the placeholder widget.
        """
        self.pg_view = pg.ImageView()

        # Lineout plot (hidden by default, lives in a splitter below the image)
        self._lineout_visible = False
        self._lineout_plot = pg.PlotWidget()
        self._lineout_plot.setLabel('bottom', 'Distance (µm)')
        self._lineout_plot.setLabel('left', 'Value')
        self._lineout_plot.setVisible(False)

        # Resolution metric: two draggable vertical lines + corner text
        self._lineout_dist_um = None
        self._lineout_values = None
        self._res_lines_positioned = False
        self._res_line1 = pg.InfiniteLine(
            pos=0.0, angle=90, movable=True,
            pen=pg.mkPen(color=(80, 120, 255), width=1.5, style=Qt.DotLine)
        )
        self._res_line2 = pg.InfiniteLine(
            pos=1.0, angle=90, movable=True,
            pen=pg.mkPen(color=(80, 220, 80), width=1.5, style=Qt.DotLine)
        )
        self._res_nm = None
        self._res_region = None
        self._res_line1.sigPositionChanged.connect(self._on_res_line_moved)
        self._res_line2.sigPositionChanged.connect(self._on_res_line_moved)
        self._res_text_item = pg.TextItem(text='', color='w', anchor=(1, 0),
                                          fill=pg.mkBrush(0, 0, 0, 180))
        _res_font = QFont("Courier", 9)
        self._res_text_item.setFont(_res_font)
        self._lineout_plot.getViewBox().sigRangeChanged.connect(self._reposition_res_text)
        # Context menu action for the lineout plot
        self._find_resolution_action = QtWidgets.QAction("Find 10%-90% resolution")
        self._find_resolution_action.triggered.connect(self._find_resolution)
        self._lineout_plot.getViewBox().menu.addSeparator()
        self._lineout_plot.getViewBox().menu.addAction(self._find_resolution_action)

        splitter = QtWidgets.QSplitter(Qt.Vertical)
        splitter.addWidget(self.pg_view)
        splitter.addWidget(self._lineout_plot)
        splitter.setSizes([700, 200])

        self.graphicsView_1_layout = QtWidgets.QVBoxLayout(self.graphicsView_1)
        self.graphicsView_1_layout.setContentsMargins(0, 0, 0, 0)
        self.graphicsView_1_layout.addWidget(splitter)

        # --- Measurement overlay ---
        self._measure_clicks = []  # up to 2 (x, y) pixel-space coords

        self._measure_scatter = pg.ScatterPlotItem(
            size=12, pen=pg.mkPen('r', width=2), brush=pg.mkBrush(None), symbol='+'
        )
        self.pg_view.getView().addItem(self._measure_scatter)

        self._measure_line = pg.PlotCurveItem(
            pen=pg.mkPen('r', width=1, style=Qt.DashLine)
        )
        self.pg_view.getView().addItem(self._measure_line)

        self._distance_text = pg.TextItem(
            color=(0, 0, 0), anchor=(0.5, 0.5),
            fill=pg.mkBrush(255, 255, 255, 220),
        )
        _font = QFont()
        _font.setPointSize(10)
        self._distance_text.textItem.setFont(_font)
        self.pg_view.getView().addItem(self._distance_text)
        self._distance_text.setVisible(False)

        # Positions overlay (for _pos choice)
        self._positions_scatter_overlay = pg.ScatterPlotItem(
            size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 180)
        )
        self._positions_scatter_overlay.setVisible(False)
        self.pg_view.getView().addItem(self._positions_scatter_overlay)

        # "Copy param folder path" action
        self._copy_param_path_action = QtWidgets.QAction("Copy Param Folder Path")
        self._copy_param_path_action.triggered.connect(self._copy_current_param_path)
        self.pg_view.getView().menu.addAction(self._copy_param_path_action)
        self.pg_view.getView().menu.addSeparator()

        # "Plot Lineout" toggle in the ViewBox right-click menu
        self._lineout_action = QtWidgets.QAction("Plot Lineout")
        self._lineout_action.setCheckable(True)
        self._lineout_action.triggered.connect(self._toggle_lineout)
        self.pg_view.getView().menu.addSeparator()
        self.pg_view.getView().menu.addAction(self._lineout_action)

        # "Reset Zoom" action
        self._reset_zoom_action = QtWidgets.QAction("Reset Zoom")
        self._reset_zoom_action.triggered.connect(lambda: self.pg_view.getView().autoRange())
        self.pg_view.getView().menu.addAction(self._reset_zoom_action)

        # "Auto-reset Zoom" toggle — when checked, zoom resets on every new image load
        self._auto_reset_zoom_action = QtWidgets.QAction("Auto-reset Zoom")
        self._auto_reset_zoom_action.setCheckable(True)
        self._auto_reset_zoom_action.setChecked(True)
        self.pg_view.getView().menu.addAction(self._auto_reset_zoom_action)

        # "Full Probe Zoom" toggle — when checked, skip the square crop zoom
        self._full_probe_zoom_action = QtWidgets.QAction("Full Probe Zoom")
        self._full_probe_zoom_action.setCheckable(True)
        self.pg_view.getView().menu.addAction(self._full_probe_zoom_action)

        # "Analyze" submenu
        self._active_filter = None   # 'median' | 'gaussian' | None
        self._filter_kernel = 3.0
        self._analyze_menu = QtWidgets.QMenu("Analyze")
        analyze_menu = self._analyze_menu
        self.pg_view.getView().menu.addSeparator()
        self.pg_view.getView().menu.addMenu(analyze_menu)

        self._median_filter_action = QtWidgets.QAction("Median Filter")
        self._median_filter_action.setCheckable(True)
        self._median_filter_action.triggered.connect(
            lambda checked: self._set_filter('median', checked)
        )
        analyze_menu.addAction(self._median_filter_action)

        self._gaussian_filter_action = QtWidgets.QAction("Gaussian Filter")
        self._gaussian_filter_action.setCheckable(True)
        self._gaussian_filter_action.triggered.connect(
            lambda checked: self._set_filter('gaussian', checked)
        )
        analyze_menu.addAction(self._gaussian_filter_action)

        # --- Mouse signals ---
        self._mouse_move_proxy = pg.SignalProxy(
            self.pg_view.scene.sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )
        self.pg_view.scene.sigMouseClicked.connect(self._on_mouse_clicked)


    def _on_mouse_moved(self, event):
        pos = event[0]  # SignalProxy wraps args in a tuple
        img_item = self.pg_view.getImageItem()
        if img_item is None or img_item.image is None:
            return
        base = getattr(self, '_info_base_text', '')
        if img_item.sceneBoundingRect().contains(pos):
            pt = img_item.mapFromScene(pos)
            nx, ny = img_item.image.shape[:2]
            x_um = (pt.x() - nx / 2) * self.res_m * 1e6
            y_um = (pt.y() - ny / 2) * self.res_m * 1e6
            xi, yi = int(pt.x()), int(pt.y())
            if 0 <= xi < nx and 0 <= yi < ny:
                intens = img_item.image[xi, yi]
                self.label_plot_info.setText(f"{base}\nx={x_um:.2f}, y={y_um:.2f} µm, I={intens:.2g}")
            else:
                self.label_plot_info.setText(f"{base}\nx={x_um:.2f}, y={y_um:.2f} µm")
        else:
            self.label_plot_info.setText(base)


    def _on_mouse_clicked(self, event):
        if event.button() != Qt.LeftButton:
            return
        pos = event.scenePos()
        img_item = self.pg_view.getImageItem()
        if img_item is None or img_item.image is None:
            return
        if not img_item.sceneBoundingRect().contains(pos):
            return

        pt = img_item.mapFromScene(pos)
        x, y = pt.x(), pt.y()

        if len(self._measure_clicks) == 2:
            # Third click: clear everything
            self._measure_clicks = []
            self._measure_scatter.setData(x=[], y=[])
            self._measure_line.setData([], [])
            self._distance_text.setVisible(False)
            self._update_lineout()
            return

        self._measure_clicks.append((x, y))
        self._measure_scatter.setData(
            x=[p[0] for p in self._measure_clicks],
            y=[p[1] for p in self._measure_clicks],
        )

        if len(self._measure_clicks) == 2:
            x1, y1 = self._measure_clicks[0]
            x2, y2 = self._measure_clicks[1]
            dist_um = np.sqrt(
                ((x2 - x1) * self.res_m * 1e6) ** 2 +
                ((y2 - y1) * self.res_m * 1e6) ** 2
            )
            self._measure_line.setData([x1, x2], [y1, y2])
            # Offset the label perpendicularly away from the line
            dx, dy = x2 - x1, y2 - y1
            seg_len = np.sqrt(dx**2 + dy**2)
            offset = max(seg_len * 0.07, 30)
            if seg_len > 0:
                px, py = -dy / seg_len * offset, dx / seg_len * offset
            else:
                px, py = 0, offset
            self._distance_text.setPos((x1 + x2) / 2 + px, (y1 + y2) / 2 + py)
            dist_px = int(round(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)))
            self._distance_text.setText(f"{dist_um:.3f} µm\n{dist_px} pix")
            self._distance_text.setVisible(True)
            self._update_lineout()


    def _set_filter(self, filter_type: str, checked: bool):
        if not checked:
            self._active_filter = None
        else:
            kernel, ok = QtWidgets.QInputDialog.getDouble(
                self, f"{filter_type.capitalize()} Filter", "Kernel width:",
                value=self._filter_kernel, min=0.1, max=500.0, decimals=1
            )
            if not ok:
                # User cancelled — revert the checkmark
                action = self._median_filter_action if filter_type == 'median' else self._gaussian_filter_action
                action.setChecked(False)
                return
            self._filter_kernel = kernel
            self._active_filter = filter_type
            # Uncheck the other filter
            other = self._gaussian_filter_action if filter_type == 'median' else self._median_filter_action
            other.setChecked(False)

        # Re-display current image with (or without) the filter
        item = self.treeWidget_fileStructure.currentItem()
        if item is not None:
            self.on_tree_item_clicked(item, 2)

    def _copy_current_param_path(self):
        item = self.treeWidget_fileStructure.currentItem()
        if item is None:
            return
        param_path = Path(item.data(1, Qt.UserRole)).name
        QApplication.clipboard().setText(str(param_path))

    def _toggle_lineout(self, checked: bool):
        self._lineout_visible = checked
        self._lineout_plot.setVisible(checked)
        self._update_lineout()

    def _update_lineout(self):
        if not self._lineout_visible:
            return
        self._lineout_plot.clear()
        self._res_nm = None     # cleared by plot.clear()
        self._res_region = None
        if len(self._measure_clicks) < 2:
            self._lineout_dist_um = None
            self._lineout_values = None
            return
        x1, y1 = self._measure_clicks[0]
        x2, y2 = self._measure_clicks[1]
        dist_um, values = self._compute_lineout(x1, y1, x2, y2)
        if dist_um is not None:
            self._lineout_dist_um = dist_um
            self._lineout_values = values
            self._lineout_plot.plot(dist_um, values, pen=pg.mkPen('w', width=1))
            marker_pen = pg.mkPen('r', width=1, style=Qt.DashLine)
            self._lineout_plot.addItem(pg.InfiniteLine(pos=dist_um[0],  angle=90, pen=marker_pen))
            self._lineout_plot.addItem(pg.InfiniteLine(pos=dist_um[-1], angle=90, pen=marker_pen))
            # Position metric lines on first use, then keep user-dragged positions
            if not self._res_lines_positioned:
                span = dist_um[-1] - dist_um[0]
                self._res_line1.setPos(dist_um[0] + span / 3)
                self._res_line2.setPos(dist_um[0] + 2 * span / 3)
                self._res_lines_positioned = True
            # Re-add after clear() — ignoreBounds keeps them out of autoscale
            self._lineout_plot.addItem(self._res_line1, ignoreBounds=True)
            self._lineout_plot.addItem(self._res_line2, ignoreBounds=True)
            self._lineout_plot.addItem(self._res_text_item, ignoreBounds=True)
            self._update_res_metric()

    def _on_res_line_moved(self):
        """Called when a metric line is dragged — clears any computed resolution."""
        self._clear_resolution()
        self._update_res_metric()

    def _clear_resolution(self):
        if self._res_region is not None:
            try:
                self._lineout_plot.removeItem(self._res_region)
            except Exception:
                pass
            self._res_region = None
        self._res_nm = None

    def _update_res_metric(self):
        if self._lineout_dist_um is None or self._lineout_values is None:
            return
        xb = self._res_line1.value()
        xg = self._res_line2.value()
        yb = float(np.interp(xb, self._lineout_dist_um, self._lineout_values))
        yg = float(np.interp(xg, self._lineout_dist_um, self._lineout_values))
        delta_x = abs(xg - xb)
        lines = [
            f"Blue  X: {xb:.3f} µm",
            f"Blue  Y: {yb:.4g}",
            f"Green X: {xg:.3f} µm",
            f"Green Y: {yg:.4g}",
            f"ΔX:      {delta_x:.3f} µm",
        ]
        if self._res_nm is not None:
            lines.append(f"Resolution: {self._res_nm} nm")
        self._res_text_item.setText("\n".join(lines))
        self._reposition_res_text()

    def _find_resolution(self):
        if self._lineout_dist_um is None or self._lineout_values is None:
            return
        xb = self._res_line1.value()
        xg = self._res_line2.value()
        yb = float(np.interp(xb, self._lineout_dist_um, self._lineout_values))
        yg = float(np.interp(xg, self._lineout_dist_um, self._lineout_values))

        lo, hi = min(xb, xg), max(xb, xg)
        mask = (self._lineout_dist_um >= lo) & (self._lineout_dist_um <= hi)
        x_slice = self._lineout_dist_um[mask]
        y_slice = self._lineout_values[mask]
        if len(x_slice) < 2:
            return

        level_10 = yb + 0.1 * (yg - yb)
        level_90 = yb + 0.9 * (yg - yb)

        # np.interp requires monotonically increasing xp — sort by y value
        if y_slice[-1] >= y_slice[0]:
            x_10 = float(np.interp(level_10, y_slice, x_slice))
            x_90 = float(np.interp(level_90, y_slice, x_slice))
        else:
            x_10 = float(np.interp(level_10, y_slice[::-1], x_slice[::-1]))
            x_90 = float(np.interp(level_90, y_slice[::-1], x_slice[::-1]))

        self._res_nm = int(round(abs(x_90 - x_10) * 1e3))

        # Shade the region between the two crossings
        self._clear_resolution()   # remove any old region first
        self._res_nm = int(round(abs(x_90 - x_10) * 1e3))
        self._res_region = pg.LinearRegionItem(
            values=[min(x_10, x_90), max(x_10, x_90)],
            brush=pg.mkBrush(160, 0, 200, 70),
            movable=False,
        )
        self._lineout_plot.addItem(self._res_region, ignoreBounds=True)
        self._update_res_metric()

    def _reposition_res_text(self):
        vr = self._lineout_plot.getViewBox().viewRange()
        x_min, x_max = vr[0]
        y_min, y_max = vr[1]
        mx = (x_max - x_min) * 0.01
        my = (y_max - y_min) * 0.02
        self._res_text_item.setPos(x_max - mx, y_max - my)

    def _compute_lineout(self, x1, y1, x2, y2):
        img = self.pg_view.getImageItem().image
        if img is None:
            return None, None
        n_pts = max(int(np.hypot(x2 - x1, y2 - y1)), 10)
        xs = np.linspace(x1, x2, n_pts)
        ys = np.linspace(y1, y2, n_pts)
        values = map_coordinates(img, [xs, ys], order=1, mode='nearest')
        total_um = np.hypot((x2 - x1) * self.res_m * 1e6, (y2 - y1) * self.res_m * 1e6)
        dist_um = np.linspace(0, total_um, n_pts)
        return dist_um, values


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
        ----- Lineout viewer can be used to find the 10%-90% resolution
        ----- Drag the blue and green lines to the boundaries of a hard edge, and the right-click menu will calculate it
        --- By default, probe viewer is centered on mode 0, and the right-click menu can turn this off
        --- Right-click menu can copy parameter folder string (Ndp256...)
        --- Right-click menu can reset zoom and change default zoom behavior

        Scan goodness
        --- Row color shows scan goodness, tracked as txt file in scan folder
        --- Green is a good ptycho recon, yellow marks a scan to reanalyze, red is bad data
        
        Sample names
        --- Pulled from file 'runtable_full_{self.base_path.parent.name}.csv'
        --- File must be located in parent of base path
        --- i.e. {self.base_path.parent}
        --- Searches for scan number in column 'run', and returns corresponding string from 'sample_name'

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
        del_tool_btn = QtWidgets.QPushButton("Delete intermediate reconstructions tool")
        del_tool_btn.setStyleSheet("background-color: red; color: white;")
        del_tool_btn.clicked.connect(lambda: (dlg.accept(), self.show_delete_intermediate_tool()))
        btn_row.addWidget(close_btn)
        btn_row.addWidget(del_tool_btn)
        layout.addLayout(btn_row)

        dlg.resize(700, 400)
        dlg.exec()


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
        Tries runtable first, then falls back to parsing a filename directly.
        """
        try:
            scan_num = int(scan_path.name[1:])
        except ValueError:
            return None

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
        data = data.astype(np.float32).T if self.checkBox_transpose.isChecked() else data.astype(np.float32)
        self._displayed_shape = data.shape  # (nx, ny) in pyqtgraph convention

        # Clear measurement overlay (pixel coords are image-specific)
        self._measure_clicks = []
        self._measure_scatter.setData(x=[], y=[])
        self._measure_line.setData([], [])
        self._distance_text.setVisible(False)
        self._update_lineout()

        # Update image info label
        pix_size_nm = round(self.res_m * 1e9)
        self._info_base_text = "%.2f\u00d7%.2f µm, %d nm pix" % (data.shape[0] * self.res_m * 1e6, data.shape[1] * self.res_m * 1e6, pix_size_nm)
        self.label_plot_info.setText(self._info_base_text)

        # Apply filter
        if self._active_filter == 'median':
            data = median_filter(data, size=max(1, int(self._filter_kernel)))
        elif self._active_filter == 'gaussian':
            data = gaussian_filter(data, sigma=self._filter_kernel)

        # Display
        auto_range = self._auto_reset_zoom_action.isChecked()
        if self.checkBox_logCmap.isChecked():
            self.pg_view.setImage(np.log10(np.clip(np.abs(data), a_min=np.finfo(float).eps, a_max=None)), autoLevels=True, autoRange=auto_range)
        else:
            self.pg_view.setImage(data, autoLevels=True, autoRange=auto_range)

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
        shape = getattr(self, '_displayed_shape', None)
        if shape is None:
            return
        ny = shape[1]  # vertical height H
        # Leftmost square: x in [0, H], y in [0, H]
        self.pg_view.getView().setRange(xRange=(0, ny), yRange=(0, ny), padding=0.05)


    def _update_positions_overlay(self):
        """Show scan positions as a red scatter overlay (only for _pos choice)."""
        pos = getattr(self, '_positions_px', None)
        if pos is None or len(pos) == 0:
            self._positions_scatter_overlay.setVisible(False)
            return
        # positions_px is (N, 2) in (row, col) = (y, x) convention stored as
        # offsets from center; shift to image center in pyqtgraph (axis 0 = x, axis 1 = y)
        nx, ny = getattr(self, '_displayed_shape', (0, 0))
        self._positions_scatter_overlay.setData(x=pos[:, 1] + nx / 2, y=pos[:, 0] + ny / 2)
        self._positions_scatter_overlay.setVisible(True)


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


    def populate_tree_with_scans(self):
        """
        Depth-first population of the tree.
        """
        self._initialize_empty_data_containers()
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

        self._set_scan_watcher_ui('stopped')


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
