"""
ImagePlotWidget -- a pyqtgraph ImageView with measurement, readout, lineout and
a populated right-click menu.

See ``pg_image_tools/__init__.py`` for usage documentation.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from scipy.ndimage import median_filter, gaussian_filter

from ._lineout import LineoutPanel


class ImagePlotWidget(QtWidgets.QWidget):
    """
    Interactive 2D image viewer built on ``pyqtgraph.ImageView``.

    pyqtgraph is the only hard requirement. Every companion widget (info label,
    title label, transpose checkbox, log checkbox) is optional: supply it and it
    is wired up, omit it and that feature is silently skipped.
    """

    #: Emitted after a display option (filter / transpose / log) has been applied
    #: and the cached array re-rendered. A notification, not a request for data.
    sigRedrawRequested = QtCore.pyqtSignal()

    #: Emitted whenever the measurement points change; carries a list of 0, 1 or
    #: 2 (x, y) tuples in image pixel coordinates.
    sigPointsChanged = QtCore.pyqtSignal(list)

    #: Emitted while the cursor is over the image; (x, y) in image pixel coords.
    sigMouseMovedImage = QtCore.pyqtSignal(float, float)

    def __init__(self, container=None, *,
                 info_label=None,
                 title_label=None,
                 transpose_checkbox=None,
                 log_checkbox=None,
                 enable_lineout=True,
                 enable_measure=True,
                 enable_filters=True,
                 title_elide_width=500,
                 parent=None):
        super().__init__(parent)

        self._enable_measure = enable_measure
        self._enable_filters = enable_filters
        self._title_elide_width = title_elide_width

        self._info_label = None
        self._title_label = None
        self._transpose_checkbox = None
        self._log_checkbox = None

        # ---- display state ----
        self._raw_data = None
        self._pixel_size_m = 1.0
        self._displayed_shape = None
        self._info_base_text = ''
        self._colorbar_added = False
        self._measure_clicks = []          # up to 2 (x, y) pixel-space coords
        self._active_filter = None         # 'median' | 'gaussian' | None
        self._filter_kernel = 3.0

        # ---- widgets ----
        self.pg_view = pg.ImageView()
        self.lineout = LineoutPanel() if enable_lineout else None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if self.lineout is not None:
            splitter = QtWidgets.QSplitter(Qt.Vertical)
            splitter.addWidget(self.pg_view)
            splitter.addWidget(self.lineout)
            splitter.setSizes([700, 200])
            layout.addWidget(splitter)
            self.splitter = splitter
        else:
            layout.addWidget(self.pg_view)
            self.splitter = None

        self._build_overlays()
        self._build_menu()

        # ---- mouse signals ----
        self._mouse_move_proxy = pg.SignalProxy(
            self.pg_view.scene.sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )
        self.pg_view.scene.sigMouseClicked.connect(self._on_mouse_clicked)

        # ---- optional companions ----
        self.attach_info_label(info_label)
        self.attach_title_label(title_label)
        self.attach_transpose_checkbox(transpose_checkbox)
        self.attach_log_checkbox(log_checkbox)

        if container is not None:
            self.install_into(container)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    def install_into(self, container: QtWidgets.QWidget):
        """Fill ``container`` with this widget, adding a zero-margin layout if needed."""
        layout = container.layout()
        if layout is None:
            layout = QtWidgets.QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self)

    def _build_overlays(self):
        view = self.pg_view.getView()

        # --- Measurement overlay ---
        self._measure_scatter = pg.ScatterPlotItem(
            size=12, pen=pg.mkPen('r', width=2), brush=pg.mkBrush(None), symbol='+'
        )
        view.addItem(self._measure_scatter)

        self._measure_line = pg.PlotCurveItem(
            pen=pg.mkPen('r', width=1, style=Qt.DashLine)
        )
        view.addItem(self._measure_line)

        self._distance_text = pg.TextItem(
            color=(0, 0, 0), anchor=(0.5, 0.5),
            fill=pg.mkBrush(255, 255, 255, 220),
        )
        _font = QFont()
        _font.setPointSize(10)
        self._distance_text.textItem.setFont(_font)
        view.addItem(self._distance_text)
        self._distance_text.setVisible(False)

        # --- Generic scatter overlay (host-driven, e.g. scan positions) ---
        self._scatter_overlay = pg.ScatterPlotItem(
            size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 180)
        )
        self._scatter_overlay.setVisible(False)
        view.addItem(self._scatter_overlay)

    def _build_menu(self):
        """
        Populate the ViewBox right-click menu with this widget's own block.

        ``_menu_anchor`` is the first action we add; host actions requested with
        ``at_top=True`` are inserted before it, i.e. above our block but below
        pyqtgraph's built-in entries.
        """
        self._menu = self.pg_view.getView().menu
        self._menu_anchor = None

        self._own_add_separator()

        self.action_lineout = None
        if self.lineout is not None:
            self.action_lineout = QtWidgets.QAction("Plot Lineout")
            self.action_lineout.setCheckable(True)
            self.action_lineout.triggered.connect(self.set_lineout_visible)
            self._own_add_action(self.action_lineout)

        # No "Reset Zoom" entry — pyqtgraph's built-in "View All" already does it.
        self.action_auto_reset_zoom = QtWidgets.QAction("Auto-reset Zoom")
        self.action_auto_reset_zoom.setCheckable(True)
        self.action_auto_reset_zoom.setChecked(True)
        self._own_add_action(self.action_auto_reset_zoom)

        self.action_median_filter = None
        self.action_gaussian_filter = None
        if self._enable_filters:
            self._own_add_separator()
            self._analyze_menu = QtWidgets.QMenu("Analyze")
            self._own_add_menu(self._analyze_menu)

            self.action_median_filter = QtWidgets.QAction("Median Filter")
            self.action_median_filter.setCheckable(True)
            self.action_median_filter.triggered.connect(
                lambda checked: self.set_filter('median', checked)
            )
            self._analyze_menu.addAction(self.action_median_filter)

            self.action_gaussian_filter = QtWidgets.QAction("Gaussian Filter")
            self.action_gaussian_filter.setCheckable(True)
            self.action_gaussian_filter.triggered.connect(
                lambda checked: self.set_filter('gaussian', checked)
            )
            self._analyze_menu.addAction(self.action_gaussian_filter)

    def _note_anchor(self, action):
        if self._menu_anchor is None:
            self._menu_anchor = action
        return action

    def _own_add_action(self, action):
        self._menu.addAction(action)
        return self._note_anchor(action)

    def _own_add_separator(self):
        return self._note_anchor(self._menu.addSeparator())

    def _own_add_menu(self, menu):
        return self._note_anchor(self._menu.addMenu(menu))

    # ------------------------------------------------------------------
    # menu extension API
    # ------------------------------------------------------------------

    def add_menu_action(self, text, callback=None, checkable=False, checked=False,
                        at_top=False, after=None):
        """
        Add a host action to the image's right-click menu and return the QAction.

        at_top : insert above this widget's own block (below pyqtgraph's builtins).
                 Successive at_top calls keep their relative order.
        after  : insert directly after the given QAction (e.g. ``action_auto_reset_zoom``).
                 Takes precedence over ``at_top``.
        """
        action = QtWidgets.QAction(text, self)
        if checkable:
            action.setCheckable(True)
            action.setChecked(checked)
        if callback is not None:
            action.triggered.connect(callback)
        self._insert_menu_entry(action, at_top=at_top, after=after)
        return action

    def add_menu_separator(self, at_top=False, after=None):
        """Add a separator to the image's right-click menu; returns the QAction."""
        separator = QtWidgets.QAction(self)
        separator.setSeparator(True)
        self._insert_menu_entry(separator, at_top=at_top, after=after)
        return separator

    def add_menu_submenu(self, menu, at_top=False, after=None):
        """Add a QMenu to the image's right-click menu; returns its QAction."""
        self._insert_menu_entry(menu.menuAction(), at_top=at_top, after=after)
        return menu.menuAction()

    def _insert_menu_entry(self, action, at_top=False, after=None):
        actions = self._menu.actions()
        if after is not None and after in actions:
            idx = actions.index(after) + 1
            if idx < len(actions):
                self._menu.insertAction(actions[idx], action)
            else:
                self._menu.addAction(action)
        elif at_top and self._menu_anchor is not None:
            self._menu.insertAction(self._menu_anchor, action)
        else:
            self._menu.addAction(action)

    # ------------------------------------------------------------------
    # optional companion widgets
    # ------------------------------------------------------------------

    def attach_info_label(self, label):
        """
        QLabel that shows image dimensions plus the live cursor readout.

        Whatever the label already says is left alone until the first image.
        """
        self._info_label = label
        if label is not None and self._raw_data is not None:
            label.setText(self._info_base_text)

    def attach_title_label(self, label):
        """QLabel used by :meth:`set_title` (text is elided, full text as tooltip)."""
        self._title_label = label

    def attach_transpose_checkbox(self, checkbox):
        """QCheckBox that transposes the image on display. Live: toggling redraws."""
        self._transpose_checkbox = self._rebind(
            self._transpose_checkbox, checkbox, self._on_transpose_toggled
        )

    def attach_log_checkbox(self, checkbox):
        """QCheckBox that displays log10(|data|). Live: toggling redraws."""
        self._log_checkbox = self._rebind(
            self._log_checkbox, checkbox, self._on_log_toggled
        )

    @staticmethod
    def _rebind(old, new, slot):
        if old is not None:
            try:
                old.stateChanged.disconnect(slot)
            except TypeError:
                pass
        if new is not None:
            new.stateChanged.connect(slot)
        return new

    def _on_transpose_toggled(self, *_):
        # Transposing changes the coordinate system, so pixel-space overlays go stale.
        self.redraw(clear_overlays=True)
        self.sigRedrawRequested.emit()

    def _on_log_toggled(self, *_):
        self.redraw()
        self.sigRedrawRequested.emit()

    # ------------------------------------------------------------------
    # display
    # ------------------------------------------------------------------

    def set_image(self, data, pixel_size_m=1.0, autoRange=None):
        """
        Display a 2D array and cache it for later redraws.

        pixel_size_m : metres per pixel; drives every µm / nm readout.
        autoRange    : None means "use the Auto-reset Zoom menu action".
        """
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("set_image expects a 2D numpy array")

        self._raw_data = data
        self._pixel_size_m = float(pixel_size_m)

        # Measurement coords are image-specific
        self.clear_measure_points(update_lineout=False)
        self.clear_scatter_overlay()

        if autoRange is None:
            autoRange = self.action_auto_reset_zoom.isChecked()
        self._render(autoRange=autoRange)
        self._update_lineout()

    def redraw(self, clear_overlays=False):
        """
        Re-render the cached array after a display-option change.

        Preserves the current zoom, measurement points, lineout and scatter
        overlay unless ``clear_overlays`` is set. No-op before the first image.
        """
        if self._raw_data is None:
            return
        if clear_overlays:
            self.clear_measure_points(update_lineout=False)
            self.clear_scatter_overlay()
        self._render(autoRange=False)
        self._update_lineout()

    def _render(self, autoRange):
        data = self._raw_data.astype(np.float32)
        if self._transpose_checked():
            data = data.T
        self._displayed_shape = data.shape   # (nx, ny) in pyqtgraph convention

        pix_size_nm = round(self._pixel_size_m * 1e9)
        self._info_base_text = "%.2f×%.2f µm, %d nm pix" % (
            data.shape[0] * self._pixel_size_m * 1e6,
            data.shape[1] * self._pixel_size_m * 1e6,
            pix_size_nm,
        )
        if self._info_label is not None:
            self._info_label.setText(self._info_base_text)

        if self._active_filter == 'median':
            data = median_filter(data, size=max(1, int(self._filter_kernel)))
        elif self._active_filter == 'gaussian':
            data = gaussian_filter(data, sigma=self._filter_kernel)

        if self._log_checked():
            data = np.log10(np.clip(np.abs(data), a_min=np.finfo(float).eps, a_max=None))

        self.pg_view.setImage(data, autoLevels=True, autoRange=autoRange)

        if not self._colorbar_added:
            self.pg_view.ui.histogram.show()
            self._colorbar_added = True

    def set_title(self, text):
        """Set the title label, eliding each line in the middle. No-op without a label."""
        if self._title_label is None:
            return
        metrics = self._title_label.fontMetrics()
        elided = "\n".join(
            metrics.elidedText(line, Qt.ElideMiddle, self._title_elide_width)
            for line in text.split("\n")
        )
        self._title_label.setText(elided)
        self._title_label.setToolTip(text)

    def _transpose_checked(self):
        return self._transpose_checkbox is not None and self._transpose_checkbox.isChecked()

    def _log_checked(self):
        return self._log_checkbox is not None and self._log_checkbox.isChecked()

    # ------------------------------------------------------------------
    # zoom
    # ------------------------------------------------------------------

    def reset_zoom(self):
        """Fit the view to the image — same as pyqtgraph's built-in "View All"."""
        self.pg_view.getView().autoRange()

    def set_view_range(self, **kwargs):
        """Passthrough to the ViewBox, e.g. ``set_view_range(xRange=(0, 10))``."""
        self.pg_view.getView().setRange(**kwargs)

    def zoom_to_left_square(self, padding=0.05):
        """Zoom to the leftmost H x H square of a wide-and-short image."""
        if self._displayed_shape is None:
            return
        ny = self._displayed_shape[1]   # vertical height H
        self.pg_view.getView().setRange(xRange=(0, ny), yRange=(0, ny), padding=padding)

    # ------------------------------------------------------------------
    # overlays
    # ------------------------------------------------------------------

    def set_scatter_overlay(self, x, y):
        """Show a scatter overlay at the given image-pixel coordinates."""
        self._scatter_overlay.setData(x=x, y=y)
        self._scatter_overlay.setVisible(True)

    def clear_scatter_overlay(self):
        self._scatter_overlay.setVisible(False)

    @property
    def measure_points(self):
        return list(self._measure_clicks)

    def clear_measure_points(self, update_lineout=True):
        self._measure_clicks = []
        self._measure_scatter.setData(x=[], y=[])
        self._measure_line.setData([], [])
        self._distance_text.setVisible(False)
        self.sigPointsChanged.emit([])
        if update_lineout:
            self._update_lineout()

    # ------------------------------------------------------------------
    # lineout
    # ------------------------------------------------------------------

    def set_lineout_visible(self, visible: bool):
        if self.lineout is None:
            return
        self.lineout.set_lineout_visible(visible)
        if self.action_lineout is not None and self.action_lineout.isChecked() != visible:
            self.action_lineout.setChecked(visible)
        self._update_lineout()

    def _update_lineout(self):
        if self.lineout is None:
            return
        img_item = self.pg_view.getImageItem()
        image = None if img_item is None else img_item.image
        self.lineout.update_lineout(image, self._measure_clicks, self._pixel_size_m)

    # ------------------------------------------------------------------
    # filters
    # ------------------------------------------------------------------

    def set_filter(self, filter_type: str, checked: bool):
        """Enable/disable a display filter, prompting for its kernel width."""
        if not checked:
            self._active_filter = None
        else:
            kernel, ok = QtWidgets.QInputDialog.getDouble(
                self, f"{filter_type.capitalize()} Filter", "Kernel width:",
                value=self._filter_kernel, min=0.1, max=500.0, decimals=1
            )
            if not ok:
                # User cancelled — revert the checkmark
                action = (self.action_median_filter if filter_type == 'median'
                          else self.action_gaussian_filter)
                if action is not None:
                    action.setChecked(False)
                return
            self._filter_kernel = kernel
            self._active_filter = filter_type
            # Uncheck the other filter
            other = (self.action_gaussian_filter if filter_type == 'median'
                     else self.action_median_filter)
            if other is not None:
                other.setChecked(False)

        # Re-render from the cached array — no reload from the host
        self.redraw()
        self.sigRedrawRequested.emit()

    # ------------------------------------------------------------------
    # mouse
    # ------------------------------------------------------------------

    def _on_mouse_moved(self, event):
        pos = event[0]   # SignalProxy wraps args in a tuple
        img_item = self.pg_view.getImageItem()
        if img_item is None or img_item.image is None:
            return
        base = self._info_base_text
        if img_item.sceneBoundingRect().contains(pos):
            pt = img_item.mapFromScene(pos)
            nx, ny = img_item.image.shape[:2]
            x_um = (pt.x() - nx / 2) * self._pixel_size_m * 1e6
            y_um = (pt.y() - ny / 2) * self._pixel_size_m * 1e6
            xi, yi = int(pt.x()), int(pt.y())
            self.sigMouseMovedImage.emit(pt.x(), pt.y())
            if self._info_label is None:
                return
            if 0 <= xi < nx and 0 <= yi < ny:
                intens = img_item.image[xi, yi]
                self._info_label.setText(
                    f"{base}\nx={x_um:.2f}, y={y_um:.2f} µm, I={intens:.2g}"
                )
            else:
                self._info_label.setText(f"{base}\nx={x_um:.2f}, y={y_um:.2f} µm")
        elif self._info_label is not None:
            self._info_label.setText(base)

    def _on_mouse_clicked(self, event):
        if not self._enable_measure:
            return
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
            self.clear_measure_points()
            return

        self._measure_clicks.append((x, y))
        self._measure_scatter.setData(
            x=[p[0] for p in self._measure_clicks],
            y=[p[1] for p in self._measure_clicks],
        )
        self.sigPointsChanged.emit(list(self._measure_clicks))

        if len(self._measure_clicks) == 2:
            x1, y1 = self._measure_clicks[0]
            x2, y2 = self._measure_clicks[1]
            dist_um = np.sqrt(
                ((x2 - x1) * self._pixel_size_m * 1e6) ** 2 +
                ((y2 - y1) * self._pixel_size_m * 1e6) ** 2
            )
            self._measure_line.setData([x1, x2], [y1, y2])
            # Offset the label perpendicularly away from the line
            dx, dy = x2 - x1, y2 - y1
            seg_len = np.sqrt(dx ** 2 + dy ** 2)
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

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------

    @property
    def image_view(self):
        """The underlying ``pg.ImageView``."""
        return self.pg_view

    @property
    def view(self):
        """The underlying ``pg.ViewBox``."""
        return self.pg_view.getView()

    @property
    def image_item(self):
        """The underlying ``pg.ImageItem``."""
        return self.pg_view.getImageItem()

    @property
    def menu(self):
        """The ViewBox right-click ``QMenu``."""
        return self._menu

    @property
    def displayed_shape(self):
        """(nx, ny) of the displayed image, post-transpose. None before first image."""
        return self._displayed_shape

    @property
    def pixel_size_m(self):
        return self._pixel_size_m

    @property
    def raw_data(self):
        """The array last passed to :meth:`set_image`, before any display transform."""
        return self._raw_data

    @property
    def info_text(self):
        """The dimension/pixel-size string currently shown in the info label."""
        return self._info_base_text
