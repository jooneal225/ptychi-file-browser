"""
Lineout panel with a draggable 25%-75% resolution metric.

See ``pg_image_tools/__init__.py`` for usage documentation.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from scipy.ndimage import map_coordinates


def compute_lineout(image, p1, p2, pixel_size_m):
    """
    Sample ``image`` along the segment p1 -> p2.

    Parameters
    ----------
    image : 2d ndarray, indexed [x, y] (pyqtgraph convention)
    p1, p2 : (x, y) tuples in image pixel coordinates
    pixel_size_m : float, metres per pixel

    Returns
    -------
    dist_um : 1d ndarray of distances along the segment, in µm
    values : 1d ndarray of interpolated image values
    """
    if image is None:
        return None, None
    x1, y1 = p1
    x2, y2 = p2
    n_pts = max(int(np.hypot(x2 - x1, y2 - y1)), 10)
    xs = np.linspace(x1, x2, n_pts)
    ys = np.linspace(y1, y2, n_pts)
    values = map_coordinates(image, [xs, ys], order=1, mode='nearest')
    total_um = np.hypot((x2 - x1) * pixel_size_m * 1e6, (y2 - y1) * pixel_size_m * 1e6)
    dist_um = np.linspace(0, total_um, n_pts)
    return dist_um, values


class LineoutPanel(pg.PlotWidget):
    """
    A pyqtgraph plot of an image lineout, with two draggable vertical cursors
    (blue / green) that report their positions and the value under them, plus a
    right-click "Find 25%-75% resolution" action that measures the edge width
    between them.

    The panel is driven entirely by :meth:`update_lineout`; it never reads data
    itself. It is hidden by default -- call :meth:`set_lineout_visible`.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setLabel('bottom', 'Distance (µm)')
        self.setLabel('left', 'Value')
        self.setVisible(False)

        # Explicit visibility flag rather than isVisible(), so updates issued
        # before the host window is shown still take effect.
        self._visible = False

        self._dist_um = None
        self._values = None
        self._res_lines_positioned = False
        self._res_nm = None
        self._res_region = None

        self._res_line1 = pg.InfiniteLine(
            pos=0.0, angle=90, movable=True,
            pen=pg.mkPen(color=(80, 120, 255), width=1.5, style=Qt.DotLine)
        )
        self._res_line2 = pg.InfiniteLine(
            pos=1.0, angle=90, movable=True,
            pen=pg.mkPen(color=(80, 220, 80), width=1.5, style=Qt.DotLine)
        )
        self._res_line1.sigPositionChanged.connect(self._on_res_line_moved)
        self._res_line2.sigPositionChanged.connect(self._on_res_line_moved)

        self._res_text_item = pg.TextItem(text='', color='w', anchor=(1, 0),
                                          fill=pg.mkBrush(0, 0, 0, 180))
        self._res_text_item.setFont(QFont("Courier", 9))
        self.getViewBox().sigRangeChanged.connect(self._reposition_res_text)

        self.find_resolution_action = QtWidgets.QAction("Find 25%-75% resolution")
        self.find_resolution_action.triggered.connect(self.find_resolution)
        self.getViewBox().menu.addSeparator()
        self.getViewBox().menu.addAction(self.find_resolution_action)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def lineout_visible(self):
        return self._visible

    def set_lineout_visible(self, visible: bool):
        self._visible = bool(visible)
        self.setVisible(self._visible)

    def update_lineout(self, image, points, pixel_size_m):
        """
        Redraw from ``image`` sampled between the two entries of ``points``.

        ``points`` is a sequence of 0, 1 or 2 (x, y) tuples; with fewer than two
        the panel simply clears. No-op while the panel is hidden.
        """
        if not self._visible:
            return

        self.clear()
        self._res_nm = None       # both were removed by clear()
        self._res_region = None

        if image is None or points is None or len(points) < 2:
            self._dist_um = None
            self._values = None
            return

        dist_um, values = compute_lineout(image, points[0], points[1], pixel_size_m)
        if dist_um is None:
            return

        self._dist_um = dist_um
        self._values = values
        self.plot(dist_um, values, pen=pg.mkPen('w', width=1))

        marker_pen = pg.mkPen('r', width=1, style=Qt.DashLine)
        self.addItem(pg.InfiniteLine(pos=dist_um[0], angle=90, pen=marker_pen))
        self.addItem(pg.InfiniteLine(pos=dist_um[-1], angle=90, pen=marker_pen))

        # Position metric lines on first use, then keep user-dragged positions
        if not self._res_lines_positioned:
            span = dist_um[-1] - dist_um[0]
            self._res_line1.setPos(dist_um[0] + span / 3)
            self._res_line2.setPos(dist_um[0] + 2 * span / 3)
            self._res_lines_positioned = True

        # Re-add after clear() — ignoreBounds keeps them out of autoscale
        self.addItem(self._res_line1, ignoreBounds=True)
        self.addItem(self._res_line2, ignoreBounds=True)
        self.addItem(self._res_text_item, ignoreBounds=True)
        self._update_res_metric()

    def find_resolution(self):
        """Measure the 25%-75% crossing width between the two cursors."""
        if self._dist_um is None or self._values is None:
            return
        xb = self._res_line1.value()
        xg = self._res_line2.value()
        yb = float(np.interp(xb, self._dist_um, self._values))
        yg = float(np.interp(xg, self._dist_um, self._values))

        lo, hi = min(xb, xg), max(xb, xg)
        mask = (self._dist_um >= lo) & (self._dist_um <= hi)
        x_slice = self._dist_um[mask]
        y_slice = self._values[mask]
        if len(x_slice) < 2:
            return

        level_25 = yb + 0.25 * (yg - yb)
        level_75 = yb + 0.75 * (yg - yb)

        # np.interp requires monotonically increasing xp — sort by y value
        if y_slice[-1] >= y_slice[0]:
            x_25 = float(np.interp(level_25, y_slice, x_slice))
            x_75 = float(np.interp(level_75, y_slice, x_slice))
        else:
            x_25 = float(np.interp(level_25, y_slice[::-1], x_slice[::-1]))
            x_75 = float(np.interp(level_75, y_slice[::-1], x_slice[::-1]))

        # Drop any previous result first — _clear_resolution() also nulls
        # _res_nm, so the new value has to be assigned after it, not before.
        self._clear_resolution()

        self._res_nm = int(round(abs(x_75 - x_25) * 1e3))

        # Shade the region between the two crossings
        self._res_region = pg.LinearRegionItem(
            values=[min(x_25, x_75), max(x_25, x_75)],
            brush=pg.mkBrush(160, 0, 200, 70),
            movable=False,
        )
        self.addItem(self._res_region, ignoreBounds=True)
        self._update_res_metric()

    @property
    def resolution_nm(self):
        """Last computed 25%-75% width in nm, or None."""
        return self._res_nm

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _on_res_line_moved(self):
        """Called when a metric line is dragged — clears any computed resolution."""
        self._clear_resolution()
        self._update_res_metric()

    def _clear_resolution(self):
        if self._res_region is not None:
            try:
                self.removeItem(self._res_region)
            except Exception:
                pass
            self._res_region = None
        self._res_nm = None

    def _update_res_metric(self):
        if self._dist_um is None or self._values is None:
            return
        xb = self._res_line1.value()
        xg = self._res_line2.value()
        yb = float(np.interp(xb, self._dist_um, self._values))
        yg = float(np.interp(xg, self._dist_um, self._values))
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

    def _reposition_res_text(self):
        vr = self.getViewBox().viewRange()
        x_min, x_max = vr[0]
        y_min, y_max = vr[1]
        mx = (x_max - x_min) * 0.01
        my = (y_max - y_min) * 0.02
        self._res_text_item.setPos(x_max - mx, y_max - my)
