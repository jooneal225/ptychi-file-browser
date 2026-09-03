"""
Two-image comparison window.

See ``pg_image_tools/__init__.py`` for usage documentation.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from scipy.ndimage import zoom as ndi_zoom

# Pixel size ratios closer to 1 than this are treated as identical -- resampling
# by 1.0002 costs an interpolation pass and buys nothing.
RESAMPLE_TOLERANCE = 1e-3

# Gap between the two images in side-by-side mode, as a fraction of image 1's width
SIDE_BY_SIDE_GAP_FRAC = 0.03

IMAGE2_OPACITY = 0.5


class _DraggableImageItem(pg.ImageItem):
    """
    An ImageItem that reports left-button drags instead of acting on them.

    It never moves itself: it emits the drag delta in view coordinates and lets
    the owner decide what that means. In this window image 2's placement is not
    simply "where you dragged it" -- in subtract mode the item holds a
    difference whose extent grows as the two images pull apart -- so placement
    has to stay with the window.

    pyqtgraph hands drag events to whichever item claimed the button during the
    preceding hover, so ``hoverEvent`` has to call ``acceptDrags`` or the
    ViewBox takes the drag as a pan instead.
    """

    #: Emitted while dragging: (delta QPointF in view coords, is_start).
    #: The delta is measured from where the drag began, not from the last event.
    sigDragged = QtCore.pyqtSignal(object, bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.draggable = False
        self._drag_start_scene = None

    def hoverEvent(self, ev):
        if not ev.isExit() and self.draggable and self.image is not None:
            ev.acceptDrags(Qt.LeftButton)

    def mouseDragEvent(self, ev):
        if not self.draggable or ev.button() != Qt.LeftButton or self.image is None:
            ev.ignore()
            return
        ev.accept()

        view = self.getViewBox()
        if view is None:
            return
        if ev.isStart():
            self._drag_start_scene = ev.buttonDownScenePos()
        if self._drag_start_scene is None:
            return

        # Map through the view so the image tracks the cursor at any zoom level
        delta = (view.mapSceneToView(ev.scenePos())
                 - view.mapSceneToView(self._drag_start_scene))
        self.sigDragged.emit(QtCore.QPointF(delta), ev.isStart())


class ImageCompareWindow(QtWidgets.QWidget):
    """
    A standalone, non-blocking window for overlaying two images pulled from an
    :class:`~pg_image_tools.ImagePlotWidget`.

    It opens empty. "Set image 1" and "Set image 2" each take a snapshot of
    whatever the source widget is displaying at that moment, together with its
    pixel size. Image 1 is the ground truth: it is drawn solid at the origin and
    its pixel size defines the common grid, so image 2 is resampled onto it when
    the two differ. Image 2 is drawn semi-transparent on top and can be dragged
    into alignment.

    Each image has its own histogram, so levels and colormap are set
    independently. ``Subtract`` leaves image 1 alone and swaps the image 2 slot
    -- and only that slot -- for ``multiplier × image 2 - image 1``, taken over
    the union of the two, so image 2 stays draggable and the difference updates
    as it moves.
    """

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("Image Comparison")

        self._source = source

        # Each slot is None until its button is pressed, else
        # {'data': 2d ndarray, 'pixel_size_m': float, 'label': str}
        self._slots = [None, None]
        self._grid_data = [None, None]            # slots resampled onto the common grid
        self._displayed = [None, None]            # what each item actually shows
        self._image2_origin = (0.0, 0.0)          # where the image 2 item is drawn
        self._offset = QtCore.QPointF(0.0, 0.0)   # image 2 drag offset, grid pixels
        self._drag_base = None                    # offset at the start of a drag
        self._hide1_before_subtract = False       # restored when subtract goes off
        self._grid_pixel_size_m = 1.0
        self._resample_factor = 1.0

        self._build_ui()
        self._rebuild()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # --- plot plus one histogram per image ---
        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw, stretch=1)

        self._plot = self._glw.addPlot(row=0, col=0)
        self._plot.setAspectLocked(True)
        self._plot.invertY(True)    # match the orientation pg.ImageView uses

        self._img1 = pg.ImageItem()
        self._img1.setZValue(0)
        self._plot.addItem(self._img1)

        self._img2 = _DraggableImageItem()
        self._img2.setZValue(1)
        self._img2.sigDragged.connect(self._on_image2_dragged)
        self._plot.addItem(self._img2)

        self._hist1 = pg.HistogramLUTItem()
        self._hist1.setImageItem(self._img1)
        self._glw.addItem(self._hist1, row=0, col=1)

        self._hist2 = pg.HistogramLUTItem()
        self._hist2.setImageItem(self._img2)
        self._glw.addItem(self._hist2, row=0, col=2)

        self._mouse_move_proxy = pg.SignalProxy(
            self._glw.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )

        # --- readouts ---
        self._info_label = QtWidgets.QLabel()
        self._info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._info_label)

        self._cursor_label = QtWidgets.QLabel()
        self._cursor_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._cursor_label)

        # --- buttons ---
        btn_row = QtWidgets.QHBoxLayout()
        self._btn_set1 = QtWidgets.QPushButton("Set image 1")
        self._btn_set1.setToolTip("Copy what the source plot is showing into image 1 (ground truth)")
        self._btn_set1.clicked.connect(lambda: self.capture(0))
        self._btn_set2 = QtWidgets.QPushButton("Set image 2")
        self._btn_set2.setToolTip("Copy what the source plot is showing into image 2 (overlay)")
        self._btn_set2.clicked.connect(lambda: self.capture(1))
        self._btn_reset = QtWidgets.QPushButton("Reset offset")
        self._btn_reset.clicked.connect(self.reset_offset)

        btn_row.addWidget(self._btn_set1)
        btn_row.addWidget(self._btn_set2)
        btn_row.addWidget(self._btn_reset)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        # --- checkboxes ---
        chk_row = QtWidgets.QHBoxLayout()
        self._chk_side_by_side = QtWidgets.QCheckBox("Side-by-side comparison")
        self._chk_side_by_side.setToolTip(
            "Draw both images solid, placed horizontally adjacent instead of overlaid"
        )
        self._chk_hide1 = QtWidgets.QCheckBox("Hide image 1")
        self._chk_hide2 = QtWidgets.QCheckBox("Hide image 2")
        for chk in (self._chk_side_by_side, self._chk_hide1, self._chk_hide2):
            chk.toggled.connect(self._rebuild)
            chk_row.addWidget(chk)

        # Not in the loop above: it has to lock out side-by-side and re-level
        # image 2, whose range changes completely when it becomes a difference.
        self._chk_subtract = QtWidgets.QCheckBox("Subtract")
        self._chk_subtract.setToolTip(
            "Leave image 1 alone and plot (multiplier × image 2) - image 1 in the\n"
            "image 2 slot, over the union of the two. Image 2 stays draggable."
        )
        self._chk_subtract.toggled.connect(self._on_subtract_toggled)
        chk_row.insertWidget(1, self._chk_subtract)

        chk_row.addSpacing(12)
        chk_row.addWidget(QtWidgets.QLabel("multiplier"))
        self._edit_multiplier = QtWidgets.QLineEdit("1.0")
        self._edit_multiplier.setToolTip("Image 2 is scaled by this before the subtraction")
        self._edit_multiplier.setValidator(QDoubleValidator())
        self._edit_multiplier.setMaximumWidth(80)
        self._edit_multiplier.editingFinished.connect(self._on_multiplier_changed)
        chk_row.addWidget(self._edit_multiplier)

        chk_row.addStretch(1)
        layout.addLayout(chk_row)

        self.resize(900, 700)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def capture(self, index: int):
        """Pull the source widget's current image into slot ``index`` (0 or 1)."""
        data, pixel_size_m = self._read_source()
        if data is None:
            self._info_label.setText("Source plot has no image to copy.")
            return

        self._slots[index] = {
            'data': np.array(data, dtype=np.float32),   # snapshot, not a live reference
            'pixel_size_m': float(pixel_size_m),
            'label': self._source.info_text,
        }
        self._refresh_grid()
        self._rebuild(autorange=True)
        self._sync_levels(index)

    def reset_offset(self):
        """Put image 2 back at the origin of the common grid."""
        self._offset = QtCore.QPointF(0.0, 0.0)
        self._rebuild()

    @property
    def offset(self):
        """Image 2's drag offset as an (x, y) tuple in common-grid pixels."""
        return self._offset.x(), self._offset.y()

    @property
    def multiplier(self):
        """The subtraction's image 2 scale factor; 1.0 if the field is unusable."""
        try:
            return float(self._edit_multiplier.text())
        except ValueError:
            return 1.0

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------

    def _read_source(self):
        """(data, pixel_size_m) as currently displayed by the source widget."""
        item = self._source.image_item
        data = None if item is None else item.image
        if data is None:
            data = self._source.raw_data
        if data is None:
            return None, 1.0
        return data, self._source.pixel_size_m

    def _refresh_grid(self):
        """
        Re-derive the common grid and resample the slots onto it.

        Split out from :meth:`_rebuild` because it is the expensive step and
        only a new snapshot can change its result -- dragging must not pay for
        an interpolation pass on every mouse move.
        """
        slot1, slot2 = self._slots

        # Image 1's pixel size is the grid; with no image 1, image 2 keeps its own
        if slot1 is not None:
            self._grid_pixel_size_m = slot1['pixel_size_m']
        elif slot2 is not None:
            self._grid_pixel_size_m = slot2['pixel_size_m']
        else:
            self._grid_pixel_size_m = 1.0

        if slot2 is not None and self._grid_pixel_size_m > 0:
            self._resample_factor = slot2['pixel_size_m'] / self._grid_pixel_size_m
        else:
            self._resample_factor = 1.0

        self._grid_data = [self._resample(0), self._resample(1)]

    def _resample(self, index):
        """
        Slot ``index``'s array on the common grid, or None if the slot is empty.

        Only image 2 is ever resampled; image 1 defines the grid.
        """
        slot = self._slots[index]
        if slot is None:
            return None
        if index == 0 or abs(self._resample_factor - 1.0) <= RESAMPLE_TOLERANCE:
            return slot['data']
        return ndi_zoom(slot['data'], self._resample_factor, order=1)

    def _difference(self, data1, data2):
        """
        ``multiplier × data2 - data1`` over the union of the two.

        Both are already on the common grid, but they need not have the same
        shape and image 2 may have been dragged, so the result spans the
        bounding box of both, each contributing zero where it does not reach.
        Misalignment therefore shows up as the images' edges standing out
        against each other rather than being cropped away. The offset is rounded
        to whole pixels; the union's corner goes in ``_image2_origin`` so the
        result can be drawn back in image 1's coordinates.
        """
        if data1 is None or data2 is None:
            return None

        off_x, off_y = int(round(self._offset.x())), int(round(self._offset.y()))
        x0, y0 = min(0, off_x), min(0, off_y)
        x1 = max(data1.shape[0], off_x + data2.shape[0])
        y1 = max(data1.shape[1], off_y + data2.shape[1])

        out = np.zeros((x1 - x0, y1 - y0), dtype=np.float32)
        ax, ay = off_x - x0, off_y - y0
        out[ax:ax + data2.shape[0], ay:ay + data2.shape[1]] = data2 * self.multiplier
        out[-x0:-x0 + data1.shape[0], -y0:-y0 + data1.shape[1]] -= data1

        self._image2_origin = (x0, y0)
        return out

    # ------------------------------------------------------------------
    # display
    # ------------------------------------------------------------------

    def _rebuild(self, *_args, autorange=False):
        """
        Recompute what each item shows and where, from the current grid.

        Called for every change -- a new snapshot, a checkbox, the multiplier, a
        drag -- so the difference and the placements never go stale.
        """
        subtract = self._chk_subtract.isChecked()
        side_by_side = self._chk_side_by_side.isChecked() and not subtract

        # Image 1 is left alone either way; only the image 2 slot becomes the
        # difference, so the two histograms keep their separate meanings.
        self._displayed = list(self._grid_data)
        if subtract:
            self._displayed[1] = self._difference(*self._grid_data)
        elif side_by_side:
            # Adjacent, both solid. Shapes need not match -- nothing is padded.
            width1 = self._grid_data[0].shape[0] if self._grid_data[0] is not None else 0
            self._image2_origin = (width1 * (1 + SIDE_BY_SIDE_GAP_FRAC), 0.0)
        else:
            self._image2_origin = (self._offset.x(), self._offset.y())

        data1, data2 = self._displayed
        for item, data in ((self._img1, data1), (self._img2, data2)):
            if data is None:
                item.clear()
            else:
                # Auto-level only the very first array an item is given; after
                # that the histogram is the user's, and a rebuild triggered by a
                # checkbox or a drag must not throw their levels away.
                item.setImage(data, autoLevels=item.levels is None)

        show1 = data1 is not None and not self._chk_hide1.isChecked()

        self._img1.setPos(0.0, 0.0)
        self._img2.setPos(*self._image2_origin)
        # Image 2 is only ever see-through when there is something to see
        # through it to -- solid side by side, and solid once image 1 is hidden.
        self._img2.setOpacity(IMAGE2_OPACITY if show1 and not side_by_side else 1.0)
        self._img2.draggable = not side_by_side

        self._img1.setVisible(show1)
        self._img2.setVisible(data2 is not None and not self._chk_hide2.isChecked())

        self._chk_side_by_side.setEnabled(not subtract)
        self._btn_reset.setEnabled(not side_by_side)

        if autorange:
            self._plot.getViewBox().autoRange()
        self._update_info()

    def _sync_levels(self, index):
        """
        Put one image's histogram and levels on its array's true range.

        Same reasoning as ``ImagePlotWidget._sync_levels``: leaving it to
        autoLevels lets the histogram's range drift out of step with the levels
        after a few images.
        """
        data = self._displayed[index]
        if data is None:
            return
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = lo + 1.0   # a flat image still needs a non-empty range

        hist = self._hist1 if index == 0 else self._hist2
        hist.setHistogramRange(lo, hi)
        hist.setLevels(lo, hi)

    # ------------------------------------------------------------------
    # interaction
    # ------------------------------------------------------------------

    def _on_subtract_toggled(self, checked):
        """
        Subtract owns the layout: side-by-side is cleared and locked out, and
        image 1 is hidden, the difference already carrying it. Both remain the
        user's to change afterwards; turning subtract back off restores whatever
        "Hide image 1" was before.
        """
        if checked:
            self._hide1_before_subtract = self._chk_hide1.isChecked()
            self._chk_side_by_side.setChecked(False)   # each may rebuild on its own
            self._chk_hide1.setChecked(True)
        else:
            self._chk_hide1.setChecked(self._hide1_before_subtract)
        self._rebuild(autorange=True)
        # The difference straddles zero -- nothing like image 2's own range
        self._sync_levels(1)

    def _on_multiplier_changed(self):
        self._rebuild()
        if self._chk_subtract.isChecked():
            self._sync_levels(1)

    def _on_image2_dragged(self, delta, is_start):
        """Apply a drag delta to the offset. Placement is _rebuild's business."""
        if is_start or self._drag_base is None:
            self._drag_base = QtCore.QPointF(self._offset)
        self._offset = self._drag_base + delta
        self._rebuild()

    def _on_mouse_moved(self, event):
        pos = event[0]   # SignalProxy wraps args in a tuple
        if not self._plot.sceneBoundingRect().contains(pos):
            self._cursor_label.setText('')
            return
        pt = self._plot.getViewBox().mapSceneToView(pos)

        # Report position relative to image 1's centre, as the source widget does
        data1 = self._displayed[0]
        cx = data1.shape[0] / 2 if data1 is not None else 0.0
        cy = data1.shape[1] / 2 if data1 is not None else 0.0
        x_um = (pt.x() - cx) * self._grid_pixel_size_m * 1e6
        y_um = (pt.y() - cy) * self._grid_pixel_size_m * 1e6

        # Topmost visible image under the cursor wins, i.e. what you are looking at
        for name, item, data in (("image 2", self._img2, self._displayed[1]),
                                 ("image 1", self._img1, self._displayed[0])):
            if data is None or not item.isVisible():
                continue
            ix = int(np.floor(pt.x() - item.pos().x()))
            iy = int(np.floor(pt.y() - item.pos().y()))
            if 0 <= ix < data.shape[0] and 0 <= iy < data.shape[1]:
                self._cursor_label.setText(
                    "x=%.2f, y=%.2f um, I=%.2e  [%s]" % (x_um, y_um, data[ix, iy], name)
                )
                return

        self._cursor_label.setText("x=%.2f, y=%.2f um" % (x_um, y_um))

    # ------------------------------------------------------------------
    # readout
    # ------------------------------------------------------------------

    def _update_info(self):
        lines = []
        for i, slot in enumerate(self._slots):
            name = "Image %d%s" % (i + 1, " (ground truth)" if i == 0 else "")
            if slot is None or self._grid_data[i] is None:
                lines.append(f"{name}: not set")
                continue
            shape = self._grid_data[i].shape
            text = "%s: %d×%d px, %d nm pix" % (
                name, shape[0], shape[1], round(slot['pixel_size_m'] * 1e9)
            )
            if i == 1 and abs(self._resample_factor - 1.0) > RESAMPLE_TOLERANCE:
                text += " → resampled ×%.4g onto image 1's grid" % self._resample_factor
            lines.append(text)

        if self._chk_subtract.isChecked():
            diff = self._displayed[1]
            if diff is None:
                lines.append("Subtract: needs both images")
            else:
                finite = diff[np.isfinite(diff)]
                extent = ("range %.4g to %.4g" % (float(finite.min()), float(finite.max()))
                          if finite.size else "no finite values")
                lines.append(
                    "Subtract: %.4g × image 2 - image 1 over %d×%d px, %s"
                    % (self.multiplier, diff.shape[0], diff.shape[1], extent)
                )

        dx, dy = self._offset.x(), self._offset.y()
        offset_text = "Image 2 offset: %.1f, %.1f px  (%.3f, %.3f µm)" % (
            dx, dy,
            dx * self._grid_pixel_size_m * 1e6,
            dy * self._grid_pixel_size_m * 1e6,
        )
        if not self._img2.draggable:
            offset_text += "  — dragging disabled"
        lines.append(offset_text)

        self._info_label.setText("\n".join(lines))
