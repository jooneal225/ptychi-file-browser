"""
ImageStackWindow -- a non-blocking builder for combining several same-type
images (e.g. object_mag_NiterXXX.tiff across many scans) into a single 3D
stack, optionally resampling everything onto a common pixel size and
cropping to a common shape, then saving as .h5 or .tiff.

Each image is cached at full size, resampled but never cropped. Cropping,
phase unwrapping and the relative shifts from "Align images" are all applied
on the way out -- in _display_frame() -- so the preview and the saved stack
always agree and nothing is ever baked destructively into the cache.
"""

from pathlib import Path

import numpy as np
import h5py
import tifffile
import pyqtgraph as pg
from scipy.ndimage import zoom as ndi_zoom
from scipy.ndimage import shift as ndi_shift

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QSettings

from pg_image_tools import ImagePlotWidget

# Matches pg_image_tools/_compare.py's RESAMPLE_TOLERANCE: skip an
# interpolation pass when pixel sizes already agree to within this fraction.
RESAMPLE_TOLERANCE = 1e-3

# The alignment ROI's initial extent, as a fraction of the displayed image
ALIGN_ROI_FRAC = 0.5

# Sub-pixel resolution of the phase correlation: 1/10 pixel.
ALIGN_UPSAMPLE = 10

ALIGN_MODES = [
    "Align adjacent",
    "Align to first",
    "Align to middle",
    "Align to end",
    "Align to mean",
]


def _resample_to_pixel_size(data, src_pixel_size_m, dst_pixel_size_m):
    """Resample a 2D array from src_pixel_size_m onto dst_pixel_size_m's grid."""
    if dst_pixel_size_m <= 0:
        return data
    factor = src_pixel_size_m / dst_pixel_size_m
    if abs(factor - 1.0) <= RESAMPLE_TOLERANCE:
        return data
    return ndi_zoom(data, factor, order=1)


def _center_crop(data, target_shape):
    """Crop a 2D array to target_shape, trimming evenly from both sides."""
    h, w = data.shape
    th, tw = target_shape
    top = max((h - th) // 2, 0)
    left = max((w - tw) // 2, 0)
    return data[top:top + th, left:left + tw]


def _extra_center_crop(data, crop_x, crop_y):
    """
    Trim crop_x pixels from the top and bottom and crop_y pixels from the
    left and right, on top of whatever cropping already happened.
    """
    h, w = data.shape
    top = min(crop_x, h // 2)
    left = min(crop_y, w // 2)
    if top == 0 and left == 0:
        return data
    return data[top:h - top, left:w - left]


def _apply_shift(data, shift_xy):
    """
    Translate a 2D array by (dx, dy), where dx is along axis 0 and dy along
    axis 1 -- the same (x, y) the image plot's cursor readout uses.

    Edges are filled by replicating the border rather than with zeros or NaN:
    a black band would drag the saved stack's min/max around, and a NaN band
    would poison the color levels.
    """
    dx, dy = float(shift_xy[0]), float(shift_xy[1])
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return data
    return ndi_shift(data.astype(np.float32), (dx, dy), order=1, mode="nearest")


def _phase_correlation_shift(reference, moving):
    """
    (dx, dy) that :func:`_apply_shift` must apply to ``moving`` to register it
    onto ``reference``, from sub-pixel phase correlation.
    """
    from skimage.registration import phase_cross_correlation

    result = phase_cross_correlation(
        np.nan_to_num(reference), np.nan_to_num(moving),
        upsample_factor=ALIGN_UPSAMPLE, normalization="phase",
    )
    # Older/newer scikit-image differ on whether the error and phase
    # difference come back alongside the shift.
    shift = result[0] if isinstance(result, tuple) else result
    return np.asarray(shift, dtype=float)


def _print_progress(label, done, total, width=30):
    """One-line, self-overwriting terminal progress bar."""
    total = max(int(total), 1)
    done = min(int(done), total)
    filled = int(round(width * done / total))
    print(
        f"\r[Image Stack] {label} [{'#' * filled}{'.' * (width - filled)}] {done}/{total}",
        end="\n" if done >= total else "",
        flush=True,
    )


class ImageStackWindow(QtWidgets.QDialog):
    """
    Non-blocking window (same pattern as RuntableWindow): a real, independent
    top-level window rather than a modal popup.
    """

    def __init__(self, browser, extension, base_path: Path, seed_path):
        super().__init__(browser)
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("Create Image Stack")

        self._browser = browser
        self._extension = extension
        self._base_path = base_path
        # Each entry keeps "data" (exactly as read from disk) and "cached" (the
        # full-size image the rest of the window works from: "data" resampled
        # onto the reference pixel size when "Resample" is on, "data" itself
        # otherwise). Neither is ever cropped, shifted or unwrapped -- those are
        # applied on the way out, in _display_frame().
        self._entries = []          # [{"path": Path, "pixel_size_m": float, "data", "cached"}]
        self._target_shape = None   # recomputed on every list change / resample toggle

        # ---- alignment state ----
        # The one place the relative shifts live: an (N, 2) float array of
        # (dx, dy) in cached-image pixels, or None when nothing is aligned.
        self._shifts = None
        self._align_roi = None      # pg.RectROI while the alignment bar is open
        self._unwrap_patches = None # per entry: the unwrapped ROI region, or None
        self._unwrap_slices = None  # per entry: (x0, x1, y0, y1) those patches go back into

        self._build_ui()

        if seed_path is not None and Path(seed_path).exists():
            self._add_paths([Path(seed_path)])

        geom = QSettings("temp", "PtychiFileBrowser").value("image_stack_window_geometry")
        if geom is not None:
            self.restoreGeometry(geom)
        else:
            self.resize(900, 600)


    def closeEvent(self, event):
        QSettings("temp", "PtychiFileBrowser").setValue(
            "image_stack_window_geometry", self.saveGeometry()
        )
        super().closeEvent(event)


    # ------------------------------------------------------------------
    # ui
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        splitter = QtWidgets.QSplitter(Qt.Horizontal)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        splitter.addWidget(self.list_widget)

        self.plot = ImagePlotWidget(
            enable_lineout=False, enable_measure=False,
            enable_filters=False, enable_compare=False,
        )

        # Shift-vs-index plot, stacked under the image and hidden until the
        # "Open alignment shifts" right-click entry asks for it.
        self.shift_plot = pg.PlotWidget()
        self.shift_plot.setLabel("bottom", "Image index")
        self.shift_plot.setLabel("left", "Shift (pixels)")
        self.shift_plot.showGrid(x=True, y=True, alpha=0.3)
        self.shift_plot.addLegend(offset=(-10, 10))

        self._current_index_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("k", width=1, style=Qt.DashLine)
        )
        self.shift_plot.addItem(self._current_index_line)

        self.plot_splitter = QtWidgets.QSplitter(Qt.Vertical)
        self.plot_splitter.addWidget(self.plot)
        self.plot_splitter.addWidget(self.shift_plot)
        self.plot_splitter.setStretchFactor(0, 3)
        self.plot_splitter.setStretchFactor(1, 1)
        self.shift_plot.hide()   # after addWidget(): QSplitter shows what it adopts
        splitter.addWidget(self.plot_splitter)

        self.action_show_shifts = self.plot.add_menu_action(
            "Open alignment shifts", self._on_show_shifts_toggled,
            checkable=True, at_top=True,
        )

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([300, 600])

        layout.addWidget(splitter, stretch=1)

        controls = QtWidgets.QHBoxLayout()

        add_btn = QtWidgets.QPushButton("+")
        add_btn.setStyleSheet("background-color: green; color: white; font-weight: bold;")
        add_btn.setToolTip("Add the image currently selected in the main window")
        add_btn.clicked.connect(self._on_add_clicked)
        controls.addWidget(add_btn)

        remove_btn = QtWidgets.QPushButton("-")
        remove_btn.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        remove_btn.clicked.connect(self._on_remove_clicked)
        controls.addWidget(remove_btn)

        pattern_btn = QtWidgets.QPushButton("Add scan range by image pattern")
        pattern_btn.clicked.connect(self._on_add_scan_range_clicked)
        controls.addWidget(pattern_btn)

        up_btn = QtWidgets.QPushButton("↑")
        up_btn.setStyleSheet("background-color: #fff9c4;")
        up_btn.clicked.connect(lambda: self._move_selected(-1))
        controls.addWidget(up_btn)

        down_btn = QtWidgets.QPushButton("↓")
        down_btn.setStyleSheet("background-color: #fff9c4;")
        down_btn.clicked.connect(lambda: self._move_selected(1))
        controls.addWidget(down_btn)

        self.checkBox_resample = QtWidgets.QCheckBox("Resample")
        self.checkBox_resample.toggled.connect(self._on_resample_toggled)
        controls.addWidget(self.checkBox_resample)

        controls.addWidget(QtWidgets.QLabel("Crop X"))
        self.spinBox_crop_x = QtWidgets.QSpinBox()
        self.spinBox_crop_x.setRange(0, 9999)
        self.spinBox_crop_x.valueChanged.connect(self._on_crop_changed)
        controls.addWidget(self.spinBox_crop_x)

        controls.addWidget(QtWidgets.QLabel("Crop Y"))
        self.spinBox_crop_y = QtWidgets.QSpinBox()
        self.spinBox_crop_y.setRange(0, 9999)
        self.spinBox_crop_y.valueChanged.connect(self._on_crop_changed)
        controls.addWidget(self.spinBox_crop_y)

        self.btn_align_images = QtWidgets.QPushButton("Align images")
        self.btn_align_images.setToolTip(
            "Pick a region and register every image onto a reference with phase correlation"
        )
        self.btn_align_images.clicked.connect(self._on_align_images_clicked)
        controls.addWidget(self.btn_align_images)

        self.btn_undo_align = QtWidgets.QPushButton("Undo alignment")
        self.btn_undo_align.setEnabled(False)
        self.btn_undo_align.clicked.connect(self._on_undo_alignment_clicked)
        controls.addWidget(self.btn_undo_align)

        controls.addStretch(1)

        save_btn = QtWidgets.QPushButton("Save as")
        save_btn.clicked.connect(self._on_save_as_clicked)
        controls.addWidget(save_btn)

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.close)
        controls.addWidget(cancel_btn)

        layout.addLayout(controls)
        layout.addWidget(self._build_align_bar())


    def _build_align_bar(self):
        """The second controls row, shown only while alignment mode is active."""
        self.align_bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.align_bar)
        row.setContentsMargins(0, 0, 0, 0)

        hint = QtWidgets.QLabel("Drag/resize the red box to set the region to correlate over.")
        hint.setStyleSheet("color: red;")
        row.addWidget(hint)

        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("Mode"))
        self.comboBox_align_mode = QtWidgets.QComboBox()
        self.comboBox_align_mode.addItems(ALIGN_MODES)
        self.comboBox_align_mode.setToolTip("Which pairs of images get phase-correlated")
        row.addWidget(self.comboBox_align_mode)

        row.addStretch(1)

        align_btn = QtWidgets.QPushButton("Align")
        align_btn.clicked.connect(self._on_align_clicked)
        row.addWidget(align_btn)

        self.btn_unwrap_phase = QtWidgets.QPushButton("Unwrap phase")
        self.btn_unwrap_phase.setCheckable(True)
        self.btn_unwrap_phase.setToolTip("Unwrap every image inside the red box only")
        self.btn_unwrap_phase.toggled.connect(self._on_unwrap_phase_toggled)
        row.addWidget(self.btn_unwrap_phase)

        align_cancel_btn = QtWidgets.QPushButton("Cancel")
        align_cancel_btn.setToolTip(
            "Leave alignment mode (keeps any shifts / unwrapping already applied)"
        )
        align_cancel_btn.clicked.connect(self._exit_align_mode)
        row.addWidget(align_cancel_btn)

        self.align_bar.hide()
        return self.align_bar


    # ------------------------------------------------------------------
    # list management
    # ------------------------------------------------------------------

    def _relative_label(self, path: Path) -> str:
        try:
            return str(path.relative_to(self._base_path))
        except ValueError:
            return str(path)


    def _add_paths(self, paths):
        errors = []
        for path in paths:
            try:
                data, pixel_size_m, _ = self._browser.read_image_and_pixel_size(path, self._extension)
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
                continue

            self._entries.append({
                "path": path,
                "pixel_size_m": pixel_size_m if pixel_size_m is not None else 1.0,
                "data": data,
                "cached": data,
            })
            self.list_widget.addItem(self._relative_label(path))

        self._on_entries_changed()

        if errors:
            QtWidgets.QMessageBox.warning(
                self, "Some images could not be loaded", "\n".join(errors)
            )

        if self.list_widget.count() and self.list_widget.currentRow() < 0:
            self.list_widget.setCurrentRow(0)


    def _on_add_clicked(self):
        """Add whatever image is currently selected/displayed in the main window."""
        path = self._browser.file_load_path
        if path is None or not Path(path).exists():
            QtWidgets.QMessageBox.warning(
                self, "No image selected",
                "Select an image in the main window first."
            )
            return
        self._add_paths([Path(path)])


    def _on_remove_clicked(self):
        rows = sorted({idx.row() for idx in self.list_widget.selectedIndexes()}, reverse=True)
        for row in rows:
            self.list_widget.takeItem(row)
            del self._entries[row]
        self._on_entries_changed()


    def _move_selected(self, offset):
        row = self.list_widget.currentRow()
        new_row = row + offset
        if row < 0 or not (0 <= new_row < self.list_widget.count()):
            return

        self._entries[row], self._entries[new_row] = self._entries[new_row], self._entries[row]

        item = self.list_widget.takeItem(row)
        self.list_widget.insertItem(new_row, item)
        self.list_widget.setCurrentRow(new_row)

        self._on_entries_changed()


    def _on_add_scan_range_clicked(self):
        if self._base_path is None:
            return

        default_pattern = ""
        seed_label = "(none -- no image currently displayed in the main window)"
        if self._browser.file_load_path is not None:
            seed_path = Path(self._browser.file_load_path)
            default_pattern = seed_path.parent.name
            seed_label = self._relative_label(seed_path)

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Add Scan Range")
        dlg.setMinimumWidth(560)
        layout = QtWidgets.QVBoxLayout(dlg)

        layout.addWidget(QtWidgets.QLabel("First scan (e.g. S0042):"))
        first_edit = QtWidgets.QLineEdit()
        layout.addWidget(first_edit)

        layout.addWidget(QtWidgets.QLabel("Last scan (e.g. S0050):"))
        last_edit = QtWidgets.QLineEdit()
        layout.addWidget(last_edit)

        seed_file_label = QtWidgets.QLabel(f"Based on currently displayed image:\n{seed_label}")
        seed_file_label.setWordWrap(True)
        seed_file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(seed_file_label)

        layout.addWidget(QtWidgets.QLabel("Parameter folder name to match in every scan:"))
        pattern_edit = QtWidgets.QLineEdit(default_pattern)
        layout.addWidget(pattern_edit)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dlg.accept)
        button_box.rejected.connect(dlg.reject)
        layout.addWidget(button_box)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        def parse_scan(text):
            t = text.strip()
            if len(t) == 5 and t.startswith("S") and t[1:].isdigit():
                return int(t[1:])
            return None

        first_num = parse_scan(first_edit.text())
        last_num = parse_scan(last_edit.text())
        if first_num is None or last_num is None:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Format", "Expected scan format: S#### (e.g., S0042)"
            )
            return

        param_pattern = pattern_edit.text().strip()
        if not param_pattern:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Pattern", "Enter the parameter folder name to match."
            )
            return

        scan_nums = list(range(min(first_num, last_num), max(first_num, last_num) + 1))
        found = []
        missing = []
        for i, num in enumerate(scan_nums, start=1):
            scan_name = f"S{num:04d}"
            print(f"[Image Stack] ({i}/{len(scan_nums)}) {scan_name}: looking in '{param_pattern}'...")
            scan_path = self._base_path / scan_name
            candidate = self._resolve_scan_image(scan_path, param_pattern)
            if candidate is not None and candidate.exists():
                found.append(candidate)
                print(f"[Image Stack]   found: {candidate}")
            else:
                missing.append(scan_name)
                print(f"[Image Stack]   FAILED: no image found for {scan_name} in parameter folder '{param_pattern}'")

        if found:
            self._add_paths(found)

        if missing:
            QtWidgets.QMessageBox.warning(
                self, "Scan Image Not Found",
                f"No matching image found in parameter folder '{param_pattern}' for:\n"
                + "\n".join(missing)
            )


    def _resolve_scan_image(self, scan_path: Path, param_pattern: str):
        """The display file for self._extension within scan_path's param_pattern folder."""
        if not scan_path.exists():
            return None
        param_folder = scan_path / param_pattern
        if not param_folder.is_dir():
            return None
        latest_recon, _ = self._browser.get_latest_recon_file(param_folder)
        if latest_recon is None:
            return None
        return self._browser.resolve_display_path(latest_recon, self._extension)


    # ------------------------------------------------------------------
    # preview
    # ------------------------------------------------------------------

    def _on_entries_changed(self):
        """
        Rebuild everything that depends on the entry list: the cached
        (full-size, resampled) images, the common target shape, and the
        preview. Any alignment is dropped -- the shifts are stored per row,
        so they stop meaning anything once the rows move.
        """
        self._rebuild_cache()
        self._clear_alignment()
        self._recompute_target_shape()
        self._refresh_preview()


    def _rebuild_cache(self):
        """
        Refresh every entry's "cached" image: the full-size array, resampled
        onto the reference pixel size when "Resample" is on. No cropping --
        cropping, shifting and unwrapping all happen in _display_frame().
        """
        resample_on = self.checkBox_resample.isChecked()
        reference_pixel_size = self._entries[0]["pixel_size_m"] if self._entries else None

        for entry in self._entries:
            if resample_on and reference_pixel_size > 0:
                entry["cached"] = _resample_to_pixel_size(
                    entry["data"], entry["pixel_size_m"], reference_pixel_size
                )
            else:
                entry["cached"] = entry["data"]


    def _reference_pixel_size(self, index=0):
        """
        The pixel size to label an image with: entry 0's whenever "Resample"
        is on (everything has been put on that grid), otherwise the entry's own.
        """
        if not self._entries:
            return 1.0
        if self.checkBox_resample.isChecked():
            return self._entries[0]["pixel_size_m"]
        if not 0 <= index < len(self._entries):
            index = 0
        return self._entries[index]["pixel_size_m"]


    def _recompute_target_shape(self):
        """Elementwise-minimum shape across the cached (already resampled) images."""
        if not self._entries:
            self._target_shape = None
            return

        heights, widths = zip(*(entry["cached"].shape for entry in self._entries))
        self._target_shape = (min(heights), min(widths))


    def _on_resample_toggled(self, _checked):
        self._on_entries_changed()


    def _on_crop_changed(self, _value):
        self._refresh_preview()


    def _on_row_changed(self, _row):
        self._refresh_preview()


    def _crop_origin(self, index):
        """
        (top, left) offset from the cached image's own pixel grid to the
        displayed one, i.e. the origin _center_crop + _extra_center_crop land on.
        """
        h, w = self._entries[index]["cached"].shape
        th, tw = self._target_shape
        top = max((h - th) // 2, 0) + min(self.spinBox_crop_x.value(), th // 2)
        left = max((w - tw) // 2, 0) + min(self.spinBox_crop_y.value(), tw // 2)
        return top, left


    def _displayed_shape(self):
        """Shape every displayed/saved frame has, i.e. target shape minus the extra crop."""
        if self._target_shape is None:
            return None
        th, tw = self._target_shape
        return (th - 2 * min(self.spinBox_crop_x.value(), th // 2),
                tw - 2 * min(self.spinBox_crop_y.value(), tw // 2))


    def _source_frame(self, index):
        """
        The cached image for ``index``, with the unwrapped region spliced back
        in if "Unwrap phase" is on. The cached array itself stays untouched.
        """
        data = self._entries[index]["cached"]
        if self._unwrap_patches is None:
            return data
        out = np.array(data, dtype=np.float32)
        x0, x1, y0, y1 = self._unwrap_slices[index]
        out[x0:x1, y0:y1] = self._unwrap_patches[index]
        return out


    def _display_frame(self, index, apply_shift=True):
        """
        One frame exactly as it is shown and saved: cached image, unwrap
        spliced in, shifted by its stored relative shift, then cropped.

        The shift is applied before cropping so the crop pulls in real
        neighbouring data instead of the replicated border.
        """
        data = self._source_frame(index)

        if apply_shift and self._shifts is not None:
            data = _apply_shift(data, self._shifts[index])

        if self._target_shape is not None:
            data = _center_crop(data, self._target_shape)
        return _extra_center_crop(
            data, self.spinBox_crop_x.value(), self.spinBox_crop_y.value()
        )


    def _refresh_preview(self):
        row = self.list_widget.currentRow()
        if row < 0 or row >= len(self._entries):
            self.plot.clear_image()
            return

        self.plot.set_title(self._relative_label(self._entries[row]["path"]))
        self.plot.set_image(
            self._display_frame(row),
            pixel_size_m=self._reference_pixel_size(row),
            autoRange=False,
        )
        self._current_index_line.setPos(row)


    # ------------------------------------------------------------------
    # alignment
    # ------------------------------------------------------------------

    def _on_align_images_clicked(self):
        if len(self._entries) < 2:
            QtWidgets.QMessageBox.warning(
                self, "Nothing to align", "Add at least two images first."
            )
            return

        self.btn_align_images.setEnabled(False)
        self.align_bar.show()
        self._place_align_roi()


    def _exit_align_mode(self):
        """Leave alignment mode. Any shifts already computed stay applied."""
        self._remove_align_roi()
        self.align_bar.hide()
        self.btn_align_images.setEnabled(True)


    def _place_align_roi(self):
        """(Re)create the draggable/resizable red box marking the region to correlate."""
        self._remove_align_roi()

        shape = self._displayed_shape()
        if shape is None:
            return
        nx, ny = shape

        w = max(nx * ALIGN_ROI_FRAC, 4.0)
        h = max(ny * ALIGN_ROI_FRAC, 4.0)
        roi = pg.RectROI([(nx - w) / 2, (ny - h) / 2], [w, h],
                         pen=pg.mkPen("r", width=2), sideScalers=True)
        roi.setZValue(10)
        self.plot.view.addItem(roi)
        self._align_roi = roi


    def _remove_align_roi(self):
        if self._align_roi is not None:
            self.plot.view.removeItem(self._align_roi)
            self._align_roi = None


    def _roi_slices(self):
        """
        The red box as (x0, x1, y0, y1) integer bounds in displayed-image
        coordinates, clipped to the image. None if it has no usable overlap.
        """
        if self._align_roi is None:
            return None
        shape = self._displayed_shape()
        if shape is None:
            return None
        nx, ny = shape

        pos, size = self._align_roi.pos(), self._align_roi.size()
        x0 = int(np.clip(round(pos.x()), 0, nx))
        y0 = int(np.clip(round(pos.y()), 0, ny))
        x1 = int(np.clip(round(pos.x() + size.x()), 0, nx))
        y1 = int(np.clip(round(pos.y() + size.y()), 0, ny))

        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        return x0, x1, y0, y1


    def _roi_patch(self, index, roi_slices):
        """The red box's contents for one entry, in that entry's cached pixel grid."""
        x0, x1, y0, y1 = roi_slices
        top, left = self._crop_origin(index)
        patch = self._source_frame(index)[top + x0:top + x1, left + y0:left + y1]
        return np.nan_to_num(np.asarray(patch, dtype=np.float64))


    def _on_align_clicked(self):
        roi_slices = self._roi_slices()
        if roi_slices is None:
            QtWidgets.QMessageBox.warning(
                self, "Align", "The red box does not cover a usable region of the image."
            )
            return

        try:
            import skimage.registration  # noqa: F401  (availability check)
        except ImportError:
            QtWidgets.QMessageBox.warning(
                self, "Align",
                "Alignment needs scikit-image, which is not installed in this environment."
            )
            return

        mode = self.comboBox_align_mode.currentText()
        x0, x1, y0, y1 = roi_slices
        print(f"[Image Stack] {mode}: phase correlation over "
              f"x {x0}-{x1}, y {y0}-{y1} of {len(self._entries)} images")

        shifts = None
        error = None
        # The correlation loop pumps the event queue to stay repainting, so the
        # window is locked while it runs -- otherwise a click on Remove or a
        # second Align would mutate the list out from under it.
        self.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Deliberately correlates the *unshifted* images, so re-aligning
            # with another mode replaces the old shifts instead of compounding.
            patches = []
            for index in range(len(self._entries)):
                patches.append(self._roi_patch(index, roi_slices))
                _print_progress("reading", index + 1, len(self._entries))
            shifts = self._compute_shifts(mode, patches)
        except Exception as exc:
            error = exc
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.setEnabled(True)

        if error is not None:
            QtWidgets.QMessageBox.critical(self, "Align", f"Alignment failed:\n{error}")
            return

        self._shifts = shifts
        self.btn_undo_align.setEnabled(True)

        print("[Image Stack] relative shifts (dx, dy) in pixels:")
        for index, (dx, dy) in enumerate(shifts):
            print(f"[Image Stack]   {index:3d}  {dx:+8.2f}  {dy:+8.2f}   "
                  f"{self._relative_label(self._entries[index]['path'])}")

        self._refresh_list_labels()
        self._refresh_shift_plot()
        self._refresh_preview()


    def _compute_shifts(self, mode, patches):
        """
        (N, 2) array of (dx, dy) per entry for the chosen pairing mode.

        "Align adjacent" correlates each image against its predecessor and
        accumulates; the others correlate every image against one common
        reference -- the first, middle or last image, or the (unaligned) mean
        of all of them, in which case the result is recentred so the stack as
        a whole does not drift.
        """
        n = len(patches)
        shifts = np.zeros((n, 2), dtype=float)

        if mode == "Align adjacent":
            for i in range(1, n):
                shifts[i] = shifts[i - 1] + _phase_correlation_shift(patches[i - 1], patches[i])
                _print_progress("correlating", i, n - 1)
                QtWidgets.QApplication.processEvents()
            return shifts

        if mode == "Align to mean":
            reference = np.mean(patches, axis=0)
            for i in range(n):
                shifts[i] = _phase_correlation_shift(reference, patches[i])
                _print_progress("correlating", i + 1, n)
                QtWidgets.QApplication.processEvents()
            return shifts - shifts.mean(axis=0)

        reference_index = {
            "Align to first": 0,
            "Align to middle": n // 2,
            "Align to end": n - 1,
        }[mode]
        reference = patches[reference_index]
        done = 0
        for i in range(n):
            if i != reference_index:
                shifts[i] = _phase_correlation_shift(reference, patches[i])
            done += 1
            _print_progress("correlating", done, n)
            QtWidgets.QApplication.processEvents()
        return shifts


    def _on_undo_alignment_clicked(self):
        self._shifts = None
        self.btn_undo_align.setEnabled(False)
        self._refresh_list_labels()
        self._refresh_shift_plot()
        self._refresh_preview()


    def _clear_alignment(self):
        """Drop the shifts and the unwrap -- both are indexed by row."""
        self._shifts = None
        self._unwrap_patches = None
        self._unwrap_slices = None
        self.btn_undo_align.setEnabled(False)
        if self.btn_unwrap_phase.isChecked():
            self.btn_unwrap_phase.blockSignals(True)
            self.btn_unwrap_phase.setChecked(False)
            self.btn_unwrap_phase.blockSignals(False)
        self._refresh_shift_plot()


    def _refresh_list_labels(self):
        """Show each entry's shift next to its name, so the numbers are visible in situ."""
        for index, entry in enumerate(self._entries):
            label = self._relative_label(entry["path"])
            if self._shifts is not None:
                dx, dy = self._shifts[index]
                label += f"   [{dx:+.1f}, {dy:+.1f} px]"
            item = self.list_widget.item(index)
            if item is not None:
                item.setText(label)


    # ------------------------------------------------------------------
    # unwrap phase (inside the red box only)
    # ------------------------------------------------------------------

    def _on_unwrap_phase_toggled(self, checked):
        if not checked:
            self._unwrap_patches = None
            self._unwrap_slices = None
            self._refresh_preview()
            return

        roi_slices = self._roi_slices()
        if roi_slices is None:
            QtWidgets.QMessageBox.warning(
                self, "Unwrap Phase", "The red box does not cover a usable region of the image."
            )
            self._set_unwrap_checked(False)
            return

        try:
            from skimage.restoration import unwrap_phase
        except ImportError:
            QtWidgets.QMessageBox.warning(
                self, "Unwrap Phase",
                "Unwrap Phase needs scikit-image, which is not installed in this environment."
            )
            self._set_unwrap_checked(False)
            return

        x0, x1, y0, y1 = roi_slices
        print(f"[Image Stack] unwrapping x {x0}-{x1}, y {y0}-{y1} of {len(self._entries)} images")

        patches, slices = [], []
        error = None
        self.setEnabled(False)   # same reason as in _on_align_clicked()
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for index, entry in enumerate(self._entries):
                top, left = self._crop_origin(index)
                region = entry["cached"][top + x0:top + x1, left + y0:left + y1]
                # Fold to [-pi, pi) first: the stored phase may already have
                # been unwrapped once, and unwrap_phase expects wrapped input.
                wrapped = np.angle(np.exp(1j * np.nan_to_num(region).astype(np.float64)))
                patches.append(unwrap_phase(wrapped).astype(np.float32))
                slices.append((top + x0, top + x1, left + y0, left + y1))
                _print_progress("unwrapping", index + 1, len(self._entries))
                QtWidgets.QApplication.processEvents()
        except Exception as exc:
            error = exc
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.setEnabled(True)

        if error is not None:
            QtWidgets.QMessageBox.critical(self, "Unwrap Phase", f"Unwrapping failed:\n{error}")
            self._set_unwrap_checked(False)
            return

        self._unwrap_patches = patches
        self._unwrap_slices = slices
        self._refresh_preview()


    def _set_unwrap_checked(self, checked):
        """Sync the button without re-entering _on_unwrap_phase_toggled."""
        self.btn_unwrap_phase.blockSignals(True)
        self.btn_unwrap_phase.setChecked(checked)
        self.btn_unwrap_phase.blockSignals(False)


    # ------------------------------------------------------------------
    # shift plot
    # ------------------------------------------------------------------

    def _on_show_shifts_toggled(self, checked):
        self.shift_plot.setVisible(checked)
        if checked:
            self._refresh_shift_plot()
            if self.plot_splitter.sizes()[1] == 0:
                total = sum(self.plot_splitter.sizes())
                self.plot_splitter.setSizes([int(total * 0.72), int(total * 0.28)])


    def _refresh_shift_plot(self):
        """Redraw the X/Y shift curves. Cheap, so it runs whether or not it is visible."""
        for item in list(self.shift_plot.plotItem.curves):
            self.shift_plot.removeItem(item)
        legend = self.shift_plot.plotItem.legend
        if legend is not None:
            legend.clear()

        if self._shifts is None or len(self._shifts) == 0:
            return

        indices = np.arange(len(self._shifts))
        self.shift_plot.plot(
            indices, self._shifts[:, 0], name="X shift",
            pen=pg.mkPen("r", width=2), symbol="o", symbolSize=5,
            symbolBrush="r", symbolPen="r",
        )
        self.shift_plot.plot(
            indices, self._shifts[:, 1], name="Y shift",
            pen=pg.mkPen("b", width=2), symbol="o", symbolSize=5,
            symbolBrush="b", symbolPen="b",
        )


    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    def _build_stack(self):
        """
        Build the 3D stack from exactly what the preview shows: the cached
        (already resampled) images, unwrapped and shifted as configured, then
        center-cropped to self._target_shape. Returns (stack, pixel_size_m).

        Each frame is transposed before stacking: ImageJ (and this stack's
        own tiff/h5 files, viewed elsewhere) displays arrays transposed
        relative to this app's internal (row, col) convention, so writing
        the transpose here is what makes the saved stack look right when
        opened outside this app.
        """
        frames = []
        for index in range(len(self._entries)):
            frames.append(self._display_frame(index).T)
            _print_progress("building", index + 1, len(self._entries))

        return np.stack(frames, axis=0), self._reference_pixel_size()


    def _on_save_as_clicked(self):
        if not self._entries:
            QtWidgets.QMessageBox.warning(self, "Nothing to save", "The list is empty.")
            return

        fmt_dlg = QtWidgets.QDialog(self)
        fmt_dlg.setWindowTitle("Save Image Stack")
        layout = QtWidgets.QVBoxLayout(fmt_dlg)
        layout.addWidget(QtWidgets.QLabel("Save format:"))

        btn_row = QtWidgets.QHBoxLayout()
        chosen = {"format": None}

        def pick(fmt):
            chosen["format"] = fmt
            fmt_dlg.accept()

        h5_btn = QtWidgets.QPushButton("h5")
        h5_btn.clicked.connect(lambda: pick("h5"))
        btn_row.addWidget(h5_btn)

        tiff_btn = QtWidgets.QPushButton("tiff")
        tiff_btn.clicked.connect(lambda: pick("tiff"))
        btn_row.addWidget(tiff_btn)

        cancel_btn = QtWidgets.QPushButton("cancel")
        cancel_btn.clicked.connect(fmt_dlg.reject)
        btn_row.addWidget(cancel_btn)

        layout.addLayout(btn_row)

        if fmt_dlg.exec_() != QtWidgets.QDialog.Accepted or chosen["format"] is None:
            return

        start_dir = str(self._base_path) if self._base_path is not None else str(Path.home())
        if chosen["format"] == "h5":
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Image Stack", start_dir, "HDF5 (*.h5)"
            )
            if not save_path:
                return
            if not save_path.lower().endswith((".h5", ".hdf5")):
                save_path += ".h5"
            self._save_h5(Path(save_path))
        else:
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Image Stack", start_dir, "TIFF (*.tiff)"
            )
            if not save_path:
                return
            if not save_path.lower().endswith((".tiff", ".tif")):
                save_path += ".tiff"
            self._save_tiff(Path(save_path))


    def _save_h5(self, save_path: Path):
        stack, pixel_size_m = self._build_stack()
        paths = [self._relative_label(entry["path"]) for entry in self._entries]

        with h5py.File(save_path, "w") as f:
            dset = f.create_dataset("image_stack", data=stack)
            dset.attrs["pixel_size_m"] = pixel_size_m
            f.create_dataset("image_paths", data=np.array(paths, dtype=h5py.string_dtype()))

        QtWidgets.QMessageBox.information(self, "Saved", f"Saved image stack to:\n{save_path}")


    def _save_tiff(self, save_path: Path):
        stack, pixel_size_m = self._build_stack()

        stack = stack.astype(np.float64)
        data_min = stack.min()
        data_max = stack.max()
        if data_max > data_min:
            scaled = (stack - data_min) / (data_max - data_min) * np.iinfo(np.uint16).max
        else:
            scaled = np.zeros_like(stack)
        stack_uint16 = scaled.astype(np.uint16)

        tifffile.imwrite(
            save_path, stack_uint16, imagej=True,
            metadata={"pixel_size": pixel_size_m * 1e6},
        )

        QtWidgets.QMessageBox.information(self, "Saved", f"Saved image stack to:\n{save_path}")
