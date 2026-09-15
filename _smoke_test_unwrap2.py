"""Headless smoke test for the toggle-based Unwrap Phase behavior."""
import unittest.mock as mock

import numpy as np
from PyQt5 import QtWidgets

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from pg_image_tools._image_view import ImagePlotWidget


def make_wrapped(nx=64, ny=64, amp=12.0):
    x = np.linspace(-2, 2, nx)[:, None]
    y = np.linspace(-2, 2, ny)[None, :]
    phase = amp * np.pi * np.exp(-(x ** 2 + y ** 2))
    return np.angle(np.exp(1j * phase)), phase


def overlay_buttons(w):
    layout = w._unwrap_overlay.layout()
    return layout.itemAt(0).widget(), layout.itemAt(1).widget()   # Apply, Cancel


# ---- 1. Entire image: enable auto-applies, disable reverts ----
w = ImagePlotWidget(enable_lineout=False, enable_compare=False)
wrapped, _true = make_wrapped()
w.set_image(wrapped)
orig = w.pg_view.getImageItem().image.copy()

with mock.patch.object(QtWidgets.QInputDialog, 'getItem', return_value=("Entire image", True)):
    w.action_unwrap_phase.trigger()   # real click: toggles checked AND emits triggered

assert w.action_unwrap_phase.isChecked()
assert w._unwrap_enabled and w._unwrap_kind == 'full'
unwrapped_display = w.pg_view.getImageItem().image
assert not np.allclose(unwrapped_display, orig), "entire-image unwrap should change the display"
print("OK: enabling with 'Entire image' auto-applies immediately")

wrapped2, _ = make_wrapped(amp=20.0)
w.set_image(wrapped2)
disp2 = w.pg_view.getImageItem().image
raw2_folded = np.angle(np.exp(1j * wrapped2.astype(np.float32)))
assert not np.allclose(disp2, raw2_folded), "new image should be auto-unwrapped while enabled (full mode)"
print("OK: new image auto-unwrapped while enabled (entire image mode)")

w.action_unwrap_phase.trigger()   # click again -> disable
assert not w.action_unwrap_phase.isChecked()
assert not w._unwrap_enabled
disp3 = w.pg_view.getImageItem().image
assert np.allclose(disp3, raw2_folded, atol=1e-4), "disabling should revert to the wrapped display"
print("OK: disabling reverts to the non-unwrapped image")


# ---- 2. Rectangular region: enable waits for Apply, drag+Apply affects only new region ----
w2 = ImagePlotWidget(enable_lineout=False, enable_compare=False)
wrapped, _true = make_wrapped()
w2.set_image(wrapped)
raw_folded = np.angle(np.exp(1j * wrapped.astype(np.float32)))

with mock.patch.object(QtWidgets.QInputDialog, 'getItem', return_value=("Rectangular region", True)):
    w2.action_unwrap_phase.trigger()

assert w2.action_unwrap_phase.isChecked(), "action must stay checked once a region is chosen"
assert w2._unwrap_kind == 'rect'
assert w2._unwrap_roi is not None
assert not w2._unwrap_overlay.isHidden()
disp_pending = w2.pg_view.getImageItem().image
assert np.allclose(disp_pending, raw_folded, atol=1e-4), "should show raw data while pending (no Apply yet)"
print("OK: rect region enable shows red ROI + overlay, waits for Apply (no auto-apply)")

btn_apply, btn_cancel = overlay_buttons(w2)

w2._unwrap_roi.setPos([5, 5])
w2._unwrap_roi.setSize([20, 20])
btn_apply.click()

mask = w2._roi_mask(w2._unwrap_roi, 'rect')
disp_applied = w2.pg_view.getImageItem().image
assert np.allclose(disp_applied[~mask], raw_folded[~mask], atol=1e-4), "outside ROI must stay untouched"
assert not np.allclose(disp_applied[mask], raw_folded[mask], atol=1e-6), "inside ROI must be unwrapped"
print("OK: clicking the Apply button unwraps only inside the rect ROI")

w2._unwrap_roi.setPos([40, 40])
w2._unwrap_roi.setSize([15, 15])
btn_apply.click()
mask2 = w2._roi_mask(w2._unwrap_roi, 'rect')
disp_applied2 = w2.pg_view.getImageItem().image
assert np.allclose(disp_applied2[~mask2], raw_folded[~mask2], atol=1e-4), \
    "after moving the ROI, everything outside the NEW region (including the old region) must revert"
assert not np.allclose(disp_applied2[mask2], raw_folded[mask2], atol=1e-6)
print("OK: moving the ROI and re-applying affects only the new region (old region reverts)")

# Cancel button must actually clean up (this was the reported bug)
btn_apply, btn_cancel = overlay_buttons(w2)
btn_cancel.click()
assert not w2.action_unwrap_phase.isChecked(), "Cancel must uncheck the menu item"
assert not w2._unwrap_enabled
assert w2._unwrap_roi is None, "Cancel must remove the red ROI"
assert w2._unwrap_overlay.isHidden(), "Cancel must hide the overlay"
disp_off = w2.pg_view.getImageItem().image
assert np.allclose(disp_off, raw_folded, atol=1e-4)
print("OK: Cancel button unchecks the action, removes the ROI/overlay, and reverts the display")

# And the action must now correctly re-open the dialog rather than misbehave
with mock.patch.object(QtWidgets.QInputDialog, 'getItem', return_value=("Rectangular region", True)) as m:
    w2.action_unwrap_phase.trigger()
assert m.called
assert w2.action_unwrap_phase.isChecked()
print("OK: after Cancel, clicking the menu item again correctly re-opens the region dialog")
w2.action_unwrap_phase.trigger()   # clean up for the next block


# ---- 3. New image while ROI is up: auto-applies using the ROI's current geometry ----
w3 = ImagePlotWidget(enable_lineout=False, enable_compare=False)
wrapped, _ = make_wrapped()
w3.set_image(wrapped)
with mock.patch.object(QtWidgets.QInputDialog, 'getItem', return_value=("Circular region", True)):
    w3.action_unwrap_phase.trigger()

# Never click Apply -- move the ROI, then load a new image straight away.
w3._unwrap_roi.setPos([10, 10])
w3._unwrap_roi.setSize([18, 18])

wrapped_new, _ = make_wrapped(amp=30.0)
w3.set_image(wrapped_new)

assert w3.action_unwrap_phase.isChecked(), "must stay checked across a new image"
assert w3._unwrap_roi is not None, "ROI should persist across a new image"
assert w3._unwrap_mask is not None, "a new image with the ROI present must auto-apply, no Apply click needed"
raw_new_folded = np.angle(np.exp(1j * wrapped_new.astype(np.float32)))
mask3 = w3._roi_mask(w3._unwrap_roi, 'circle')
disp_new = w3.pg_view.getImageItem().image
assert not np.allclose(disp_new[mask3], raw_new_folded[mask3], atol=1e-6), \
    "new image should already be unwrapped inside the ROI, automatically"
assert np.allclose(disp_new[~mask3], raw_new_folded[~mask3], atol=1e-4)
print("OK: displaying a new image while the red ROI is up auto-applies the unwrap immediately")


# ---- 4. Dialog cancel leaves the action unchecked and nothing enabled ----
w4 = ImagePlotWidget(enable_lineout=False, enable_compare=False)
wrapped, _ = make_wrapped()
w4.set_image(wrapped)
with mock.patch.object(QtWidgets.QInputDialog, 'getItem', return_value=("", False)):
    w4.action_unwrap_phase.trigger()
assert not w4.action_unwrap_phase.isChecked()
assert not w4._unwrap_enabled
print("OK: cancelling the region dialog leaves unwrap disabled and unchecked")


# ---- 5. Missing scikit-image handled gracefully ----
import builtins
w5 = ImagePlotWidget(enable_lineout=False, enable_compare=False)
wrapped, _ = make_wrapped()
w5.set_image(wrapped)

real_import = builtins.__import__
def fake_import(name, *a, **kw):
    if name == 'skimage.restoration' or name.startswith('skimage'):
        raise ImportError("no skimage")
    return real_import(name, *a, **kw)

with mock.patch('builtins.__import__', side_effect=fake_import):
    w5.action_unwrap_phase.trigger()
assert not w5.action_unwrap_phase.isChecked()
assert not w5._unwrap_enabled
print("OK: missing scikit-image handled without raising, action stays unchecked")

print("\nALL SMOKE TESTS PASSED")
