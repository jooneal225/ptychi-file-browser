"""Throwaway offscreen check: the runtable plot pane."""
import os, sys, shutil, tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).parent))

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QSettings
import numpy as np
import h5py
import ptychi_file_browser as M

FAILS = []
def check(label, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + label + (f"   {extra}" if extra else ""))
    if not cond:
        FAILS.append(label)

def npts(scatter_item):
    """Point count of a ScatterPlotItem, empty or not."""
    x, _ = scatter_item.getData()
    return 0 if x is None else len(x)

def logview(a):
    """What ImagePlotWidget._render produces with the log checkbox on."""
    mag = np.abs(a.astype(np.float32))
    pos = mag[mag > 0]
    floor = float(pos.min()) if pos.size else np.finfo(np.float32).tiny
    return np.log10(np.clip(mag, floor, None))

root = Path(tempfile.mkdtemp(prefix="ptychi_plotchk_"))
run = root / "2025_Dec"
base = run / "ptychi_recons"

for s in ("S0041", "S0042", "S0043", "S0044"):
    (base / s).mkdir(parents=True)

# S0041 -> preproc/{scan}: 3d /dp + a _para. file
(run / "preproc" / "S0041").mkdir(parents=True)
DP = (np.arange(5 * 64 * 64).reshape(5, 64, 64) % 977).astype(np.uint16)
with h5py.File(run / "preproc" / "S0041" / "foo_dp.h5", "w") as f:
    f["dp"] = DP
PPX = np.linspace(-1, 1, 100)
PPY = np.sin(PPX)
with h5py.File(run / "preproc" / "S0041" / "foo_para.h5", "w") as f:
    f["ppX"] = PPX
    f["ppY"] = PPY
# a decoy that must be ignored (neither _dp. nor _para.)
(run / "preproc" / "S0041" / "notes.txt").write_text("x")

# S0042 -> ptycho/{scan3} = ptycho/042, several files: one pattern per file.
# 2d data reached by the second dataset name only, and a master file to skip.
(run / "ptycho" / "042").mkdir(parents=True)
IMG2D = np.arange(32 * 48).reshape(32, 48).astype(np.float32)
with h5py.File(run / "ptycho" / "042" / "bbb.h5", "w") as f:
    f.create_dataset("/entry/data/data00001", data=IMG2D)
with h5py.File(run / "ptycho" / "042" / "aaa.h5", "w") as f:   # first alphabetically
    f.create_dataset("/entry/data/data00001", data=IMG2D * 2)
with h5py.File(run / "ptycho" / "042" / "ccc.h5", "w") as f:
    f.create_dataset("/entry/data/data00001", data=IMG2D * 3)
# sorts ahead of the data files, so it wins 'first' unless it is skipped
with h5py.File(run / "ptycho" / "042" / "0000_master.h5", "w") as f:
    f.create_dataset("/entry/data/data00001", data=np.zeros((4, 4), np.float32))

# S0044 -> SAXS/S0044, a single stacked file: totals come from the stack
(run / "SAXS" / "S0044").mkdir(parents=True)
STACK = (np.arange(7 * 16 * 16).reshape(7, 16, 16) % 31).astype(np.float32)
with h5py.File(run / "SAXS" / "S0044" / "only.h5", "w") as f:
    f.create_dataset("/entry/data/data", data=STACK)
with h5py.File(run / "SAXS" / "S0044" / "only_master.h5", "w") as f:
    f.create_dataset("/entry/data/data", data=np.zeros((4, 4), np.float32))

# S0043 -> nothing anywhere

CSV = """scan,completed,ExpTime,n_pos,phi,date,time,scan_type,sample_name,scan_step_size
S0041,yes,0.05,900,12.5,2025-12-01,10:00:00,fly,sampleA,X 0.000500 Z 0.000500
S0042,yes,0.10,400,13.0,2025-12-01,10:30:00,step,sampleB,X 0.000250
S0043,yes,0.20,100,14.0,2025-12-01,11:00:00,fly,sampleC,X 0.001000
S0044,yes,0.30,200,15.0,2025-12-01,11:30:00,fly,sampleD,X 0.000500
"""
(run / "runlog.csv").write_text(CSV)

app = QtWidgets.QApplication([])
QSettings("temp", "PtychiFileBrowser").clear()
QSettings("temp", "PtychiFileBrowser").setValue("last_base_path", str(base))
ui = Path(__file__).parent / "ptychi_file_browser.ui"

w = M.PtychiReconBrowser(ui)
w.populate_tree_with_scans()
w.show_runtable_window()
t = w.tableWidget_runtable

# ------------------------------------------------------ 0. the new column
headers = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
check("sample_name sits just after completed",
      headers[:3] == ["scan", "completed", "sample_name"], headers)
check("'bad' is still the far right column", headers[-1] == "bad", headers)
srow = {t.item(r, 0).text(): r for r in range(t.rowCount())}
check("sample_name cells are filled",
      [t.item(srow[s], 2).text() for s in ("S0041", "S0042", "S0043")]
      == ["sampleA", "sampleB", "sampleC"],
      [t.item(srow[s], 2).text() for s in ("S0041", "S0042", "S0043")])

TOTALS = M.RUNTABLE_TOTALS_TEXT.__mod__

# --------------------------------------------------------- 1. the button
buttons = {b.text(): b for b in w.runtable_window.findChildren(QtWidgets.QPushButton)}
check("'Show Plots' button exists", "Show Plots" in buttons, sorted(buttons))
plots_btn = buttons["Show Plots"]
check("button is checkable and starts unchecked",
      plots_btn.isCheckable() and not plots_btn.isChecked())
check("plot pane not built before the first check", w._runtable_plot_panel is None)
check("plot controls exist in the button row, hidden until the pane is shown",
      w._runtable_plot_controls is not None and w._runtable_plot_controls.isHidden())
check("controls are siblings of the buttons, not of the plots",
      w._runtable_plot_controls.parent() is plots_btn.parent())

plots_btn.setChecked(True)
check("plot pane built on check", w._runtable_plot_panel is not None)
check("plot controls appear with the pane", not w._runtable_plot_controls.isHidden())

check("runtable window is 10pt", w.runtable_window.font().pointSize() == 10,
      w.runtable_window.font().pointSize())
check("controls and pane inherit the 10pt font",
      w.spinBox_runtableSlice.font().pointSize() == 10
      and w.checkBox_runtableLog.font().pointSize() == 10
      and w.label_runtableImagePath.font().pointSize() == 10,
      (w.spinBox_runtableSlice.font().pointSize(),
       w.label_runtableImagePath.font().pointSize()))
check("plot pane is the splitter's lower half",
      w._runtable_splitter.count() == 2
      and w._runtable_splitter.widget(0) is t
      and w._runtable_splitter.widget(1) is w._runtable_plot_panel)
check("plot pane visible", not w._runtable_plot_panel.isHidden())
check("plots start blank",
      w._runtable_image_plot.raw_data is None
      and npts(w._runtable_scatter) == 0)
check("totals start as dashes",
      w.label_runtableTotals.text() == TOTALS(("-", "-"))
      and w.label_runtableSliceTotal.text() == "/-",
      (w.label_runtableTotals.text(), w.label_runtableSliceTotal.text()))

rows = {t.item(r, 0).text(): r for r in range(t.rowCount())}
img, scat = w._runtable_image_plot, w._runtable_scatter
spin = w.spinBox_runtableSlice

# ------------------------------------------- 2. preproc, 3d /dp + scatter
t.selectRow(rows["S0041"])
check("S0041 image is one 64x64 slice",
      img.raw_data is not None and img.raw_data.shape == (64, 64),
      None if img.raw_data is None else img.raw_data.shape)
check("S0041 image is slice 0", np.array_equal(img.raw_data, DP[0]))
check("slice spinbox range follows the stack",
      (spin.minimum(), spin.maximum(), spin.value()) == (0, 4, 0),
      (spin.minimum(), spin.maximum(), spin.value()))
check("slice total label matches the stack", w.label_runtableSliceTotal.text() == "/5",
      w.label_runtableSliceTotal.text())
check("a stacked _dp. file reports its depth as the total",
      w.label_runtableTotals.text() == TOTALS((5, 100)), w.label_runtableTotals.text())
check("image path label names the _dp. file",
      "foo_dp.h5" in w.label_runtableImagePath.toolTip(),
      w.label_runtableImagePath.toolTip())
lo, hi = img.image_view.ui.histogram.getLevels()
disp = img.image_view.getImageItem().image
check("levels are synced to the displayed array",
      np.isclose(lo, disp.min(), rtol=1e-5, atol=1e-6)
      and np.isclose(hi, disp.max(), rtol=1e-5, atol=1e-6),
      (lo, hi, float(disp.min()), float(disp.max())))

sx, sy = scat.getData()
check("S0041 scatter is mean-subtracted and in microns",
      len(sx) == 100
      and np.allclose(sx, (PPX - PPX.mean()) * 1e6)
      and np.allclose(sy, (PPY - PPY.mean()) * 1e6),
      len(sx))
check("scatter path label names the _para. file",
      "foo_para.h5" in w.label_runtableScatterPath.toolTip(),
      w.label_runtableScatterPath.toolTip())

# ------------------------------------------------------ 3. the slice picker
spin.setValue(3)
check("slice 3 is displayed", np.array_equal(img.raw_data, DP[3]))
check("source cached so the slice change skips the file search",
      w._runtable_img_source is not None
      and w._runtable_img_source[0] == "S0041"
      and w._runtable_img_source[2] == "/dp")

# ---------- 4. ptycho/{scan3}, one pattern per file, master skipped, 2d data
t.selectRow(rows["S0042"])
check("S0042 found under ptycho/042",
      "042" in w.label_runtableImagePath.toolTip()
      and w.label_runtableImagePath.toolTip().endswith("aaa.h5"),
      w.label_runtableImagePath.toolTip())
check("master.h5 is skipped even though it sorts first",
      img.raw_data is not None and np.array_equal(img.raw_data, IMG2D * 2),
      None if img.raw_data is None else img.raw_data.shape)
check("several files means one pattern per file: only the first is readable",
      (spin.maximum(), spin.value()) == (0, 0)
      and w.label_runtableSliceTotal.text() == "/1",
      (spin.maximum(), spin.value(), w.label_runtableSliceTotal.text()))
check("the file count is the pattern total, master excluded",
      w.label_runtableTotals.text() == TOTALS((3, "-")), w.label_runtableTotals.text())
check("S0042 has no scatter file",
      npts(scat) == 0
      and w.label_runtableScatterPath.text() == M.RUNTABLE_NO_FILE_TEXT,
      w.label_runtableScatterPath.text())

# ------------------- 4b. SAXS, a single stacked file: the stack is the total
t.selectRow(rows["S0044"])
check("S0044 found under SAXS/S0044",
      w.label_runtableImagePath.toolTip().endswith("only.h5"),
      w.label_runtableImagePath.toolTip())
check("a lone stacked file pages through its whole depth",
      (spin.maximum(), spin.value()) == (6, 0)
      and w.label_runtableSliceTotal.text() == "/7"
      and np.array_equal(img.raw_data, STACK[0]),
      (spin.maximum(), w.label_runtableSliceTotal.text()))
check("its depth is also the pattern total",
      w.label_runtableTotals.text() == TOTALS((7, "-")), w.label_runtableTotals.text())
spin.setValue(5)
check("slices of a lone stacked file are reachable",
      np.array_equal(img.raw_data, STACK[5]))

# ------------------------------------------------- 5. nothing on disk at all
t.selectRow(rows["S0043"])
check("S0043 image blanked", img.raw_data is None)
check("S0043 image label says so",
      w.label_runtableImagePath.text() == M.RUNTABLE_NO_FILE_TEXT,
      w.label_runtableImagePath.text())
check("S0043 scatter blanked",
      npts(scat) == 0
      and w.label_runtableScatterPath.text() == M.RUNTABLE_NO_FILE_TEXT)
check("no stale source cached", w._runtable_img_source is None)
check("totals go back to dashes",
      w.label_runtableTotals.text() == TOTALS(("-", "-"))
      and w.label_runtableSliceTotal.text() == "/-",
      (w.label_runtableTotals.text(), w.label_runtableSliceTotal.text()))

# ------------------------------------------ 6. still right after sorting
t.sortItems(0, Qt.DescendingOrder)
rows = {t.item(r, 0).text(): r for r in range(t.rowCount())}
t.selectRow(rows["S0041"])
check("after sort, the right scan is plotted",
      img.raw_data is not None and img.raw_data.shape == (64, 64)
      and "foo_dp.h5" in w.label_runtableImagePath.toolTip(),
      w.label_runtableImagePath.toolTip())
check("after sort, the scatter follows too", npts(scat) == 100)

# --------------------------------------------------- 7. display toggles
check("log and transpose default to on",
      w.checkBox_runtableLog.isChecked() and w.checkBox_runtableTranspose.isChecked())

ref = img.raw_data.astype(np.float32)          # whichever slice is up right now
check("default view is transposed and log",
      np.allclose(img.image_view.getImageItem().image, logview(ref.T), atol=1e-4))
check("log floors at the smallest value present, not at eps",
      logview(ref.T).min() > -1.0, logview(ref.T).min())
w.checkBox_runtableLog.setChecked(False)
check("log off leaves the transposed linear array",
      np.allclose(img.image_view.getImageItem().image, ref.T))
w.checkBox_runtableTranspose.setChecked(False)
check("transpose off leaves the plain array",
      np.allclose(img.image_view.getImageItem().image, ref))
check("raw data untouched by the toggles", np.array_equal(img.raw_data, ref))

lo, hi = img.image_view.ui.histogram.getLevels()
check("levels re-sync after a toggle redraw",
      np.isclose(lo, ref.min(), rtol=1e-5, atol=1e-6)
      and np.isclose(hi, ref.max(), rtol=1e-5, atol=1e-6),
      (lo, hi, float(ref.min()), float(ref.max())))
w.checkBox_runtableLog.setChecked(True)
w.checkBox_runtableTranspose.setChecked(True)

# ------------------------------------------------ 8. refresh does not disturb
before = w.label_runtableImagePath.toolTip()
w.on_runtable_refresh()
check("refresh leaves the plots alone",
      w.label_runtableImagePath.toolTip() == before and img.raw_data is not None)

# --------------------------------------------------------- 9. toggle back off
plots_btn.setChecked(False)
check("unchecking hides the pane", w._runtable_plot_panel.isHidden())
check("unchecking hides the controls too", w._runtable_plot_controls.isHidden())
rows = {t.item(r, 0).text(): r for r in range(t.rowCount())}
t.selectRow(rows["S0043"])
check("hidden pane is not replotted", img.raw_data is not None)
plots_btn.setChecked(True)
check("rechecking replots the current row",
      img.raw_data is None
      and w.label_runtableImagePath.text() == M.RUNTABLE_NO_FILE_TEXT)

check("error banner stays hidden through all of the above",
      w.label_runtableError.isHidden())

# ------------------------------- 10. a file that is there but will not read
# Stands in for compressed data whose HDF5 filter plugin is not installed:
# the file opens, the dataset is present, the bytes will not come out.
PLUGIN_ERR = "Can't read data (can't open directory: /usr/local/hdf5/lib/plugin)"

class _BoomDataset:
    ndim, shape = 3, (5, 64, 64)
    def __getitem__(self, key):
        raise OSError(PLUGIN_ERR)

class _BoomFile:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __contains__(self, name): return name == "/dp"
    def __getitem__(self, name): return _BoomDataset()

real_File = M.h5py.File
M.h5py.File = _BoomFile
try:
    w._runtable_plot_scan = None      # force a fresh search + read
    w._runtable_img_source = None
    t.selectRow(rows["S0041"])
finally:
    M.h5py.File = real_File

check("unreadable dataset shows the banner", not w.label_runtableError.isHidden())
tip = w.label_runtableError.toolTip()
check("banner names the plugin as the likely cause",
      "HDF5 plugin may be required" in tip, tip)
check("banner names the dataset and the file",
      "/dp" in tip and "foo_dp.h5" in tip, tip)
check("banner quotes the underlying error", PLUGIN_ERR in tip, tip)
check("banner is about the image, not the positions",
      tip.startswith("image:") and "positions:" not in tip, tip)
check("the plot itself is emptied", img.raw_data is None)

# a scan with nothing on disk is a miss, not a failure -> no banner
t.selectRow(rows["S0043"])
check("no banner for a scan with no files at all",
      w.label_runtableError.isHidden() and w.label_runtableError.text() == "",
      w.label_runtableError.text())

# and a good read clears it again
w._runtable_plot_scan = None
w._runtable_img_source = None
t.selectRow(rows["S0041"])
check("banner clears once a file reads again",
      w.label_runtableError.isHidden() and img.raw_data is not None)

print("\n" + "=" * 60)
print(f"{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
w.runtable_window.close()
w.close()
shutil.rmtree(root, ignore_errors=True)
QSettings("temp", "PtychiFileBrowser").clear()
sys.exit(1 if FAILS else 0)
