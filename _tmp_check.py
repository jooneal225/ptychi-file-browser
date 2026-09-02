"""Throwaway offscreen check: bad checkbox column + scan_bad_list.csv."""
import os, sys, shutil, tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).parent))

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QSettings
import pandas as pd
import ptychi_file_browser as M

FAILS = []
def check(label, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + label + (f"   {extra}" if extra else ""))
    if not cond:
        FAILS.append(label)

root = Path(tempfile.mkdtemp(prefix="ptychi_chk_"))
run = root / "2025_Dec"
base = run / "ptychi_recons"

# S0041 recon+unknown, S0042 no recon, S0043 completed=no, S0044 recon+good
for s in ("S0041", "S0042", "S0043", "S0044"):
    (base / s).mkdir(parents=True)
for s in ("S0041", "S0044"):
    p = base / s / "Ndp256_a"; p.mkdir(); (p / "recon_Niter100.h5").write_bytes(b"")
(base / "S0042" / "Ndp256_a").mkdir()
(base / "S0043" / "Ndp256_a").mkdir()
(base / "S0043" / "Ndp256_a" / "recon_Niter100.h5").write_bytes(b"")
(base / "S0044" / "scan_is_good.txt").write_text("good\n")

CSV = """scan,completed,ExpTime,n_pos,phi,date,time,scan_type,sample_name,scan_step_size
S0041,yes,0.05,900,12.5,2025-12-01,10:00:00,fly,sampleA,X 0.000500 Z 0.000500
S0042,yes,0.10,400,13.0,2025-12-01,10:30:00,step,sampleB,X 0.000250 time 0.030000
S0043,no,0.20,100,14.0,2025-12-01,11:00:00,fly,sampleC,phi 0.250000 X 0.001000
S0044,yes,0.30,200,15.0,2025-12-01,11:30:00,fly,sampleD,X 0.000500
"""
log = run / "runlog.csv"
log.write_text(CSV)
badlist = run / "scan_bad_list.csv"

app = QtWidgets.QApplication([])
QSettings("temp", "PtychiFileBrowser").clear()
QSettings("temp", "PtychiFileBrowser").setValue("last_base_path", str(base))
ui = Path(__file__).parent / "ptychi_file_browser.ui"

w = M.PtychiReconBrowser(ui)
w.populate_tree_with_scans()

# ---------------------------------------------- 1. bad list created on populate
check("scan_bad_list.csv created on populate", badlist.exists())
bl = pd.read_csv(badlist)
check("bad list columns", list(bl.columns) == ["scan", "is bad"], list(bl.columns))
check("bad list has all log csv scans",
      list(bl["scan"]) == ["S0041", "S0042", "S0043", "S0044"], list(bl["scan"]))
check("bad list seeded from coloring (only completed=no is yes)",
      list(bl["is bad"]) == ["no", "no", "yes", "no"], list(bl["is bad"]))

# --------------------------------------------------------- 2. the bad column
w.show_runtable_window()
t = w.tableWidget_runtable
headers = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
check("'bad' is the far right column", headers[-1] == "bad", headers)
rows = {t.item(r, 0).text(): r for r in range(t.rowCount())}
BAD = t.columnCount() - 1

def box(scan): return t.item(rows[scan], BAD)
def bg(scan):  return t.item(rows[scan], 0).background().color().getRgb()[:3]
G, PALE = M.GOODNESS_COLORS, M.RECON_EXISTS_COLOR.getRgb()[:3]
RED, YEL = G["bad"].getRgb()[:3], G["reanalyze"].getRgb()[:3]

check("checkbox has no label", box("S0041").text() == "", repr(box("S0041").text()))
check("checkbox is checkable, not editable",
      bool(box("S0041").flags() & Qt.ItemIsUserCheckable)
      and not (box("S0041").flags() & Qt.ItemIsEditable))
check("red row -> checked (S0043 completed=no)",
      box("S0043").checkState() == Qt.Checked and bg("S0043") == RED)
check("non-red rows unchecked",
      all(box(s).checkState() == Qt.Unchecked for s in ("S0041", "S0042", "S0044")))
check("checkbox cell is colored too",
      box("S0043").background().color().getRgb()[:3] == RED)

# ------------------------------------------------- 3. manual check turns red
box("S0041").setCheckState(Qt.Checked)
check("manual check turns row red", bg("S0041") == RED, bg("S0041"))
check("manual check remembered", w._log_bad_by_scan["S0041"] is True)

# ------------------------------------------------------- 4. manual uncheck
box("S0041").setCheckState(Qt.Unchecked)
check("uncheck reverts to computed color (pale green)", bg("S0041") == PALE, bg("S0041"))
check("uncheck remembered", w._log_bad_by_scan["S0041"] is False)

box("S0043").setCheckState(Qt.Unchecked)
check("uncheck cannot clear completed=no red", bg("S0043") == RED, bg("S0043"))
check("checkbox snaps back to checked", box("S0043").checkState() == Qt.Checked)

# ----------------------------------------- 5. toggle still works after sorting
box("S0042").setCheckState(Qt.Checked)
t.sortItems(0, Qt.DescendingOrder)
rows = {t.item(r, 0).text(): r for r in range(t.rowCount())}
check("after sort, S0042 still red", bg("S0042") == RED, bg("S0042"))
check("after sort, checkbox rides with its row", box("S0042").checkState() == Qt.Checked)
box("S0042").setCheckState(Qt.Unchecked)
check("after sort, uncheck hits the right row", bg("S0042") == YEL, bg("S0042"))
check("after sort, other rows untouched", bg("S0044") == G["good"].getRgb()[:3])
t.sortItems(0, Qt.AscendingOrder)
rows = {t.item(r, 0).text(): r for r in range(t.rowCount())}

# --------------------------------------------------- 6. saved on refresh
box("S0041").setCheckState(Qt.Checked)
w.on_runtable_refresh()
bl = pd.read_csv(badlist)
check("refresh writes the bad list",
      dict(zip(bl["scan"], bl["is bad"]))
      == {"S0041": "yes", "S0042": "no", "S0043": "yes", "S0044": "no"},
      dict(zip(bl["scan"], bl["is bad"])))
rows = {t.item(r, 0).text(): r for r in range(t.rowCount())}
check("after refresh the mark survives in the table",
      box("S0041").checkState() == Qt.Checked and bg("S0041") == RED)

# ------------------------------------------------------ 7. saved on close
box("S0044").setCheckState(Qt.Checked)
w.runtable_window.close()
bl = pd.read_csv(badlist)
check("closing the viewer writes the bad list",
      dict(zip(bl["scan"], bl["is bad"]))["S0044"] == "yes",
      dict(zip(bl["scan"], bl["is bad"])))

# ------------------------------------- 8. reloaded on a fresh app instance
w2 = M.PtychiReconBrowser(ui)
w2.populate_tree_with_scans()
check("stored 'yes' reloaded", w2._log_bad_by_scan["S0041"] is True)
w2.show_runtable_window()
t = w2.tableWidget_runtable
rows = {t.item(r, 0).text(): r for r in range(t.rowCount())}
BAD = t.columnCount() - 1
check("stored 'yes' forces red on reload", bg("S0041") == RED, bg("S0041"))
check("stored 'yes' shows as checked", box("S0041").checkState() == Qt.Checked)
check("stored 'no' leaves color alone", bg("S0042") == YEL, bg("S0042"))

# untick and confirm it clears for good
box("S0041").setCheckState(Qt.Unchecked)
w2.runtable_window.close()
w3 = M.PtychiReconBrowser(ui)
w3.populate_tree_with_scans()
check("unticking clears the mark permanently", w3._log_bad_by_scan["S0041"] is False)

# ------------------------------------ 9. main window close writes the list
w3._log_bad_by_scan["S0042"] = True
w3.close()
bl = pd.read_csv(badlist)
check("main window close writes the bad list",
      dict(zip(bl["scan"], bl["is bad"]))["S0042"] == "yes",
      dict(zip(bl["scan"], bl["is bad"])))

# ---------------------- 10. bad list is not mistaken for a log csv candidate
QSettings("temp", "PtychiFileBrowser").remove("log_csv_path")
QSettings("temp", "PtychiFileBrowser").remove("log_csv_base_path")
asked = {"n": 0, "cands": None}
M.PtychiReconBrowser._ask_which_log_csv = lambda self, c: (
    asked.__setitem__("n", asked["n"] + 1), asked.__setitem__("cands", c), c[0])[-1]
w4 = M.PtychiReconBrowser(ui)
w4.populate_tree_with_scans()
check("scan_bad_list.csv excluded from discovery, so no prompt",
      asked["n"] == 0 and w4.log_csv_path == log,
      f"asked={asked['n']} path={w4.log_csv_path}")

# --------------------------------------------------------------- 11. tips
shown = {}
real_exec = QtWidgets.QDialog.exec
QtWidgets.QDialog.exec = lambda self: shown.setdefault(
    "txt", self.findChild(QtWidgets.QTextEdit).toPlainText()) and 0
w4.show_secret_features()
QtWidgets.QDialog.exec = real_exec
txt = shown.get("txt", "")
check("tips describe the checkbox", 'Far right "bad" checkbox' in txt)
check("tips name the file", "'scan_bad_list.csv'" in txt, [l for l in txt.splitlines() if "scan_bad" in l])

print("\n" + "=" * 60)
print(f"{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
shutil.rmtree(root, ignore_errors=True)
QSettings("temp", "PtychiFileBrowser").clear()
sys.exit(1 if FAILS else 0)
