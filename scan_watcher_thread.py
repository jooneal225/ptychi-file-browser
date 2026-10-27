from PyQt5.QtCore import QThread, pyqtSignal
from pathlib import Path
import time

import fs_budget


class ScanWatcherThread(QThread):

    scan_found, finished_adding_scans = pyqtSignal(Path), pyqtSignal()

    def __init__(self, base_path: Path, poll_interval=2.0):
        super().__init__()

        self.base_path = base_path
        self.csv_path = base_path / "recon_completed.csv"
        self.poll_interval = poll_interval

        self._running = True
        self._last_mtime = 0
        self._file_pos = 0   # byte offset

    # -----------------------------------------------------

    # NOTE: everything in this class runs on the watcher thread, so every
    # fs_budget call passes pump_events=False. processEvents() may only be
    # called from the GUI thread. The budget itself is shared with the main
    # thread and is lock-protected.

    def run(self):

        # startup - initialize offset to EOF
        try:
            fs_budget.charge_data_read(self.csv_path, pump_events=False)
            with open(self.csv_path, "rb") as f:
                f.seek(0, 2)           # seek to end
                self._file_pos = f.tell()

            self._last_mtime = fs_budget.stat(
                self.csv_path, pump_events=False).st_mtime
        except OSError:
            # No log yet; the poll loop below will pick it up when it appears.
            pass

        while self._running:

            try:
                stat = fs_budget.stat(self.csv_path, pump_events=False)
            except OSError:
                stat = None

            if stat is not None and stat.st_mtime != self._last_mtime:
                self._last_mtime = stat.st_mtime
                self._read_appended()

            time.sleep(self.poll_interval)

    # -----------------------------------------------------

    def stop(self):
        self._running = False

    # -----------------------------------------------------

    def _read_appended(self):
        try:
            fs_budget.charge_data_read(self.csv_path, pump_events=False)
            with open(self.csv_path, "r") as f:
                f.seek(self._file_pos)

                new_text = f.read()      # ONLY new bytes
                self._file_pos = f.tell()
        except OSError:
            return

        # The log gets one line per completed reconstruction, so a scan with
        # several param folders shows up several times in a single batch.
        # Emitting each line would make the main thread re-scan that folder
        # once per line, multiplying the cost for no benefit.
        seen = set()
        for line in new_text.splitlines():
            scan_id = line.strip()
            if not scan_id or scan_id in seen:
                continue
            seen.add(scan_id)

            # No exists() probe here: the main thread already handles a
            # missing folder, and probing would cost a round trip per line.
            self.scan_found.emit(self.base_path / scan_id)

        self.finished_adding_scans.emit()
