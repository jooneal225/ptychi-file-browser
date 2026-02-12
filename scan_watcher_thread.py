from PyQt5.QtCore import QThread, pyqtSignal
from pathlib import Path
import time


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

    def run(self):

        # startup - initialize offset to EOF
        if self.csv_path.exists():
            with open(self.csv_path, "rb") as f:
                f.seek(0, 2)           # seek to end
                self._file_pos = f.tell()

            self._last_mtime = self.csv_path.stat().st_mtime

        while self._running:

            if self.csv_path.exists():
                stat = self.csv_path.stat()

                if stat.st_mtime != self._last_mtime:
                    self._last_mtime = stat.st_mtime
                    self._read_appended()

            time.sleep(self.poll_interval)

    # -----------------------------------------------------

    def stop(self):
        self._running = False

    # -----------------------------------------------------

    def _read_appended(self):
        with open(self.csv_path, "r") as f:
            f.seek(self._file_pos)

            new_text = f.read()      # ONLY new bytes
            self._file_pos = f.tell()

        for line in new_text.splitlines():
            scan_id = line.strip()
            if not scan_id:
                continue

            scan_path = self.base_path / scan_id

            if scan_path.exists():
                self.scan_found.emit(scan_path)
        
        self.finished_adding_scans.emit()
