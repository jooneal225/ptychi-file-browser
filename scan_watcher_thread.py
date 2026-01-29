from PyQt5.QtCore import QThread, pyqtSignal
import time
from pathlib import Path
import os

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class ScanWatcherThread(QThread):
    scan_found = pyqtSignal(Path)
    param_folder_found = pyqtSignal(Path)
    recon_file_found = pyqtSignal(Path, Path)

    def __init__(
        self,
        base_path: Path,
        seen_scans=None,
        seen_param_folders=None,
        seen_recon_files=None,
    ):
        super().__init__()

        self.base_path = base_path
        self._running = True

        # Prepopulated state
        self._seen_scans = seen_scans or set()
        self._seen_param_folders = seen_param_folders or {}
        self._seen_recon_files = seen_recon_files or {}

        self._observer = None


    def run(self):
        event_handler = PtychiEventHandler(self)

        self._observer = Observer()
        self._observer.schedule(
            event_handler,
            str(self.base_path),
            recursive=True,
        )
        self._observer.start()

        while self._running:
            time.sleep(0.5)

        self._observer.stop()
        self._observer.join()


    def stop(self):
        self._running = False



class PtychiEventHandler(FileSystemEventHandler):
    def __init__(self, watcher):
        self.watcher = watcher

    def on_created(self, event):
        time.sleep(1.)
        if event.is_directory:
            self._handle_directory(Path(event.src_path))
        else:
            self._handle_file(Path(event.src_path))

    def on_moved(self, event):
        # treat moves as creates
        self.on_created(event)


    def _handle_directory(self, path: Path):
        try:
            depth = len(path.relative_to(self.watcher.base_path).parts)
        except ValueError:
            # path is not under base_path
            return

        name = path.name
        if depth == 1:
            # New scan folder
            if (
                len(name) == 5
                and name.startswith("S")
                and name[1:].isdigit()
                and name not in self.watcher._seen_scans
            ):
                self.watcher._seen_scans.add(name)
                self.watcher.scan_found.emit(path)
                self.watcher._seen_param_folders.setdefault(name, set())
                self.watcher._seen_recon_files.setdefault(name, {})
                self.watcher.scan_found.emit(path)

        elif depth == 2:
            scan_name = path.parent.name
            # New param folder
            if scan_name in self.watcher._seen_scans and name.startswith("Ndp") and path not in self.watcher._seen_param_folders[scan_name]:
                self.watcher._seen_param_folders[scan_name].add(path)
                self.watcher._seen_recon_files[scan_name].setdefault(path, set())
                self.watcher.param_folder_found.emit(path)


    def _handle_file(self, path: Path):
        try:
            depth = len(path.relative_to(self.watcher.base_path).parts)
        except ValueError:
            # path is not under base_path
            return
        
        if depth > 2:
            if (
                path.suffix == ".h5"
                and path.name.startswith("recon_Niter")
            ):
                scan_name = path.parents[depth-2].name       # parent of param folder
                param_path = path.parents[depth-3]

                if (
                    scan_name in self.watcher._seen_scans
                    and param_path in self.watcher._seen_param_folders[scan_name]
                    and path not in self.watcher._seen_recon_files[scan_name][param_path.name]
                ):
                    self.watcher._seen_recon_files[scan_name][param_path.name].add(path)
                    self.watcher.recon_file_found.emit(param_path, path)
