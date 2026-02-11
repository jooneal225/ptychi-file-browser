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


























# from PyQt5.QtCore import QThread, pyqtSignal
# import time
# from pathlib import Path
# import os

# from watchdog.observers import Observer
# from watchdog.observers.polling import PollingObserver
# from watchdog.events import FileSystemEventHandler


# class ScanWatcherThread(QThread):
#     scan_found = pyqtSignal(Path)
#     param_folder_found = pyqtSignal(Path)
#     recon_file_found = pyqtSignal(Path, Path)

#     def __init__(
#         self,
#         base_path: Path,
#         seen_scans=None,
#         seen_param_folders=None,
#         seen_recon_files=None,
#     ):
#         super().__init__()

#         self.base_path = base_path
#         self._running = True

#         # Prepopulated state
#         self._seen_scans = seen_scans or set()
#         self._seen_param_folders = seen_param_folders or {}
#         self._seen_recon_files = seen_recon_files or {}

#         self._observer = None


#     def run(self):
#         event_handler = PtychiEventHandler(self)

#         print('creating polling observer')
#         self._observer = PollingObserver(timeout=2.0)
#         self._observer.schedule(
#             event_handler,
#             str(self.base_path),
#             recursive=True,
#         )
#         self._observer.start()

#         while self._running:
#             print('poll running')
#             time.sleep(0.5)

#         self._observer.stop()
#         self._observer.join()


#     def stop(self):
#         self._running = False



# class PtychiEventHandler(FileSystemEventHandler):
#     def __init__(self, watcher):
#         self.watcher = watcher


#     def on_created(self, event):
#         self._dispatch(event, "created")

#     def on_modified(self, event):
#         self._dispatch(event, "modified")

#     def on_moved(self, event):
#         print("moved:", event.src_path, "->", event.dest_path)
#         self._dispatch(event, "moved", event.dest_path)

#     def _dispatch(self, event, etype, path_override=None):
#         path = Path(path_override or event.src_path)
#         time.sleep(0.1)

#         print(etype, path)

#         if event.is_directory:
#             self._handle_directory(path)
#         else:
#             self._handle_file(path)


    # def _handle_directory(self, path: Path):
    #     try:
    #         depth = len(path.relative_to(self.watcher.base_path).parts)
    #     except ValueError:
    #         # path is not under base_path
    #         return

    #     name = path.name
    #     if depth == 1:
    #         # New scan folder
    #         if (
    #             len(name) == 5
    #             and name.startswith("S")
    #             and name[1:].isdigit()
    #             and name not in self.watcher._seen_scans
    #         ):
    #             self.watcher._seen_scans.add(name)
    #             self.watcher.scan_found.emit(path)
    #             self.watcher._seen_param_folders.setdefault(name, set())
    #             self.watcher._seen_recon_files.setdefault(name, {})
    #             self.watcher.scan_found.emit(path)

    #     elif depth == 2:
    #         scan_name = path.parent.name
    #         # New param folder
    #         if scan_name in self.watcher._seen_scans and name.startswith("Ndp") and path not in self.watcher._seen_param_folders[scan_name]:
    #             self.watcher._seen_param_folders[scan_name].add(path)
    #             self.watcher._seen_recon_files[scan_name].setdefault(path, set())
    #             self.watcher.param_folder_found.emit(path)


    # def _handle_file(self, path: Path):
    #     try:
    #         depth = len(path.relative_to(self.watcher.base_path).parts)
    #     except ValueError:
    #         # path is not under base_path
    #         return
        
    #     if depth > 2:
    #         if (
    #             path.suffix == ".h5"
    #             and path.name.startswith("recon_Niter")
    #         ):
    #             scan_name = path.parents[depth-2].name       # parent of param folder
    #             param_path = path.parents[depth-3]

    #             if (
    #                 scan_name in self.watcher._seen_scans
    #                 and param_path in self.watcher._seen_param_folders[scan_name]
    #                 and path not in self.watcher._seen_recon_files[scan_name][param_path.name]
    #             ):
    #                 self.watcher._seen_recon_files[scan_name][param_path.name].add(path)
    #                 self.watcher.recon_file_found.emit(param_path, path)
