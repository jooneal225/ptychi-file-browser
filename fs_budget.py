"""
Central rate limiter for every filesystem interaction in this application.

Why this exists
---------------
The scan directory lives on an NFS mount. Every metadata call (readdir, stat,
lookup) is a network round trip to a shared file server. Unthrottled, a serial
walk of a few thousand scan folders issues hundreds of RPCs per second. That is
a metadata storm, and it degrades the filer for everyone mounting it, not just
this app.

Everything here that touches disk goes through the wrappers in this module.
That is the entire point of the design: this file is the only place a reviewer
has to read in order to know how hard we hit the file server.

Accounting model
----------------
One "token" approximates one NFS RPC.

  * A single-path op (stat / exists / is_dir / opening a file) costs 1 token.
  * A directory read costs one token per ENTRIES_PER_CHUNK entries returned,
    minimum 1, because an NFS readdir of a large directory is served in chunks
    rather than in one round trip.

The directory-read charge is a MODEL, not a measurement. The true chunk size
depends on the server, the mount options and the length of the entry names.
It is deliberately not generous: enumerating a 3000-entry directory should cost
real budget instead of looking as cheap as a single stat.

Tokens accumulate while the app is idle, up to BURST_CAPACITY. That is what
keeps the UI feeling responsive -- a user clicking around spends banked tokens
and does not wait -- while a sustained bulk walk still averages out to
RATE_OPS_PER_SEC.

Known gaps (not rate limited; documented rather than hidden)
------------------------------------------------------------
  * QSettings      - Windows registry, not the filesystem.
  * uic.loadUi     - one-shot read of the .ui file at startup.
  * QFileDialog    - Qt enumerates directories inside C++, not interceptable
                     from Python.
"""

import fnmatch
import math
import os
import threading
import time
from pathlib import Path

from PyQt5 import QtWidgets


# ----------------------------------------------------------------------
# Tunables - these are the numbers to change, and the only ones.
# ----------------------------------------------------------------------

RATE_OPS_PER_SEC = 5.0   # sustained ceiling on filesystem ops
BURST_CAPACITY = 15.0     # tokens that may accumulate while idle
ENTRIES_PER_CHUNK = 100   # assumed NFS readdir chunk size, in entries

# Rough, UNMEASURED guess used only for the warning popup before a full walk.
# Basis: ~3000 scans x (1 + 1-5) directory reads ~= 6,000-18,000 tokens, which
# at RATE_OPS_PER_SEC lands around 10-30 minutes. Measure it and update.
ESTIMATED_FULL_WALK_MINUTES = 25

_SLEEP_SLICE_S = 0.05     # granularity of waiting; also the UI repaint rate


class FsBudgetCancelled(Exception):
    """Raised out of a filesystem wrapper when the user cancelled the work."""


def _pump_qt_events():
    """Let the GUI repaint and stay clickable while we are throttling."""
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.processEvents()


class FsBudget:
    """
    A plain token bucket.

    Tokens refill at RATE_OPS_PER_SEC and cap at BURST_CAPACITY. Taking more
    tokens than are available sleeps until they are available.
    """

    def __init__(self, rate=RATE_OPS_PER_SEC, capacity=BURST_CAPACITY):
        self.rate = float(rate)
        self.capacity = float(capacity)

        self._tokens = float(capacity)   # start full, so startup is not slow
        self._last_refill = time.monotonic()
        self._cancelled = False
        self._lock = threading.Lock()    # the watcher thread shares this budget

        # Observability - printed in the summary after a long walk.
        self.ops_charged = 0
        self.tokens_charged = 0.0
        self.total_wait_s = 0.0

    # -- cancellation -------------------------------------------------

    def cancel(self):
        """Make the next wrapper call raise FsBudgetCancelled."""
        with self._lock:
            self._cancelled = True

    def reset_cancel(self):
        with self._lock:
            self._cancelled = False

    def reset_counters(self):
        with self._lock:
            self.ops_charged = 0
            self.tokens_charged = 0.0
            self.total_wait_s = 0.0

    def summary(self):
        return (f"{self.ops_charged} filesystem ops "
                f"({self.tokens_charged:.0f} tokens), "
                f"{self.total_wait_s:.1f}s spent waiting on the rate limit")

    # -- the bucket ---------------------------------------------------

    def _refill_locked(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

    def take(self, cost=1.0, pump_events=True):
        """
        Block until `cost` tokens are available, then spend them.

        A cost larger than the bucket capacity is spent in capacity-sized
        slices, so a very large directory read cannot deadlock the bucket.
        """
        remaining = float(cost)
        if remaining <= 0:
            return

        while remaining > 0:
            slice_cost = min(remaining, self.capacity)
            self._take_one_slice(slice_cost, pump_events)
            remaining -= slice_cost

    def _take_one_slice(self, cost, pump_events):
        """
        Spend `cost` tokens, where cost <= capacity.

        Never holds the lock across a sleep.
        """
        while True:
            with self._lock:
                if self._cancelled:
                    raise FsBudgetCancelled("Filesystem work cancelled by user")

                self._refill_locked()

                if self._tokens >= cost:
                    self._tokens -= cost
                    self.ops_charged += 1
                    self.tokens_charged += cost
                    return

                # How long until enough tokens exist? Sleep in small slices so
                # the UI stays alive and cancellation stays responsive.
                deficit = cost - self._tokens

            nap = min(_SLEEP_SLICE_S, deficit / self.rate)
            time.sleep(nap)
            self.total_wait_s += nap

            if pump_events:
                _pump_qt_events()


# The single shared budget. Import this, do not build your own.
BUDGET = FsBudget()


# ----------------------------------------------------------------------
# Wrappers - the only approved way to touch the filesystem.
# ----------------------------------------------------------------------

def _charge_listing_tail(n_entries, pump_events):
    """
    Pay the rest of a directory read once we know how many entries came back.

    One token was already charged up front. A readdir of n entries is modelled
    as ceil(n / ENTRIES_PER_CHUNK) round trips, so charge the difference. The
    debt simply delays the next operation.
    """
    extra = max(0, math.ceil(n_entries / ENTRIES_PER_CHUNK) - 1)
    if extra:
        BUDGET.take(extra, pump_events)


def scandir(path, pump_events=True):
    """
    Rate-limited os.scandir. Returns a fully consumed list of DirEntry.

    The returned entries carry their name and, on most platforms, their type,
    so callers may use entry.is_dir() / entry.is_file() WITHOUT charging again
    -- that data came from the readdir we already paid for.
    """
    BUDGET.take(1, pump_events)
    with os.scandir(os.fspath(path)) as it:
        entries = list(it)
    _charge_listing_tail(len(entries), pump_events)
    return entries


def listdir_paths(path, pump_events=True):
    """Rate-limited replacement for Path.iterdir(). Returns a list of Path."""
    return [Path(e.path) for e in scandir(path, pump_events)]


def glob(path, pattern, pump_events=True):
    """
    Rate-limited replacement for Path.glob() for single-level patterns.

    Filtering happens in Python on one directory read; a narrower pattern does
    not reduce server traffic, it only shortens the returned list.
    """
    return [Path(e.path) for e in scandir(path, pump_events)
            if fnmatch.fnmatch(e.name, pattern)]


def stat(path, pump_events=True):
    BUDGET.take(1, pump_events)
    return os.stat(os.fspath(path))


def exists(path, pump_events=True):
    BUDGET.take(1, pump_events)
    return os.path.exists(os.fspath(path))


def is_dir(path, pump_events=True):
    BUDGET.take(1, pump_events)
    return os.path.isdir(os.fspath(path))


def is_file(path, pump_events=True):
    BUDGET.take(1, pump_events)
    return os.path.isfile(os.fspath(path))


def charge_data_read(path=None, pump_events=True):
    """
    Charge for opening a file for reading.

    Call this immediately before h5py.File / tifffile / PIL.Image.open /
    pandas.read_csv / open(). The read itself is not wrapped, because those
    libraries want a real path; this just makes it cost budget.
    """
    BUDGET.take(1, pump_events)


def charge_write(path=None, pump_events=True):
    """Charge for a write, unlink or rmtree. Same contract as charge_data_read."""
    BUDGET.take(1, pump_events)
