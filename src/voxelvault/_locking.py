"""Cross-platform file locking for VoxelVault index operations.

Provides a context-manager that acquires an exclusive lock on a lockfile
to prevent concurrent processes from rebuilding the spatial index at the
same time.  Uses platform-native APIs:

- Windows: ``msvcrt.locking()``
- POSIX:   ``fcntl.flock()``

The lock is advisory — it only serializes cooperating VoxelVault
processes, not arbitrary external access to the index files.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import TracebackType


class VaultFileLock:
    """Advisory file lock for serializing index operations.

    Usage::

        with VaultFileLock(vault_path / "index" / ".lock"):
            # ... rebuild or update index ...
    """

    def __init__(self, lock_path: Path, timeout: float = 30.0) -> None:
        self._lock_path = lock_path
        self._timeout = timeout
        self._fd: int | None = None

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> VaultFileLock:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR)
        self._acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._release()
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    # -- platform dispatch ---------------------------------------------------

    def _acquire(self) -> None:
        if sys.platform == "win32":
            self._acquire_windows()
        else:
            self._acquire_posix()

    def _release(self) -> None:
        if sys.platform == "win32":
            self._release_windows()
        else:
            self._release_posix()

    # -- Windows (msvcrt) ----------------------------------------------------

    def _acquire_windows(self) -> None:
        import msvcrt

        assert self._fd is not None
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Could not acquire index lock at {self._lock_path} "
                        f"within {self._timeout}s"
                    )
                time.sleep(0.05)

    def _release_windows(self) -> None:
        import msvcrt

        if self._fd is not None:
            try:
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass

    # -- POSIX (fcntl) -------------------------------------------------------

    def _acquire_posix(self) -> None:
        import fcntl

        assert self._fd is not None
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Could not acquire index lock at {self._lock_path} "
                        f"within {self._timeout}s"
                    )
                time.sleep(0.05)

    def _release_posix(self) -> None:
        import fcntl

        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
