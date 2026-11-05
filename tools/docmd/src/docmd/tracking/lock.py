"""Single-writer lockfiles for tracked state under an output root.

A sync run holds its repo's lock for the whole run (two concurrent syncs of
one repo would corrupt the snapshot and race the archive); the registry gets
its own short-lived lock only around the read-modify-write upsert, so syncs of
*different* repos into one output root stay concurrent. A lock is published
atomically with its full holder payload (hard link of a fully-written temp
file), so a concurrent reader can never observe a half-written lockfile and
misjudge a live holder as stale. A dead pid on the same host is stale and
silently replaced, an unreadable lock is treated the same. Liveness cannot be
probed across hosts, so a foreign hostname counts as live.
"""

from __future__ import annotations

import contextlib
import os
import socket
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from pydantic import BaseModel

from docmd.errors import RepoLockedError
from docmd.logging_config import get_logger

log = get_logger("lock")

REGISTRY_LOCK_NAME = "_registry"


class LockHolder(BaseModel):
    """Lockfile payload. Only pid + hostname decide staleness; the rest is
    informational, so a payload from another docmd version stays honorable."""

    schema_version: int = 1
    pid: int
    hostname: str
    run_id: str = ""
    started_at: datetime | None = None


def lock_path(out_root: Path, name: str) -> Path:
    return out_root / ".docmd" / "locks" / f"{name}.lock"


def _holder_is_stale(path: Path) -> bool:
    try:
        holder = LockHolder.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True  # unreadable or malformed: no live holder can prove itself
    if holder.hostname != socket.gethostname():
        return False
    try:
        os.kill(holder.pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False  # exists, owned by someone else
    return False


class StateLock:
    """Context manager owning one named lockfile under ``<out>/.docmd/locks/``."""

    def __init__(self, out_root: Path, name: str, *, run_id: str, wait_s: float = 0.0) -> None:
        self.path = lock_path(out_root, name)
        self.name = name
        self.run_id = run_id
        self.wait_s = wait_s
        self._held = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            LockHolder(
                pid=os.getpid(),
                hostname=socket.gethostname(),
                run_id=self.run_id,
                started_at=datetime.now(UTC),
            ).model_dump_json(indent=2)
            + "\n"
        )
        deadline = time.monotonic() + self.wait_s
        while True:
            if self._try_publish(payload):
                self._held = True
                return
            if _holder_is_stale(self.path):
                self._claim_stale()
                continue
            if time.monotonic() < deadline:
                time.sleep(0.05)
                continue
            raise RepoLockedError(
                f"{self.name!r} is locked by another docmd run ({self.path}); "
                "wait for it to finish, or delete the lock if that run is gone",
                path=self.path,
            ) from None

    def _try_publish(self, payload: str) -> bool:
        """Create the lockfile with its payload in one atomic step; False if held."""
        fd, tmp_name = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            try:
                os.link(tmp, self.path)
                return True
            except FileExistsError:
                return False
            except OSError:
                # Filesystem without hard links: O_CREAT|O_EXCL creation stays
                # atomic; only the payload write is a (tiny) separate step.
                try:
                    raw = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    return False
                with os.fdopen(raw, "w", encoding="utf-8") as fh:
                    fh.write(payload)
                return True
        finally:
            tmp.unlink(missing_ok=True)

    def _claim_stale(self) -> None:
        """Remove a stale lock without ever unlinking a live one.

        Renaming aside is atomic, so of several waiters exactly one claims the
        file and the losers loop without touching a lock they did not judge
        themselves. The claim is re-verified after the rename, and a lock that
        turned out live (a fresh holder published between judgment and claim)
        is restored.
        """
        claim = self.path.with_name(f".{self.path.name}.claim.{os.getpid()}")
        try:
            self.path.rename(claim)
        except FileNotFoundError:
            return  # another waiter claimed it first
        if _holder_is_stale(claim):
            log.warning("removed stale lock %s", self.path)
        else:
            with contextlib.suppress(FileExistsError):
                os.link(claim, self.path)  # live after all: put it back
                # (suppressed: slot retaken meanwhile; the restored copy is obsolete)
        claim.unlink(missing_ok=True)

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        try:
            holder = LockHolder.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.path.unlink(missing_ok=True)  # missing or unreadable: clear our slot
            return
        if holder.pid == os.getpid() and holder.hostname == socket.gethostname():
            self.path.unlink(missing_ok=True)
        else:
            log.warning(
                "lock %s no longer ours (held by pid %d); leaving it in place",
                self.path,
                holder.pid,
            )

    def __enter__(self) -> StateLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
