"""A many-readers/one-writer lock for the engine's in-memory state.

One mutex over everything is correct and, for a small vault, invisible. It
stops being invisible when a write takes real time: every search queues
behind it. Measured on a production vault, a search cost 15.9 ms idle and
63.6 ms while indexing ran — and when a write wedged entirely (a matmul
spinning inside a broken BLAS thread pool), every read behind it wedged too.

Readers don't conflict with each other, so letting them run together removes
the queue between searches.

What this does NOT do, and it matters: an exclusive writer that never
finishes still blocks every reader. Splitting the lock does not shrink the
blast radius of a wedged write — only ``timeout=`` does. Retrieval paths
should pass one and degrade to "no memory" rather than inherit someone
else's hang; that is what turned a single stuck matmul into a 27-hour
outage with every health signal green.

FAIR (first-come, first-served), which is not the obvious choice and is worth
saying why. The first cut was writer-preferring — any waiting writer held new
readers back — on the reasoning that a read-mostly engine would otherwise
postpone indexing forever. Measured against a continuous indexing load, that
inverted the problem completely: the writer re-queued the instant it released,
so searches never got a turn at all and the benchmark never finished. Neither
side may starve here, so arrival order decides. Readers that queued together
still run together; a reader waits only for writers that arrived *before* it.

NOT reentrant. The single ``RLock`` it replaces tolerated a locked method
calling another locked method; here that self-deadlocks, so every such pair
must be split into a public locked entry point and an unlocked internal
(``_get_text_unlocked``, ``_trust_feedback_unlocked``).
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator, List, Optional


class MemoryBusy(RuntimeError):
    """Raised when a bounded acquisition ran out of patience.

    A caller that can proceed without memory (retrieval on a turn) should
    catch this and do so. A caller that cannot should let it propagate — an
    error beats a hang, because a hang has no signal at all.
    """


class RWLock:
    """Many concurrent readers, or one exclusive writer, in arrival order."""

    __slots__ = ("_cond", "_readers", "_writer", "_ticket", "_waiting_readers", "_waiting_writers")

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._ticket = 0
        # Arrival-ordered tickets of everyone still waiting. Comparing a
        # waiter's ticket against these is what makes the order fair.
        self._waiting_readers: List[int] = []
        self._waiting_writers: List[int] = []

    # ── acquisition ──────────────────────────────────────────────────
    @contextmanager
    def read(self, timeout: Optional[float] = None) -> Iterator[None]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            mine = self._ticket
            self._ticket += 1
            self._waiting_readers.append(mine)
            try:
                # Wait out any writer already running, and any writer that got
                # in line first. Writers queued *behind* this reader do not
                # hold it back — otherwise a steady write load starves reads.
                while self._writer or (self._waiting_writers and self._waiting_writers[0] < mine):
                    if deadline is None:
                        self._cond.wait()
                        continue
                    left = deadline - time.monotonic()
                    if left <= 0 or not self._cond.wait(left):
                        if self._writer or (
                            self._waiting_writers and self._waiting_writers[0] < mine
                        ):
                            raise MemoryBusy(
                                f"memory is busy writing; gave up after {timeout:.1f}s"
                            )
            finally:
                self._waiting_readers.remove(mine)
                # A writer may have been waiting solely on this reader's
                # place in line; leaving the queue changes its answer.
                self._cond.notify_all()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write(self, timeout: Optional[float] = None) -> Iterator[None]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            mine = self._ticket
            self._ticket += 1
            self._waiting_writers.append(mine)
            try:
                # Exclusive: no other writer, no active reader, and nobody
                # ahead in line — including readers, so a busy read load
                # cannot postpone indexing indefinitely.
                def _blocked() -> bool:
                    return bool(
                        self._writer
                        or self._readers
                        or (self._waiting_writers and self._waiting_writers[0] != mine)
                        or (self._waiting_readers and self._waiting_readers[0] < mine)
                    )

                while _blocked():
                    if deadline is None:
                        self._cond.wait()
                        continue
                    left = deadline - time.monotonic()
                    if (left <= 0 or not self._cond.wait(left)) and _blocked():
                        raise MemoryBusy(f"memory is busy; gave up after {timeout:.1f}s")
            finally:
                self._waiting_writers.remove(mine)
                self._cond.notify_all()
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()

    # ── introspection (tests, diagnostics) ───────────────────────────
    @property
    def readers(self) -> int:
        return self._readers

    @property
    def writing(self) -> bool:
        return self._writer
