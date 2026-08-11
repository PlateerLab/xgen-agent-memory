"""RWLock — effect-proving tests.

The property that matters is not "it locks" but *which* operations stop
waiting for each other. One mutex over the engine meant a search queued
behind every index, and a write that never finished took every read with it.
"""

from __future__ import annotations

import threading
import time

import pytest

from xgen_agent_memory._rwlock import MemoryBusy, RWLock


def _spawn(fn, n=1):
    threads = [threading.Thread(target=fn, daemon=True) for _ in range(n)]
    for t in threads:
        t.start()
    return threads


def test_readers_run_concurrently():
    """THE point. Under one mutex this test cannot pass."""
    lock = RWLock()
    both_in = threading.Barrier(3, timeout=5)
    ok = []

    def reader():
        with lock.read():
            both_in.wait()  # only reachable if both hold it at once
            ok.append(1)

    threads = _spawn(reader, 2)
    both_in.wait()
    for t in threads:
        t.join(5)
    assert len(ok) == 2


def test_a_writer_excludes_readers():
    lock = RWLock()
    writer_in = threading.Event()
    reader_got_in = threading.Event()
    release = threading.Event()

    def writer():
        with lock.write():
            writer_in.set()
            release.wait(5)

    def reader():
        with lock.read():
            reader_got_in.set()

    _spawn(writer)
    writer_in.wait(5)
    _spawn(reader)

    assert not reader_got_in.wait(0.2), "a reader entered during a write"
    release.set()
    assert reader_got_in.wait(5), "the reader never got in after the write"


def test_a_writer_waits_for_readers():
    lock = RWLock()
    reader_in = threading.Event()
    writer_got_in = threading.Event()
    release = threading.Event()

    def reader():
        with lock.read():
            reader_in.set()
            release.wait(5)

    def writer():
        with lock.write():
            writer_got_in.set()

    _spawn(reader)
    reader_in.wait(5)
    _spawn(writer)

    assert not writer_got_in.wait(0.2), "a writer ran during a read"
    release.set()
    assert writer_got_in.wait(5)


def test_writers_are_mutually_exclusive():
    lock = RWLock()
    concurrent = []
    active = {"n": 0}
    guard = threading.Lock()

    def writer():
        with lock.write():
            with guard:
                active["n"] += 1
                concurrent.append(active["n"])
            time.sleep(0.01)
            with guard:
                active["n"] -= 1

    threads = _spawn(writer, 4)
    for t in threads:
        t.join(5)
    assert max(concurrent) == 1


def test_a_waiting_writer_blocks_readers_that_arrive_after_it():
    """Arrival order. A reader that shows up AFTER a writer is queued waits
    its turn, so a steady retrieval load cannot postpone indexing — notes
    silently ceasing to be searchable while every dashboard stays green is
    the failure on that side."""
    lock = RWLock()
    first_reader_in = threading.Event()
    release_first = threading.Event()
    writer_waiting = threading.Event()
    late_reader_in = threading.Event()

    def first_reader():
        with lock.read():
            first_reader_in.set()
            release_first.wait(5)

    def writer():
        writer_waiting.set()
        with lock.write():
            pass

    def late_reader():
        with lock.read():
            late_reader_in.set()

    _spawn(first_reader)
    first_reader_in.wait(5)
    _spawn(writer)
    writer_waiting.wait(5)
    time.sleep(0.05)  # let the writer actually enqueue
    _spawn(late_reader)

    assert not late_reader_in.wait(0.2), "a late reader jumped the writer"
    release_first.set()
    assert late_reader_in.wait(5)


def test_the_lock_is_released_when_the_body_raises():
    lock = RWLock()

    with pytest.raises(ValueError):
        with lock.write():
            raise ValueError("boom")
    assert lock.writing is False

    with pytest.raises(ValueError):
        with lock.read():
            raise ValueError("boom")
    assert lock.readers == 0

    with lock.write():  # would hang if the first release leaked
        pass


def test_counters_return_to_zero():
    lock = RWLock()

    def churn():
        for _ in range(50):
            with lock.read():
                pass
            with lock.write():
                pass

    ts = _spawn(churn, 4)
    for t in ts:
        t.join(10)
    assert lock.readers == 0
    assert lock.writing is False


def test_a_continuous_writer_does_not_starve_readers():
    """The other side of fairness, and the reason strict writer preference
    was wrong here. A writer that re-queues the instant it releases would,
    under writer preference, never let a reader in at all — measured as a
    search benchmark that simply never finished against a live indexing
    load. Arrival order gives the waiting reader the next turn.
    """
    lock = RWLock()
    stop = threading.Event()
    reads = []

    def writer():
        while not stop.is_set():
            with lock.write():
                time.sleep(0.005)

    def reader():
        for _ in range(20):
            with lock.read():
                reads.append(1)

    w = _spawn(writer)
    time.sleep(0.02)  # let the write loop get going
    r = _spawn(reader)
    for t in r:
        t.join(10)
    stop.set()
    for t in w:
        t.join(5)

    assert len(reads) == 20, f"reader starved: only {len(reads)}/20 got through"


def test_a_continuous_reader_does_not_starve_writers():
    lock = RWLock()
    stop = threading.Event()
    writes = []

    def reader():
        while not stop.is_set():
            with lock.read():
                time.sleep(0.005)

    def writer():
        for _ in range(20):
            with lock.write():
                writes.append(1)

    rs = _spawn(reader, 3)
    time.sleep(0.02)
    w = _spawn(writer)
    for t in w:
        t.join(10)
    stop.set()
    for t in rs:
        t.join(5)

    assert len(writes) == 20, f"writer starved: only {len(writes)}/20 got through"


# ── bounded acquisition: the only thing that shrinks a wedge's blast radius ──


def test_a_reader_can_give_up_on_a_stuck_writer():
    """THE property. Splitting the lock does NOT stop a wedged write from
    blocking reads — an exclusive writer that never returns still holds
    everyone. A deadline does: retrieval fails fast and the caller proceeds
    without memory instead of inheriting someone else's hang."""
    lock = RWLock()
    writer_in = threading.Event()
    release = threading.Event()

    def stuck_writer():
        with lock.write():
            writer_in.set()
            release.wait(10)

    _spawn(stuck_writer)
    writer_in.wait(5)

    started = time.monotonic()
    with pytest.raises(MemoryBusy):
        with lock.read(timeout=0.1):
            pass
    assert time.monotonic() - started < 2.0, "gave up far later than asked"

    release.set()


def test_a_writer_can_give_up_too():
    lock = RWLock()
    reader_in = threading.Event()
    release = threading.Event()

    def stuck_reader():
        with lock.read():
            reader_in.set()
            release.wait(10)

    _spawn(stuck_reader)
    reader_in.wait(5)

    with pytest.raises(MemoryBusy):
        with lock.write(timeout=0.1):
            pass

    release.set()


def test_a_deadline_does_not_fire_when_the_lock_is_free():
    lock = RWLock()
    with lock.read(timeout=0.1):
        pass
    with lock.write(timeout=0.1):
        pass


def test_giving_up_leaves_the_lock_usable():
    """A waiter that timed out must not leave its place in the queue behind —
    the next writer would wait on a ticket nobody is holding."""
    lock = RWLock()
    writer_in = threading.Event()
    release = threading.Event()

    def stuck_writer():
        with lock.write():
            writer_in.set()
            release.wait(10)

    t = _spawn(stuck_writer)
    writer_in.wait(5)
    for _ in range(3):
        with pytest.raises(MemoryBusy):
            with lock.read(timeout=0.05):
                pass
    release.set()
    for th in t:
        th.join(5)

    with lock.write(timeout=2):  # would hang on a leaked queue entry
        pass
    with lock.read(timeout=2):
        pass
    assert lock.readers == 0 and lock.writing is False
