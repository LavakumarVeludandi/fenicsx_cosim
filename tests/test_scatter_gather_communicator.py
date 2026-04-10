"""
Tests for the ScatterGatherCommunicator module.

These tests verify:
  - PUSH/PULL socket setup for both master and worker roles
  - Scatter and gather of work items with ordering
  - Worker pull/push lifecycle
  - work_loop convenience method
  - Shutdown signal propagation
  - Role enforcement
  - Metadata passing

Uses real ZeroMQ sockets in separate threads.
"""

import threading
import time

import numpy as np
import pytest

from fenicsx_cosim.scatter_gather_communicator import (
    SHUTDOWN_SIGNAL,
    ScatterGatherCommunicator,
)


# ======================================================================
# Helpers
# ======================================================================

def _get_free_ports():
    """Return two distinct ports for testing."""
    import socket
    socks = []
    ports = []
    for _ in range(2):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        ports.append(s.getsockname()[1])
        socks.append(s)
    for s in socks:
        s.close()
    return ports


@pytest.fixture
def endpoints():
    """Return push and pull endpoints using free ports."""
    ports = _get_free_ports()
    push_ep = f"tcp://127.0.0.1:{ports[0]}"
    pull_ep = f"tcp://127.0.0.1:{ports[1]}"
    return push_ep, pull_ep


# ======================================================================
# Tests: Construction
# ======================================================================

class TestConstruction:
    """Tests for socket creation and role assignment."""

    def test_master_creation(self, endpoints):
        push_ep, pull_ep = endpoints
        sg = ScatterGatherCommunicator(
            role="master",
            push_endpoint=push_ep,
            pull_endpoint=pull_ep,
        )
        assert sg.role == "master"
        assert sg.is_connected
        sg.close()
        assert not sg.is_connected

    def test_worker_creation(self, endpoints):
        push_ep, pull_ep = endpoints
        # Workers connect to master's endpoints (reversed)
        # First create master so endpoints are bound
        master = ScatterGatherCommunicator(
            role="master",
            push_endpoint=push_ep,
            pull_endpoint=pull_ep,
        )
        worker = ScatterGatherCommunicator(
            role="worker",
            push_endpoint=pull_ep,  # worker pushes to master's pull
            pull_endpoint=push_ep,  # worker pulls from master's push
            timeout_ms=5000,
        )
        assert worker.role == "worker"
        assert worker.is_connected

        worker.close()
        master.close()

    def test_context_manager(self, endpoints):
        push_ep, pull_ep = endpoints
        with ScatterGatherCommunicator(
            role="master",
            push_endpoint=push_ep,
            pull_endpoint=pull_ep,
        ) as sg:
            assert sg.is_connected
        assert not sg.is_connected


# ======================================================================
# Tests: Role enforcement
# ======================================================================

class TestRoleEnforcement:
    """Tests that methods enforce correct roles."""

    def test_scatter_requires_master(self, endpoints):
        push_ep, pull_ep = endpoints
        # Create a master first so worker can connect
        master = ScatterGatherCommunicator(
            role="master",
            push_endpoint=push_ep,
            pull_endpoint=pull_ep,
        )
        worker = ScatterGatherCommunicator(
            role="worker",
            push_endpoint=pull_ep,
            pull_endpoint=push_ep,
            timeout_ms=5000,
        )
        with pytest.raises(RuntimeError, match="master"):
            worker.scatter([np.array([1.0])])

        worker.close()
        master.close()

    def test_pull_work_requires_worker(self, endpoints):
        push_ep, pull_ep = endpoints
        master = ScatterGatherCommunicator(
            role="master",
            push_endpoint=push_ep,
            pull_endpoint=pull_ep,
        )
        with pytest.raises(RuntimeError, match="worker"):
            master.pull_work()

        master.close()


# ======================================================================
# Tests: Scatter / Gather
# ======================================================================

class TestScatterGather:
    """Tests for the scatter/gather workflow."""

    def test_scatter_gather_single_worker(self, endpoints):
        push_ep, pull_ep = endpoints
        n_items = 5

        master = ScatterGatherCommunicator(
            role="master",
            push_endpoint=push_ep,
            pull_endpoint=pull_ep,
            timeout_ms=10_000,
        )
        # Give master time to bind
        time.sleep(0.1)

        worker = ScatterGatherCommunicator(
            role="worker",
            push_endpoint=pull_ep,
            pull_endpoint=push_ep,
            timeout_ms=10_000,
        )
        time.sleep(0.1)

        # Prepare work items
        work_items = [np.array([float(i), float(i * 10)]) for i in range(n_items)]

        results = [None]
        errors = [None]

        def master_side():
            try:
                master.scatter(work_items)
                results[0] = master.gather(n_items)
            except Exception as e:
                errors[0] = e

        def worker_side():
            try:
                for _ in range(n_items):
                    idx, data, meta = worker.pull_work()
                    result = data * 2.0  # Simple "RVE solve"
                    worker.push_result(idx, result)
            except Exception as e:
                errors[0] = e

        t_m = threading.Thread(target=master_side)
        t_w = threading.Thread(target=worker_side)
        t_m.start()
        t_w.start()
        t_m.join(timeout=15)
        t_w.join(timeout=15)

        assert errors[0] is None, f"Error: {errors[0]}"
        assert results[0] is not None
        assert len(results[0]) == n_items

        # Results should be ordered and doubled
        for i in range(n_items):
            expected = np.array([float(i * 2), float(i * 20)])
            np.testing.assert_array_almost_equal(results[0][i], expected)

        master.close()
        worker.close()

    def test_scatter_gather_multiple_workers(self, endpoints):
        push_ep, pull_ep = endpoints
        n_items = 8
        n_workers = 2

        master = ScatterGatherCommunicator(
            role="master",
            push_endpoint=push_ep,
            pull_endpoint=pull_ep,
            timeout_ms=15_000,
        )
        time.sleep(0.1)

        workers = []
        for _ in range(n_workers):
            w = ScatterGatherCommunicator(
                role="worker",
                push_endpoint=pull_ep,
                pull_endpoint=push_ep,
                timeout_ms=15_000,
            )
            workers.append(w)
        time.sleep(0.1)

        work_items = [np.array([float(i)]) for i in range(n_items)]

        results = [None]
        errors = [None]

        def master_side():
            try:
                master.scatter(work_items)
                results[0] = master.gather(n_items)
            except Exception as e:
                errors[0] = e

        def worker_side(w):
            try:
                while True:
                    try:
                        idx, data, meta = w.pull_work()
                        result = data ** 2  # "RVE solve"
                        w.push_result(idx, result)
                    except (StopIteration, TimeoutError):
                        break
            except Exception as e:
                pass  # Workers may time out, that's OK

        threads = [threading.Thread(target=master_side)]
        for w in workers:
            threads.append(threading.Thread(target=worker_side, args=(w,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)

        assert errors[0] is None, f"Error: {errors[0]}"
        assert results[0] is not None
        assert len(results[0]) == n_items

        # Results should be ordered (squared)
        for i in range(n_items):
            expected = np.array([float(i ** 2)])
            np.testing.assert_array_almost_equal(results[0][i], expected)

        master.close()
        for w in workers:
            w.close()

    def test_scatter_gather_convenience(self, endpoints):
        push_ep, pull_ep = endpoints
        n_items = 3

        master = ScatterGatherCommunicator(
            role="master",
            push_endpoint=push_ep,
            pull_endpoint=pull_ep,
            timeout_ms=10_000,
        )
        time.sleep(0.1)

        worker = ScatterGatherCommunicator(
            role="worker",
            push_endpoint=pull_ep,
            pull_endpoint=push_ep,
            timeout_ms=10_000,
        )
        time.sleep(0.1)

        work_items = [np.array([1.0, 2.0, 3.0])] * n_items
        results = [None]

        def master_side():
            results[0] = master.scatter_gather(work_items)

        def worker_side():
            for _ in range(n_items):
                idx, data, meta = worker.pull_work()
                worker.push_result(idx, data + 1.0)

        t_m = threading.Thread(target=master_side)
        t_w = threading.Thread(target=worker_side)
        t_m.start()
        t_w.start()
        t_m.join(timeout=10)
        t_w.join(timeout=10)

        assert results[0] is not None
        for r in results[0]:
            np.testing.assert_array_almost_equal(r, np.array([2.0, 3.0, 4.0]))

        master.close()
        worker.close()


# ======================================================================
# Tests: Metadata
# ======================================================================

class TestMetadata:
    """Tests for metadata passing with work items."""

    def test_metadata_roundtrip(self, endpoints):
        push_ep, pull_ep = endpoints

        master = ScatterGatherCommunicator(
            role="master",
            push_endpoint=push_ep,
            pull_endpoint=pull_ep,
            timeout_ms=10_000,
        )
        time.sleep(0.1)

        worker = ScatterGatherCommunicator(
            role="worker",
            push_endpoint=pull_ep,
            pull_endpoint=push_ep,
            timeout_ms=10_000,
        )
        time.sleep(0.1)

        work_items = [np.array([1.0])]
        metadata = [{"material": "steel", "E": 210e9}]

        received_meta = [None]

        def master_side():
            master.scatter(work_items, metadata)
            master.gather(1)

        def worker_side():
            idx, data, meta = worker.pull_work()
            received_meta[0] = meta
            worker.push_result(idx, data)

        t_m = threading.Thread(target=master_side)
        t_w = threading.Thread(target=worker_side)
        t_m.start()
        t_w.start()
        t_m.join(timeout=10)
        t_w.join(timeout=10)

        assert received_meta[0] is not None
        assert received_meta[0]["material"] == "steel"
        assert received_meta[0]["E"] == 210e9

        master.close()
        worker.close()


# ======================================================================
# Tests: Shutdown
# ======================================================================

class TestShutdown:
    """Tests for shutdown signal handling."""

    def test_shutdown_stops_worker(self, endpoints):
        push_ep, pull_ep = endpoints

        master = ScatterGatherCommunicator(
            role="master",
            push_endpoint=push_ep,
            pull_endpoint=pull_ep,
        )
        time.sleep(0.1)

        worker = ScatterGatherCommunicator(
            role="worker",
            push_endpoint=pull_ep,
            pull_endpoint=push_ep,
            timeout_ms=10_000,
        )
        time.sleep(0.1)

        count = [0]

        def worker_loop():
            count[0] = worker.work_loop(
                lambda idx, data, meta: data * 2.0
            )

        t = threading.Thread(target=worker_loop)
        t.start()

        # Send some work, then shutdown
        master.scatter([np.array([1.0]), np.array([2.0])])
        time.sleep(0.5)
        master.broadcast_shutdown(1)
        # Give time for results
        time.sleep(0.5)

        t.join(timeout=10)
        assert count[0] == 2  # processed 2 items before shutdown

        master.close()
        worker.close()


# ======================================================================
# Tests: Syphon prevention (SNDHWM regression)
# ======================================================================

class TestSyphonPrevention:
    """Regression test: no single worker should drain the entire queue.

    This guards against the early-joiner syphon bug where SNDHWM=0 allowed
    the first-connected worker to buffer and process all work items before
    other workers had a chance to connect.
    """

    def test_fair_distribution_4_workers_8_items(self):
        """With 4 workers and 8 items, no worker should get all 8 items.

        A perfect split is 2 each, but ZMQ round-robin under load can vary.
        We only assert that no single worker monopolises the queue (the syphon
        case), meaning each worker must get at least 1 item.
        """
        ports = _get_free_ports()
        # Need 4 port pairs — reuse helper for additional ports
        import socket
        extra_socks = []
        extra_ports = []
        for _ in range(6):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", 0))
            extra_ports.append(s.getsockname()[1])
            extra_socks.append(s)
        for s in extra_socks:
            s.close()

        push_ep = f"tcp://127.0.0.1:{ports[0]}"
        pull_ep = f"tcp://127.0.0.1:{ports[1]}"

        n_items = 8
        n_workers = 4

        master = ScatterGatherCommunicator(
            role="master",
            push_endpoint=push_ep,
            pull_endpoint=pull_ep,
            timeout_ms=15_000,
            sndhwm=n_workers,  # Fix 1: throttle to 1 per connected peer
        )
        time.sleep(0.1)

        workers = []
        for _ in range(n_workers):
            w = ScatterGatherCommunicator(
                role="worker",
                push_endpoint=pull_ep,
                pull_endpoint=push_ep,
                timeout_ms=10_000,
            )
            workers.append(w)

        # Give all workers time to connect before scatter
        time.sleep(0.3)

        work_items = [np.array([float(i)]) for i in range(n_items)]
        items_per_worker = [0] * n_workers
        lock = threading.Lock()
        results = [None]

        def master_side():
            master.scatter(work_items)
            results[0] = master.gather(n_items)
            master.broadcast_shutdown(n_workers)

        def worker_side(w, worker_idx):
            while True:
                try:
                    idx, data, meta = w.pull_work()
                    with lock:
                        items_per_worker[worker_idx] += 1
                    w.push_result(idx, data)
                except StopIteration:
                    break
                except TimeoutError:
                    break

        threads = [threading.Thread(target=master_side)]
        for i, w in enumerate(workers):
            threads.append(threading.Thread(target=worker_side, args=(w, i)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)

        assert results[0] is not None
        assert len(results[0]) == n_items

        # Core assertion: no single worker syphoned all items
        assert max(items_per_worker) < n_items, (
            f"Syphon detected: one worker processed all {n_items} items. "
            f"Distribution: {items_per_worker}"
        )
        # Every worker should have received at least one item
        assert all(c > 0 for c in items_per_worker), (
            f"Some workers received no work (starvation). "
            f"Distribution: {items_per_worker}"
        )

        master.close()
        for w in workers:
            w.close()


# ======================================================================
# Tests: work_loop
# ======================================================================

class TestWorkLoop:
    """Tests for the worker convenience work_loop."""

    def test_work_loop_processes_items(self, endpoints):
        push_ep, pull_ep = endpoints
        n_items = 4

        master = ScatterGatherCommunicator(
            role="master",
            push_endpoint=push_ep,
            pull_endpoint=pull_ep,
            timeout_ms=10_000,
        )
        time.sleep(0.1)

        worker = ScatterGatherCommunicator(
            role="worker",
            push_endpoint=pull_ep,
            pull_endpoint=push_ep,
            timeout_ms=5_000,
        )
        time.sleep(0.1)

        work_items = [np.array([float(i)]) for i in range(n_items)]

        def solve_rve(idx, data, meta):
            return data * 3.0

        results = [None]
        count = [0]

        def master_side():
            master.scatter(work_items)
            results[0] = master.gather(n_items)
            # After gathering, send shutdown
            master.broadcast_shutdown(1)

        def worker_side():
            count[0] = worker.work_loop(solve_rve)

        t_m = threading.Thread(target=master_side)
        t_w = threading.Thread(target=worker_side)
        t_m.start()
        t_w.start()
        t_m.join(timeout=15)
        t_w.join(timeout=15)

        assert results[0] is not None
        assert count[0] == n_items
        for i in range(n_items):
            np.testing.assert_array_almost_equal(
                results[0][i], np.array([float(i * 3)])
            )

        master.close()
        worker.close()
