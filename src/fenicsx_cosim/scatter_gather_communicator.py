"""
ScatterGatherCommunicator — Parallel RVE dispatch for FE² homogenization.

In a standard co-simulation, a ``zmq.PAIR`` socket connects exactly two
solvers.  For FE² however, a *single* macroscopic mesh might have 10 000+
integration points, each requiring an independent RVE solve.  Running
them sequentially is prohibitively slow.

The ``ScatterGatherCommunicator`` replaces the 1-to-1 topology with a
``PUSH/PULL`` fan-out / fan-in pattern:

* **Master (Macro-Solver):** Pushes an array of strain tensors to the
  work queue via a ``PUSH`` socket.  Gathers results from a ``PULL``
  socket.
* **Workers (Micro-Solvers / RVEs):** Each pulls a work item from the
  queue, solves the local RVE problem, and pushes the result back.

This corresponds to Section 1.3 of the Advanced Features Addendum.

Architecture diagram::

    ┌──────────┐  PUSH ──►  ┌────────────┐  PULL
    │  Master  │ ──────────►│            │ ──────────►  Worker 0
    │  (Macro) │ ──────────►│  Ventilator│ ──────────►  Worker 1
    │          │ ──────────►│  (queue)   │ ──────────►  Worker 2
    └──────────┘            └────────────┘             ...
         ▲                                              │
         │  PULL  ◄──  ┌────────────┐  PUSH ◄────────  │
         │ ◄────────── │  Collector │ ◄──────────────── │
         │ ◄────────── │  (gather)  │ ◄──────────────── │
                       └────────────┘

Typical usage (Master)
----------------------
>>> sg = ScatterGatherCommunicator(role="master",
...     push_endpoint="tcp://*:5556",
...     pull_endpoint="tcp://*:5557")
>>> sg.scatter(work_items)          # list of np.ndarray
>>> results = sg.gather(n_expected) # list of np.ndarray

Typical usage (Worker)
----------------------
>>> sg = ScatterGatherCommunicator(role="worker",
...     push_endpoint="tcp://master:5557",
...     pull_endpoint="tcp://master:5556")
>>> while True:
...     idx, data = sg.pull_work()
...     result = solve_rve(data)
...     sg.push_result(idx, result)
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal, Optional

import numpy as np
import zmq

from fenicsx_cosim.utils import (
    deserialize_array,
    get_logger,
    serialize_array,
)

logger = get_logger(__name__)

# Default timeouts
_SCATTER_TIMEOUT_MS = 60_000
_GATHER_TIMEOUT_MS = 300_000  # 5 minutes for RVE solves

# Sentinel to signal workers to shut down
SHUTDOWN_SIGNAL = b"COSIM_SHUTDOWN"


class ScatterGatherCommunicator:
    """Fan-out / fan-in communicator for dispatching RVE solves.

    Parameters
    ----------
    role : {"master", "worker"}
        Whether this process is the macro-solver (master) or an RVE
        worker.
    push_endpoint : str
        ZeroMQ endpoint for the PUSH socket.

        * Master: binds this endpoint (ventilator).
        * Worker: connects to this endpoint (to push results back).
    pull_endpoint : str
        ZeroMQ endpoint for the PULL socket.

        * Master: binds this endpoint (collector).
        * Worker: connects to this endpoint (to pull work items).
    timeout_ms : int, optional
        Receive timeout for blocking pulls.  Default 300 000 ms for master
        (waiting for RVE results), 60 000 ms for workers.
    """

    def __init__(
        self,
        role: Literal["master", "worker"],
        push_endpoint: str = "tcp://*:5556",
        pull_endpoint: str = "tcp://*:5557",
        timeout_ms: Optional[int] = None,
    ) -> None:
        self.role = role
        self.push_endpoint = push_endpoint
        self.pull_endpoint = pull_endpoint
        self._connected = False

        if timeout_ms is None:
            timeout_ms = (
                _GATHER_TIMEOUT_MS if role == "master" else _SCATTER_TIMEOUT_MS
            )
        self.timeout_ms = timeout_ms

        self._ctx = zmq.Context.instance()
        self._push_socket: Optional[zmq.Socket] = None
        self._pull_socket: Optional[zmq.Socket] = None

        self._setup_sockets()
        self._connected = True

        logger.info(
            "[ScatterGather/%s] Ready — push=%s, pull=%s",
            role, push_endpoint, pull_endpoint,
        )

    # ------------------------------------------------------------------
    # Socket setup
    # ------------------------------------------------------------------

    def _setup_sockets(self) -> None:
        """Create and bind/connect PUSH and PULL sockets."""
        if self.role == "master":
            # Master PUSHES work out (ventilator)
            self._push_socket = self._ctx.socket(zmq.PUSH)
            self._push_socket.setsockopt(zmq.LINGER, 1000)
            self._push_socket.setsockopt(zmq.SNDHWM, 0)  # unlimited queue
            self._push_socket.bind(self.push_endpoint)

            # Master PULLS results back (collector)
            self._pull_socket = self._ctx.socket(zmq.PULL)
            self._pull_socket.setsockopt(zmq.LINGER, 1000)
            self._pull_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self._pull_socket.bind(self.pull_endpoint)

            logger.debug(
                "[Master] Bound PUSH to %s, PULL to %s",
                self.push_endpoint, self.pull_endpoint,
            )
        else:
            # Worker PULLS work items from master's ventilator
            self._pull_socket = self._ctx.socket(zmq.PULL)
            self._pull_socket.setsockopt(zmq.LINGER, 1000)
            self._pull_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self._pull_socket.connect(self.pull_endpoint)

            # Worker PUSHES results to master's collector
            self._push_socket = self._ctx.socket(zmq.PUSH)
            self._push_socket.setsockopt(zmq.LINGER, 1000)
            self._push_socket.connect(self.push_endpoint)

            logger.debug(
                "[Worker] Connected PULL to %s, PUSH to %s",
                self.pull_endpoint, self.push_endpoint,
            )

    # ------------------------------------------------------------------
    # Master API
    # ------------------------------------------------------------------

    def scatter(
        self,
        work_items: list[np.ndarray],
        metadata: Optional[list[dict]] = None,
    ) -> int:
        """Scatter work items to the worker pool.

        Each work item is assigned a sequential index and sent as a
        multipart ZeroMQ message.

        Parameters
        ----------
        work_items : list[np.ndarray]
            One array per work unit (e.g. strain tensor per cell/RVE).
        metadata : list[dict], optional
            Optional metadata dicts to send alongside each work item
            (e.g. material properties, damage state).

        Returns
        -------
        int
            The number of work items dispatched.
        """
        self._check_role("master", "scatter")
        n = len(work_items)

        for idx, item in enumerate(work_items):
            header = {
                "index": idx,
                "total": n,
                "dtype": str(item.dtype),
                "shape": list(item.shape),
            }
            if metadata is not None and idx < len(metadata):
                header["metadata"] = metadata[idx]

            header_bytes = json.dumps(header).encode("utf-8")
            payload_bytes = item.tobytes()
            self._push_socket.send_multipart([header_bytes, payload_bytes])

        logger.info("[Master] Scattered %d work items", n)
        return n

    def gather(
        self,
        n_expected: int,
        ordered: bool = True,
    ) -> list[np.ndarray]:
        """Gather results from the worker pool.

        Blocks until all *n_expected* results have been received.

        Parameters
        ----------
        n_expected : int
            Number of results to wait for.
        ordered : bool, optional
            If ``True`` (default), results are returned in the same order
            as the scattered work items (by index).  If ``False``,
            results are returned in arrival order.

        Returns
        -------
        list[np.ndarray]
            The gathered results.

        Raises
        ------
        TimeoutError
            If waiting for results exceeds ``timeout_ms``.
        """
        self._check_role("master", "gather")
        results: dict[int, np.ndarray] = {}
        start_time = time.time()

        while len(results) < n_expected:
            try:
                frames = self._pull_socket.recv_multipart()
            except zmq.Again:
                elapsed = time.time() - start_time
                raise TimeoutError(
                    f"[Master] Gather timed out after {elapsed:.1f}s — "
                    f"received {len(results)}/{n_expected} results"
                )

            header = json.loads(frames[0].decode("utf-8"))
            idx = header["index"]
            array = np.frombuffer(
                frames[1], dtype=np.dtype(header["dtype"])
            ).reshape(header["shape"])

            results[idx] = array.copy()
            logger.debug(
                "[Master] Gathered result %d/%d (index=%d)",
                len(results), n_expected, idx,
            )

        logger.info("[Master] Gathered all %d results", n_expected)

        if ordered:
            return [results[i] for i in range(n_expected)]
        else:
            return list(results.values())

    def scatter_gather(
        self,
        work_items: list[np.ndarray],
        metadata: Optional[list[dict]] = None,
    ) -> list[np.ndarray]:
        """Convenience: scatter work items and then gather all results.

        Parameters
        ----------
        work_items : list[np.ndarray]
            Work items to dispatch.
        metadata : list[dict], optional
            Optional per-item metadata.

        Returns
        -------
        list[np.ndarray]
            Results in the same order as the input work items.
        """
        n = self.scatter(work_items, metadata)
        return self.gather(n)

    def broadcast_shutdown(self, n_workers: int) -> None:
        """Send shutdown signals to all workers.

        Parameters
        ----------
        n_workers : int
            Number of workers to notify.
        """
        self._check_role("master", "broadcast_shutdown")
        for _ in range(n_workers):
            self._push_socket.send(SHUTDOWN_SIGNAL)
        logger.info("[Master] Sent shutdown signal to %d workers", n_workers)

    # ------------------------------------------------------------------
    # Worker API
    # ------------------------------------------------------------------

    def pull_work(self) -> tuple[int, np.ndarray, dict]:
        """Pull a single work item from the master.

        Returns
        -------
        tuple[int, np.ndarray, dict]
            ``(index, data_array, metadata_dict)``

        Raises
        ------
        StopIteration
            If a shutdown signal is received.
        TimeoutError
            If no work arrives within ``timeout_ms``.
        """
        self._check_role("worker", "pull_work")
        try:
            frames = self._pull_socket.recv_multipart()
        except zmq.Again:
            raise TimeoutError("[Worker] Pull timed out waiting for work")

        # Check for shutdown
        if len(frames) == 1 and frames[0] == SHUTDOWN_SIGNAL:
            logger.info("[Worker] Received shutdown signal")
            raise StopIteration("Shutdown signal received")

        header = json.loads(frames[0].decode("utf-8"))
        idx = header["index"]
        array = np.frombuffer(
            frames[1], dtype=np.dtype(header["dtype"])
        ).reshape(header["shape"])

        metadata = header.get("metadata", {})

        logger.debug(
            "[Worker] Pulled work item %d/%d",
            idx + 1, header.get("total", "?"),
        )
        return idx, array.copy(), metadata

    def push_result(
        self,
        index: int,
        result: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> None:
        """Push a completed result back to the master.

        Parameters
        ----------
        index : int
            The work item index (received from :meth:`pull_work`).
        result : np.ndarray
            The computed result for this work item.
        metadata : dict, optional
            Optional result metadata.
        """
        self._check_role("worker", "push_result")
        header: dict[str, Any] = {
            "index": index,
            "dtype": str(result.dtype),
            "shape": list(result.shape),
        }
        if metadata is not None:
            header["metadata"] = metadata

        header_bytes = json.dumps(header).encode("utf-8")
        payload_bytes = result.tobytes()
        self._push_socket.send_multipart([header_bytes, payload_bytes])
        logger.debug("[Worker] Pushed result for index %d", index)

    def work_loop(self, solve_fn, **kwargs) -> int:
        """Convenience: pull work items and push results until shutdown.

        Parameters
        ----------
        solve_fn : callable
            ``solve_fn(index: int, data: np.ndarray, metadata: dict)
            -> np.ndarray``  — the RVE solver function.
        **kwargs
            Additional keyword arguments passed to *solve_fn*.

        Returns
        -------
        int
            Number of work items processed before shutdown.
        """
        self._check_role("worker", "work_loop")
        count = 0
        while True:
            try:
                idx, data, meta = self.pull_work()
            except StopIteration:
                break
            except TimeoutError:
                logger.warning("[Worker] Timed out — assuming no more work")
                break

            result = solve_fn(idx, data, meta, **kwargs)
            self.push_result(idx, result)
            count += 1

        logger.info("[Worker] Processed %d work items", count)
        return count

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close all sockets."""
        if not self._connected:
            return
        for sock in (self._push_socket, self._pull_socket):
            if sock is not None and not sock.closed:
                sock.close()
        self._connected = False
        logger.info("[ScatterGather/%s] Sockets closed", self.role)

    @property
    def is_connected(self) -> bool:
        """Whether the sockets are open."""
        return self._connected

    def __enter__(self) -> "ScatterGatherCommunicator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_role(self, expected: str, method: str) -> None:
        if self.role != expected:
            raise RuntimeError(
                f"{method}() can only be called by a '{expected}', "
                f"but this communicator's role is '{self.role}'"
            )
        if not self._connected:
            raise RuntimeError(
                f"[{self.role}] Sockets are not connected"
            )
