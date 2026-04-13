"""
DemandDrivenBroker — Dynamic REQ/REP work broker for FE² RVE scheduling.

The ``ScatterGatherCommunicator`` uses a queue-like PUSH/PULL pattern where
tasks are pre-scattered and workers consume them as available.  For highly
heterogeneous RVE costs (different convergence behavior, nonlinear response,
or mixed hardware), fixed pre-distribution can still produce idle workers.

The ``DemandDrivenBroker`` switches to strict ``REQ/REP`` handshaking:

* **Master (Broker):** Holds the full task queue and responds to each worker
  request with exactly one action: ``solve`` (new task), ``wait`` (temporarily
  no work), or ``shutdown`` (terminate worker loop).
* **Workers (RVE Solvers):** Explicitly request work only when ready.  The
  previous result can be piggybacked on the next request, so fast workers
  immediately receive more work while slow workers naturally receive less.

This gives true demand-driven scheduling and near-ideal dynamic load balancing.

Architecture diagram::

    ┌──────────┐   REQ: request / submit_result   ┌───────────────┐
    │ Worker 0 │ ─────────────────────────────────►│               │
    ├──────────┤                                    │   Master      │
    │ Worker 1 │ ─────────────────────────────────►│ (REP broker)  │
    ├──────────┤                                    │  task queue   │
    │ Worker N │ ─────────────────────────────────►│               │
    └──────────┘◄───────────────────────────────────│               │
                 REP: solve / wait / shutdown      └───────────────┘

Typical usage (Master)
----------------------
>>> broker = DemandDrivenBroker(role="master", endpoint="tcp://*:5556")
>>> results = broker.dispatch_gather(work_items, metadata=rve_metadata)
>>> broker.broadcast_shutdown(n_workers)

Typical usage (Worker)
----------------------
>>> broker = DemandDrivenBroker(role="worker", endpoint="tcp://master:5556")
>>> while True:
...     try:
...         idx, data, meta = broker.pull_work()
...     except StopIteration:
...         break
...     result = solve_rve(data, meta)
...     broker.push_result(idx, result)
"""
from __future__ import annotations

import json
import time
from typing import Any, Literal, Optional

import numpy as np
import zmq

from fenicsx_cosim.utils import get_logger

logger = get_logger(__name__)

# Default timeouts
_WORKER_TIMEOUT_MS = 60_000
_MASTER_TIMEOUT_MS = 300_000  # 5 minutes for tasks


class DemandDrivenBroker:
    """True demand-driven queue using ZMQ REQ/REP.
    
    Master holds a queue of items and dispenses exactly one item when a worker 
    requests it, achieving perfect dynamic load balancing regardless of container
    startup times or varying execution speeds.
    """

    def __init__(
        self,
        role: Literal["master", "worker"],
        endpoint: str = "tcp://*:5556",
        timeout_ms: Optional[int] = None,
    ) -> None:
        self.role = role
        self.endpoint = endpoint
        
        # Default to wait forever (-1) for long-running shakedown tasks
        if timeout_ms is None:
            timeout_ms = -1
            
        self.timeout_ms = int(timeout_ms)
        self._socket_timeout_ms = -1 if self.timeout_ms <= 0 else self.timeout_ms

        self._ctx = zmq.Context.instance()
        self._socket: Optional[zmq.Socket] = None
        self._connected = False
        
        # Worker state buffer to mimic ScatterGatherCommunicator push/pull API
        if role == "worker":
            # Start with a generic work request
            self._pending_req_frames = [json.dumps({"type": "request_work"}).encode("utf-8")]

        self._setup_socket()
        self._connected = True

        logger.info(
            "[DemandBroker/%s] Ready — endpoint=%s",
            role, endpoint,
        )

    def _setup_socket(self) -> None:
        if self.role == "master":
            self._socket = self._ctx.socket(zmq.REP)
            self._socket.setsockopt(zmq.LINGER, 1000)
            self._socket.setsockopt(zmq.RCVTIMEO, self._socket_timeout_ms)
            self._socket.bind(self.endpoint)
        else:
            self._socket = self._ctx.socket(zmq.REQ)
            self._socket.setsockopt(zmq.LINGER, 1000)
            self._socket.setsockopt(zmq.RCVTIMEO, self._socket_timeout_ms)
            self._socket.connect(self.endpoint)

    # ------------------------------------------------------------------
    # Master API
    # ------------------------------------------------------------------

    def dispatch_gather(
        self,
        work_items: list[np.ndarray],
        metadata: Optional[list[dict]] = None,
        on_result: Optional[Any] = None,
    ) -> list[np.ndarray]:
        """Serve work items to dynamic workers and gather results simultaneously.
        
        Blocks until all items are completed.
        """
        self._check_role("master", "dispatch_gather")
        n_items = len(work_items)
        results: list[Optional[np.ndarray]] = [None] * n_items
        pending_items = list(enumerate(work_items))
        completed = 0
        
        logger.info("[Master] Broker starting loop. %d total tasks in queue", n_items)
        
        while completed < n_items:
            try:
                frames = self._socket.recv_multipart()
            except zmq.Again:
                raise TimeoutError(
                    f"[Master] Broker timed out after {self.timeout_ms}ms "
                    f"waiting for worker request. ({completed}/{n_items} done)"
                )
                
            header = json.loads(frames[0].decode("utf-8"))
            req_type = header.get("type", "unknown")
            
            # Consume result if present
            if req_type == "submit_result":
                idx = header["index"]
                if results[idx] is None: 
                    # Extract result
                    array = np.frombuffer(
                        frames[1], dtype=np.dtype(header["dtype"])
                    ).reshape(header["shape"])
                    result_copy = array.copy()
                    results[idx] = result_copy
                    completed += 1
                    if on_result is not None:
                        on_result(idx, result_copy)
                    logger.debug("[Master] Received result for task %d (%d/%d done)", idx, completed, n_items)
            
            # Formulate immediately reply (either next task or wait signal)
            if pending_items:
                next_idx, next_item = pending_items.pop(0)
                
                res_header = {
                    "action": "solve",
                    "index": next_idx,
                    "dtype": str(next_item.dtype),
                    "shape": list(next_item.shape)
                }
                if metadata is not None and next_idx < len(metadata):
                    res_header["metadata"] = metadata[next_idx]
                    
                self._socket.send_multipart([
                    json.dumps(res_header).encode("utf-8"),
                    next_item.tobytes()
                ])
                logger.debug("[Master] Dispensed task %d to a waiting worker", next_idx)
            else:
                self._socket.send_multipart([json.dumps({"action": "wait"}).encode("utf-8")])
                
        logger.info("[Master] All %d tasks solved.", completed)
        # Results are guaranteed not to be None here because completed == n_items
        return results

    def broadcast_shutdown(self, n_workers: int) -> None:
        """Answer exactly n_worker requests with 'shutdown' action."""
        self._check_role("master", "broadcast_shutdown")
        logger.info("[Master] Broadcasting shutdown to %d workers...", n_workers)
        shutdowns = 0
        while shutdowns < n_workers:
            try:
                # Need a shorter timeout here in case some workers crashed
                frames = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.1)
                continue
                
            # Master ignores any late results during shutdown phase
            self._socket.send_multipart([json.dumps({"action": "shutdown"}).encode("utf-8")])
            shutdowns += 1
            
        logger.info("[Master] Shutdown complete.")

    # ------------------------------------------------------------------
    # Worker API
    # ------------------------------------------------------------------

    def pull_work(self) -> tuple[int, np.ndarray, dict]:
        """Request the next work item from the master instance."""
        self._check_role("worker", "pull_work")
        
        while True:
            # Pushing over REQ simultaneously sends previous result and requests new work
            self._socket.send_multipart(self._pending_req_frames)
            
            # Reset the buffer so we don't send the same result twice if we loop
            self._pending_req_frames = [json.dumps({"type": "request_work"}).encode("utf-8")]
            
            try:
                frames = self._socket.recv_multipart()
            except zmq.Again:
                raise TimeoutError(f"[Worker] Timed out waiting for master reply (timeout_ms={self.timeout_ms})")
                
            rep_header = json.loads(frames[0].decode("utf-8"))
            action = rep_header.get("action")
            
            if action == "shutdown":
                logger.info("[Worker] Received shutdown signal from broker.")
                raise StopIteration("Shutdown signal received")
            elif action == "wait":
                # The queue is currently empty, sleep and then loop to ask again
                time.sleep(1.0)
                continue
            elif action == "solve":
                idx = rep_header["index"]
                array = np.frombuffer(
                    frames[1], dtype=np.dtype(rep_header["dtype"])
                ).reshape(rep_header["shape"])
                meta = rep_header.get("metadata", {})
                
                logger.debug("[Worker] Claimed task %d", idx)
                return idx, array.copy(), meta
                
    def push_result(
        self,
        index: int,
        result: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> None:
        """Enqueue a result to be pushed back during the next pull_work request."""
        self._check_role("worker", "push_result")
        res_header: dict[str, Any] = {
            "type": "submit_result",
            "index": index,
            "dtype": str(result.dtype),
            "shape": list(result.shape),
        }
        if metadata is not None:
            res_header["metadata"] = metadata

        self._pending_req_frames = [
            json.dumps(res_header).encode("utf-8"),
            result.tobytes()
        ]
        logger.debug("[Worker] Queued result for task %d (Will send on next pull)", index)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if not self._connected:
            return
        if self._socket is not None and not self._socket.closed:
            self._socket.close()
        self._connected = False
        logger.info("[DemandBroker/%s] Socket closed", self.role)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def __enter__(self) -> "DemandDrivenBroker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _check_role(self, expected: str, method: str) -> None:
        if self.role != expected:
            raise RuntimeError(
                f"{method}() can only be called by a '{expected}', "
                f"but this communicator's role is '{self.role}'"
            )
        if not self._connected:
            raise RuntimeError(f"[{self.role}] Socket is not connected")
