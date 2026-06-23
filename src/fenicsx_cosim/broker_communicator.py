"""N>2 multi-solver coupling over a ZeroMQ ROUTER/DEALER broker.

The :class:`Communicator` PAIR socket is strictly 1-to-1. Coupling three or more
participants (e.g. fluid + thermal + structural) needs a routing fabric. This
module provides:

* :class:`CouplingBroker` — a ROUTER that participants connect to. It registers
  participants by name, releases a startup barrier once the expected count has
  joined (so every identity is known to the router before any data is routed),
  and forwards each ``(dest, payload)`` message to the named destination.
* :class:`BrokerClient` — a DEALER (identity = participant name) with the same
  named-array wire format as :class:`Communicator`: ``send(dest, name, array)``
  and ``receive() -> (src, name, array)``, plus an all-participant ``barrier()``.

Wire protocol (frames after ZeroMQ's automatic identity handling)
-----------------------------------------------------------------
* client -> broker, data:    ``[dest_name, *serialize_array(name, array)]``
* broker -> client, data:    ``[src_name, *serialize_array(name, array)]``
* control frames are prefixed with the sentinel :data:`_CTRL` and carry a verb
  (``REGISTER`` / ``READY`` / ``BARRIER`` / ``SHUTDOWN``).

A central broker (not a brokerless mesh) is used deliberately: named routing,
a clean join barrier, and N-way ``barrier()`` are all trivial with one ROUTER,
and it mirrors the hub model coupling tools (preCICE m2n, Kratos CoSimIO) use.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import zmq

from fenicsx_cosim.utils import (
    deserialize_array,
    get_logger,
    serialize_array,
)

logger = get_logger(__name__)

_CTRL = b"__CTRL__"
_REGISTER = b"REGISTER"
_READY = b"READY"
_BARRIER = b"BARRIER"
_SHUTDOWN = b"SHUTDOWN"

_DEFAULT_TIMEOUT_MS = 30_000


class CouplingBroker:
    """ROUTER-side hub that registers participants and routes named messages.

    Parameters
    ----------
    endpoint : str
        Bind endpoint (e.g. ``"tcp://*:5560"``).
    expected : int, optional
        Number of participants to wait for before releasing the join barrier.
        If ``None``, ``READY`` is never auto-sent (participants skip the barrier).
    poll_ms : int
        Poll interval controlling how responsively :meth:`stop` is honored.
    """

    def __init__(self, endpoint: str = "tcp://*:5560",
                 expected: Optional[int] = None, poll_ms: int = 200) -> None:
        self.endpoint = endpoint
        self.expected = expected
        self.poll_ms = poll_ms
        self._ctx = zmq.Context.instance()
        self._router = self._ctx.socket(zmq.ROUTER)
        self._router.setsockopt(zmq.LINGER, 1000)
        self._router.bind(endpoint)
        self._members: set[bytes] = set()
        self._barrier_waiting: set[bytes] = set()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        logger.info("[broker] bound to %s (expected=%s)", endpoint, expected)

    # -- lifecycle ------------------------------------------------------
    def start(self) -> "CouplingBroker":
        """Run the routing loop in a background thread."""
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if not self._router.closed:
            self._router.close()

    # -- main loop ------------------------------------------------------
    def run(self) -> None:
        poller = zmq.Poller()
        poller.register(self._router, zmq.POLLIN)
        while not self._stop.is_set():
            if not dict(poller.poll(self.poll_ms)):
                continue
            frames = self._router.recv_multipart()
            sender, body = frames[0], frames[1:]
            if body and body[0] == _CTRL:
                self._handle_control(sender, body[1:])
            else:
                # data: body = [dest_name, *payload] -> [sender_name, *payload]
                dest = body[0]
                self._router.send_multipart([dest, sender] + body[1:])

    def _handle_control(self, sender: bytes, ctrl: list[bytes]) -> None:
        verb = ctrl[0]
        if verb == _REGISTER:
            self._members.add(sender)
            logger.info("[broker] registered '%s' (%d/%s)",
                        sender.decode(), len(self._members), self.expected)
            if self.expected is not None and len(self._members) == self.expected:
                for m in self._members:
                    self._router.send_multipart([m, _CTRL, _READY])
                logger.info("[broker] all %d joined — READY broadcast",
                            self.expected)
        elif verb == _BARRIER:
            self._barrier_waiting.add(sender)
            if len(self._barrier_waiting) == len(self._members):
                for m in self._members:
                    self._router.send_multipart([m, _CTRL, _BARRIER])
                self._barrier_waiting.clear()
        elif verb == _SHUTDOWN:
            self._stop.set()


class BrokerError(Exception):
    """Raised on broker-client communication failure."""


class BrokerClient:
    """DEALER-side participant in an N-way broker-coupled simulation.

    Parameters
    ----------
    name : str
        Unique participant identifier (becomes the ZeroMQ DEALER identity).
    endpoint : str
        Broker endpoint to connect to (e.g. ``"tcp://localhost:5560"``).
    timeout_ms : int
        Blocking-receive timeout.
    """

    def __init__(self, name: str, endpoint: str = "tcp://localhost:5560",
                 timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> None:
        self.name = name
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.DEALER)
        self._sock.setsockopt(zmq.IDENTITY, name.encode())
        self._sock.setsockopt(zmq.LINGER, 1000)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.connect(endpoint)
        logger.info("[%s] connected to broker %s", name, endpoint)

    def register(self, join_timeout_ms: int = 60_000) -> None:
        """Announce presence and block until the broker's join barrier releases.

        After this returns, every participant's identity is known to the router,
        so peer-to-peer :meth:`send` will not be silently dropped.
        """
        self._sock.send_multipart([_CTRL, _REGISTER])
        old = self._sock.getsockopt(zmq.RCVTIMEO)
        self._sock.setsockopt(zmq.RCVTIMEO, join_timeout_ms)
        try:
            while True:
                frames = self._sock.recv_multipart()
                if frames and frames[0] == _CTRL and frames[1] == _READY:
                    logger.info("[%s] join barrier released", self.name)
                    return
        except zmq.Again:
            raise BrokerError(f"[{self.name}] registration timed out")
        finally:
            self._sock.setsockopt(zmq.RCVTIMEO, old)

    def send(self, dest: str, data_name: str, array: np.ndarray) -> None:
        """Send a named array to participant ``dest`` via the broker."""
        self._sock.send_multipart([dest.encode()] + serialize_array(data_name, array))
        logger.debug("[%s] -> %s : '%s' %s", self.name, dest, data_name,
                     getattr(array, "shape", None))

    def receive(self) -> tuple[str, str, np.ndarray]:
        """Receive one message. Returns ``(src_name, data_name, array)``."""
        try:
            frames = self._sock.recv_multipart()
        except zmq.Again:
            raise BrokerError(f"[{self.name}] receive timed out")
        if frames and frames[0] == _CTRL:
            raise BrokerError(
                f"[{self.name}] unexpected control frame {frames[1:]!r}")
        src = frames[0].decode()
        name, array = deserialize_array(frames[1:])
        return src, name, array

    def barrier(self) -> None:
        """Block until every registered participant reaches the barrier."""
        self._sock.send_multipart([_CTRL, _BARRIER])
        while True:
            frames = self._sock.recv_multipart()
            if frames and frames[0] == _CTRL and frames[1] == _BARRIER:
                return
            raise BrokerError(
                f"[{self.name}] expected BARRIER, got data from "
                f"{frames[0].decode()}")

    def shutdown_broker(self) -> None:
        """Ask the broker to stop its routing loop."""
        self._sock.send_multipart([_CTRL, _SHUTDOWN])

    def close(self) -> None:
        if not self._sock.closed:
            self._sock.close()

    def __enter__(self) -> "BrokerClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
