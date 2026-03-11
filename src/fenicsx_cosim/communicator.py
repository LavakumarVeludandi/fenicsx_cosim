"""
Communicator — ZeroMQ-based inter-process communication engine.

This module wraps ``pyzmq`` to handle the actual network communication
between coupled FEniCSx solver processes.  It is analogous to the
communication backends in Kratos CoSimIO (file-based, socket-based) but
uses ZeroMQ exclusively, which:

* Does **not** interfere with FEniCSx's internal MPI communicators.
* Supports TCP/IP for distributed runs and IPC (Unix sockets) for
  same-machine runs.
* Handles serialization of NumPy arrays via the helpers in :mod:`utils`.

Typical usage
-------------
>>> comm = Communicator(name="ThermalSolver", partner_name="MechSolver",
...                     role="connect", endpoint="tcp://localhost:5555")
>>> comm.send_array("Temperature", temp_array)
>>> name, disp_array = comm.receive_array()
>>> comm.close()
"""

from __future__ import annotations

import time
from typing import Literal, Optional

import numpy as np
import zmq

from fenicsx_cosim.utils import (
    HANDSHAKE_MAGIC,
    SYNC_ACK,
    SYNC_SIGNAL,
    deserialize_array,
    get_logger,
    make_handshake_msg,
    parse_handshake_msg,
    serialize_array,
)

logger = get_logger(__name__)

# Default timeout for blocking operations (ms).
_DEFAULT_TIMEOUT_MS = 30_000  # 30 seconds
_HANDSHAKE_TIMEOUT_MS = 60_000  # 60 seconds


class CommunicationError(Exception):
    """Raised when a communication operation fails."""


class Communicator:
    """ZeroMQ-based communicator for inter-process data exchange.

    Two coupled solvers each create a ``Communicator``.  One must take the
    ``"bind"`` role (the server) and the other the ``"connect"`` role (the
    client).  The underlying socket type is ``zmq.PAIR`` which gives
    bidirectional, exclusive communication — ideal for a two-solver
    coupling scenario.

    Parameters
    ----------
    name : str
        Identifier for *this* solver.
    partner_name : str
        Expected identifier of the partner solver.
    role : {"bind", "connect"}
        Whether this communicator binds (acts as server) or connects
        (acts as client).
    endpoint : str, optional
        A ZeroMQ endpoint string.  Defaults to ``"tcp://localhost:5555"``.
        For same-machine IPC use ``"ipc:///tmp/fenicsx_cosim"``.
    timeout_ms : int, optional
        Timeout in milliseconds for blocking receives.  Default 30 000.
    handshake : bool, optional
        If ``True`` (default), perform a name-exchange handshake upon
        initialization to verify that both sides are correctly paired.
    """

    def __init__(
        self,
        name: str,
        partner_name: str,
        role: Literal["bind", "connect"],
        endpoint: str = "tcp://localhost:5555",
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
        handshake: bool = True,
    ) -> None:
        self.name = name
        self.partner_name = partner_name
        self.role = role
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self._connected = False
        self._socket = None

        self._ctx = zmq.Context.instance()
        self._socket: zmq.Socket = self._ctx.socket(zmq.PAIR)
        self._socket.setsockopt(zmq.LINGER, 1000)  # clean shutdown

        # Set receive timeout
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)

        if role == "bind":
            self._socket.bind(endpoint)
            logger.info("[%s] Bound to %s", name, endpoint)
        elif role == "connect":
            self._socket.connect(endpoint)
            logger.info("[%s] Connected to %s", name, endpoint)
        else:
            raise ValueError(f"role must be 'bind' or 'connect', got '{role}'")

        self._connected = True

        if handshake:
            self._perform_handshake()

    # ------------------------------------------------------------------
    # Handshake
    # ------------------------------------------------------------------

    def _perform_handshake(self) -> None:
        """Exchange names with the partner to verify correct pairing.

        The ``bind`` side sends first, then receives.
        The ``connect`` side receives first, then sends.
        """
        # Temporarily increase timeout for handshake
        old_timeout = self._socket.getsockopt(zmq.RCVTIMEO)
        self._socket.setsockopt(zmq.RCVTIMEO, _HANDSHAKE_TIMEOUT_MS)

        try:
            msg_out = make_handshake_msg(self.name, self.partner_name)

            if self.role == "bind":
                # Server: send handshake, then wait for reply
                self._socket.send(msg_out)
                logger.debug("[%s] Handshake sent, waiting for reply...", self.name)
                reply = self._socket.recv()
            else:
                # Client: wait for handshake, then reply
                logger.debug("[%s] Waiting for handshake...", self.name)
                reply = self._socket.recv()
                self._socket.send(msg_out)

            # Validate the partner's handshake
            sender, expected_partner = parse_handshake_msg(reply)
            if sender != self.partner_name:
                raise CommunicationError(
                    f"[{self.name}] Handshake failed: expected partner "
                    f"'{self.partner_name}', got '{sender}'"
                )
            if expected_partner != self.name:
                raise CommunicationError(
                    f"[{self.name}] Handshake mismatch: partner expects "
                    f"'{expected_partner}', but our name is '{self.name}'"
                )
            logger.info(
                "[%s] Handshake successful with '%s'", self.name, sender
            )
        except zmq.Again:
            raise CommunicationError(
                f"[{self.name}] Handshake timed out after "
                f"{_HANDSHAKE_TIMEOUT_MS / 1000:.0f}s waiting for partner "
                f"'{self.partner_name}' at {self.endpoint}"
            )
        finally:
            self._socket.setsockopt(zmq.RCVTIMEO, old_timeout)

    # ------------------------------------------------------------------
    # Data exchange
    # ------------------------------------------------------------------

    def send_array(self, data_name: str, array: np.ndarray | "sps.csr_matrix") -> None:
        """Serialize and send a named NumPy array to the partner.

        Parameters
        ----------
        data_name : str
            Identifier for the data (e.g. ``"TemperatureField"``).
        array : np.ndarray or scipy.sparse.spmatrix
            The data to transmit.

        Raises
        ------
        CommunicationError
            If the socket is closed or the send fails.
        """
        self._check_connected()
        frames = serialize_array(data_name, array)
        self._socket.send_multipart(frames)
        logger.debug(
            "[%s] Sent '%s' — shape %s, dtype %s",
            self.name, data_name, array.shape, array.dtype,
        )

    def receive_array(self) -> tuple[str, np.ndarray | "sps.csr_matrix"]:
        """Receive a named NumPy array from the partner.

        Returns
        -------
        tuple[str, np.ndarray | sps.csr_matrix]
            ``(data_name, array)``

        Raises
        ------
        CommunicationError
            If a timeout occurs or the socket is closed.
        """
        self._check_connected()
        try:
            frames = self._socket.recv_multipart()
        except zmq.Again:
            raise CommunicationError(
                f"[{self.name}] Receive timed out after "
                f"{self.timeout_ms / 1000:.1f}s"
            )
        name, array = deserialize_array(frames)
        logger.debug(
            "[%s] Received '%s' — shape %s, dtype %s",
            self.name, name, array.shape, array.dtype,
        )
        return name, array

    # ------------------------------------------------------------------
    # Synchronization
    # ------------------------------------------------------------------

    def synchronize(self) -> None:
        """Block until both solvers reach this synchronization point.

        The ``bind`` side sends the sync signal first, then waits for ACK.
        The ``connect`` side waits for the sync signal, then sends ACK.
        """
        self._check_connected()
        try:
            if self.role == "bind":
                self._socket.send(SYNC_SIGNAL)
                ack = self._socket.recv()
                if ack != SYNC_ACK:
                    raise CommunicationError(
                        f"[{self.name}] Bad sync ACK: {ack!r}"
                    )
            else:
                sig = self._socket.recv()
                if sig != SYNC_SIGNAL:
                    raise CommunicationError(
                        f"[{self.name}] Bad sync signal: {sig!r}"
                    )
                self._socket.send(SYNC_ACK)
        except zmq.Again:
            raise CommunicationError(
                f"[{self.name}] Synchronization timed out"
            )
        logger.debug("[%s] Synchronization complete", self.name)

    # ------------------------------------------------------------------
    # Raw send / receive (for metadata exchange, CoSimIO-style Info)
    # ------------------------------------------------------------------

    def send_raw(self, data: bytes) -> None:
        """Send raw bytes to the partner."""
        self._check_connected()
        self._socket.send(data)

    def receive_raw(self) -> bytes:
        """Receive raw bytes from the partner."""
        self._check_connected()
        try:
            return self._socket.recv()
        except zmq.Again:
            raise CommunicationError(
                f"[{self.name}] Raw receive timed out"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Gracefully close the ZeroMQ socket."""
        if self._socket is not None and self._connected and not self._socket.closed:
            self._socket.close()
            self._connected = False
            logger.info("[%s] Socket closed", self.name)

    @property
    def is_connected(self) -> bool:
        """Whether the communicator socket is still open."""
        return (
            self._connected
            and self._socket is not None
            and not self._socket.closed
        )

    def _check_connected(self) -> None:
        if not self.is_connected:
            raise CommunicationError(
                f"[{self.name}] Socket is not connected"
            )

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Communicator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
