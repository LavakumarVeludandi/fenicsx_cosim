"""
Utility functions for fenicsx-cosim.

Provides logging configuration, serialization helpers for NumPy arrays
over ZeroMQ, and common constants used across the package.
"""

from __future__ import annotations

import json
import logging
import struct
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a consistently formatted logger for a cosim component.

    Parameters
    ----------
    name : str
        Logger name (typically the module ``__name__``).
    level : int, optional
        Logging level, by default ``logging.INFO``.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(f"fenicsx_cosim.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Serialization helpers (NumPy ↔ bytes for ZeroMQ)
# ---------------------------------------------------------------------------

# Wire protocol:
#   Frame 0  – header  (JSON bytes):  {"name": ..., "dtype": ..., "shape": ...}
#   Frame 1  – payload (raw ndarray bytes)

def serialize_array(name: str, array: np.ndarray) -> list[bytes]:
    """Serialize a named NumPy array into a multipart ZeroMQ message.

    Parameters
    ----------
    name : str
        A human-readable identifier for the data (e.g. ``"TemperatureField"``).
    array : np.ndarray
        The data to transmit.

    Returns
    -------
    list[bytes]
        A two-frame message: ``[header_bytes, payload_bytes]``.
    """
    header: dict[str, Any] = {
        "name": name,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }
    header_bytes = json.dumps(header).encode("utf-8")
    payload_bytes = array.tobytes()
    return [header_bytes, payload_bytes]


def deserialize_array(frames: list[bytes]) -> tuple[str, np.ndarray]:
    """Reconstruct a named NumPy array from a multipart ZeroMQ message.

    Parameters
    ----------
    frames : list[bytes]
        A two-frame message as produced by :func:`serialize_array`.

    Returns
    -------
    tuple[str, np.ndarray]
        ``(name, array)`` – the identifier and reconstructed array.
    """
    header = json.loads(frames[0].decode("utf-8"))
    array = np.frombuffer(frames[1], dtype=np.dtype(header["dtype"]))
    array = array.reshape(header["shape"])
    return header["name"], array


# ---------------------------------------------------------------------------
# Handshake message helpers
# ---------------------------------------------------------------------------

# The handshake is a simple name exchange so both sides can verify their
# partner before proceeding.

HANDSHAKE_MAGIC = b"FCOSIM01"


def make_handshake_msg(my_name: str, partner_name: str) -> bytes:
    """Build a handshake message.

    Format: ``MAGIC | len(my_name) | my_name | len(partner_name) | partner_name``
    """
    my_enc = my_name.encode("utf-8")
    partner_enc = partner_name.encode("utf-8")
    return (
        HANDSHAKE_MAGIC
        + struct.pack("!I", len(my_enc))
        + my_enc
        + struct.pack("!I", len(partner_enc))
        + partner_enc
    )


def parse_handshake_msg(data: bytes) -> tuple[str, str]:
    """Parse a handshake message.

    Returns
    -------
    tuple[str, str]
        ``(sender_name, expected_partner_name)``

    Raises
    ------
    ValueError
        If the magic bytes do not match.
    """
    magic_len = len(HANDSHAKE_MAGIC)
    if data[:magic_len] != HANDSHAKE_MAGIC:
        raise ValueError("Invalid handshake: bad magic bytes")
    offset = magic_len

    (name_len,) = struct.unpack("!I", data[offset : offset + 4])
    offset += 4
    sender_name = data[offset : offset + name_len].decode("utf-8")
    offset += name_len

    (partner_len,) = struct.unpack("!I", data[offset : offset + 4])
    offset += 4
    partner_name = data[offset : offset + partner_len].decode("utf-8")

    return sender_name, partner_name


# ---------------------------------------------------------------------------
# Synchronization signals
# ---------------------------------------------------------------------------

SYNC_SIGNAL = b"COSIM_SYNC"
SYNC_ACK = b"COSIM_SYNC_ACK"
