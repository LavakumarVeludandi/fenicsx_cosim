"""
Tests for the Communicator (PyZMQ networking engine).

These tests can run without DOLFINx — they only exercise the ZeroMQ
layer using raw NumPy arrays.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from fenicsx_cosim.communicator import Communicator, CommunicationError


ENDPOINT = "tcp://127.0.0.1:15555"


def _run_server(results: dict, endpoint: str = ENDPOINT) -> None:
    """Run in a background thread as the 'bind' side."""
    try:
        comm = Communicator(
            name="ServerSolver",
            partner_name="ClientSolver",
            role="bind",
            endpoint=endpoint,
            timeout_ms=10_000,
            handshake=True,
        )
        results["server_connected"] = True

        # Send an array
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        comm.send_array("TestData", data)

        # Receive an array back
        name, received = comm.receive_array()
        results["server_received_name"] = name
        results["server_received_data"] = received

        # Synchronize
        comm.synchronize()
        results["server_synced"] = True

        comm.close()
    except Exception as e:
        results["server_error"] = str(e)


def _run_client(results: dict, endpoint: str = ENDPOINT) -> None:
    """Run in a background thread as the 'connect' side."""
    try:
        # Small delay to let server bind first
        time.sleep(0.2)
        comm = Communicator(
            name="ClientSolver",
            partner_name="ServerSolver",
            role="connect",
            endpoint=endpoint,
            timeout_ms=10_000,
            handshake=True,
        )
        results["client_connected"] = True

        # Receive the array from server
        name, received = comm.receive_array()
        results["client_received_name"] = name
        results["client_received_data"] = received

        # Send an array back
        reply = received * 2.0
        comm.send_array("ReplyData", reply)

        # Synchronize
        comm.synchronize()
        results["client_synced"] = True

        comm.close()
    except Exception as e:
        results["client_error"] = str(e)


class TestCommunicatorBasic:
    """Test basic Communicator creation and lifecycle."""

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="role must be"):
            Communicator(
                name="A", partner_name="B",
                role="invalid",
                endpoint="tcp://127.0.0.1:15556",
                handshake=False,
            )


class TestCommunicatorDataExchange:
    """Test two-way data exchange between bind and connect communicators."""

    def test_send_receive_roundtrip(self):
        """Full send/receive/sync cycle between server and client."""
        results: dict = {}
        server_thread = threading.Thread(
            target=_run_server, args=(results,),
            kwargs={"endpoint": "tcp://127.0.0.1:15557"},
        )
        client_thread = threading.Thread(
            target=_run_client, args=(results,),
            kwargs={"endpoint": "tcp://127.0.0.1:15557"},
        )

        server_thread.start()
        client_thread.start()

        server_thread.join(timeout=15)
        client_thread.join(timeout=15)

        # Check no errors
        assert "server_error" not in results, results.get("server_error")
        assert "client_error" not in results, results.get("client_error")

        # Check connections established
        assert results.get("server_connected") is True
        assert results.get("client_connected") is True

        # Check client received the correct data
        assert results["client_received_name"] == "TestData"
        np.testing.assert_array_equal(
            results["client_received_data"],
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        )

        # Check server received the reply
        assert results["server_received_name"] == "ReplyData"
        np.testing.assert_array_equal(
            results["server_received_data"],
            np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
        )

        # Check sync
        assert results.get("server_synced") is True
        assert results.get("client_synced") is True


class TestCommunicatorMultidimensional:
    """Test exchange of multi-dimensional arrays."""

    def test_2d_array_exchange(self):
        results: dict = {}
        endpoint = "tcp://127.0.0.1:15558"

        def server(res):
            try:
                comm = Communicator(
                    name="A", partner_name="B",
                    role="bind", endpoint=endpoint,
                    timeout_ms=10_000, handshake=True,
                )
                coords = np.array(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=np.float64,
                )
                comm.send_array("coords", coords)
                comm.close()
                res["ok"] = True
            except Exception as e:
                res["err"] = str(e)

        def client(res):
            try:
                time.sleep(0.2)
                comm = Communicator(
                    name="B", partner_name="A",
                    role="connect", endpoint=endpoint,
                    timeout_ms=10_000, handshake=True,
                )
                name, coords = comm.receive_array()
                res["shape"] = coords.shape
                res["data"] = coords
                comm.close()
            except Exception as e:
                res["err"] = str(e)

        srv_res: dict = {}
        cli_res: dict = {}
        t1 = threading.Thread(target=server, args=(srv_res,))
        t2 = threading.Thread(target=client, args=(cli_res,))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert "err" not in srv_res, srv_res.get("err")
        assert "err" not in cli_res, cli_res.get("err")
        assert cli_res["shape"] == (3, 3)
        np.testing.assert_array_almost_equal(
            cli_res["data"][1], [1.0, 0.0, 0.0]
        )
