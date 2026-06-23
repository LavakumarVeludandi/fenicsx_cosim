"""N>2 broker coupling: three participants exchange through one ROUTER hub.

Validates the routing fabric the PAIR communicator cannot provide: named
peer-to-peer delivery among 3 participants, the join barrier (so no message is
dropped to an unregistered identity), and the N-way barrier.
"""

import threading

import numpy as np
import pytest

pytest.importorskip("zmq")

from fenicsx_cosim.broker_communicator import (  # noqa: E402
    BrokerClient, CouplingBroker,
)

_EP_BIND = "tcp://*:5571"
_EP_CONN = "tcp://localhost:5571"
_RING = {"A": "B", "B": "C", "C": "A"}  # name -> destination it sends to
_PAYLOAD = {"A": np.array([1.0, 2.0]),
            "B": np.array([3.0, 4.0]),
            "C": np.array([5.0, 6.0])}


def _participant(name: str, results: dict) -> None:
    client = BrokerClient(name, _EP_CONN, timeout_ms=15_000)
    try:
        client.register()
        client.send(_RING[name], "field", _PAYLOAD[name])
        results[name] = client.receive()  # (src, data_name, array)
        client.barrier()
    finally:
        client.close()


def test_three_way_ring_routing():
    broker = CouplingBroker(_EP_BIND, expected=3).start()
    results: dict = {}
    threads = [threading.Thread(target=_participant, args=(n, results))
               for n in ("A", "B", "C")]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
    finally:
        broker.stop()

    assert set(results) == {"A", "B", "C"}, "a participant failed to complete"
    # B receives A's payload, C receives B's, A receives C's.
    for receiver, sender in (("B", "A"), ("C", "B"), ("A", "C")):
        src, dname, arr = results[receiver]
        assert src == sender
        assert dname == "field"
        assert np.allclose(arr, _PAYLOAD[sender])


def _ci_participant(name: str, results: dict) -> None:
    from fenicsx_cosim import CouplingInterface

    ci = CouplingInterface(name=name, topology="broker", endpoint=_EP_CONN,
                           timeout_ms=15_000)
    try:
        ci.register_broker()
        ci.send_to(_RING[name], "field", _PAYLOAD[name])
        results[name] = ci.receive_from()
        ci.barrier()
    finally:
        ci.disconnect()


def test_broker_through_coupling_interface():
    """Same N=3 ring, driven through the CouplingInterface broker wrap."""
    broker = CouplingBroker(_EP_BIND, expected=3).start()
    results: dict = {}
    threads = [threading.Thread(target=_ci_participant, args=(n, results))
               for n in ("A", "B", "C")]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
    finally:
        broker.stop()

    assert set(results) == {"A", "B", "C"}
    for receiver, sender in (("B", "A"), ("C", "B"), ("A", "C")):
        src, dname, arr = results[receiver]
        assert src == sender and dname == "field"
        assert np.allclose(arr, _PAYLOAD[sender])
