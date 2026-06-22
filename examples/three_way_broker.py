"""Three-way (N>2) coupling demo over the ROUTER/DEALER broker.

A self-contained illustration of coupling more than two participants — the case
the 1-to-1 PAIR `Communicator` cannot handle. Runs the broker and three named
participants (``Fluid``, ``Thermal``, ``Structure``) as threads in one process
for clarity; in a real run each participant is its own process/solver and only
the endpoint is shared.

    python examples/three_way_broker.py

Each participant sends one field to the next in a ring and receives one from the
previous, then hits a global barrier — the minimal pattern a 3-field staggered
coupling iteration is built from.
"""

from __future__ import annotations

import threading

import numpy as np

from fenicsx_cosim.broker_communicator import BrokerClient, CouplingBroker

ENDPOINT_BIND = "tcp://*:5562"
ENDPOINT_CONN = "tcp://localhost:5562"
RING = {"Fluid": "Thermal", "Thermal": "Structure", "Structure": "Fluid"}
TAG = {"Fluid": 1.0, "Thermal": 2.0, "Structure": 3.0}  # distinct demo payloads


def participant(name: str) -> None:
    client = BrokerClient(name, ENDPOINT_CONN, timeout_ms=15_000)
    try:
        client.register()  # blocks until all three have joined
        client.send(RING[name], "interface_field", np.full(3, TAG[name]))
        src, field, values = client.receive()
        print(f"[{name}] received '{field}' from {src}: {values}")
        client.barrier()
        print(f"[{name}] passed barrier")
    finally:
        client.close()


def main() -> None:
    broker = CouplingBroker(ENDPOINT_BIND, expected=3).start()
    threads = [threading.Thread(target=participant, args=(n,)) for n in RING]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    broker.stop()
    print("three-way coupling round complete")


if __name__ == "__main__":
    main()
