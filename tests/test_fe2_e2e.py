"""End-to-end FE² scatter-gather test (real ZeroMQ, subprocess worker).

Validates the only seam not covered by the in-process tests: a real
``micro_worker`` process pulling strains over the socket, solving genuine J2
RVEs through :class:`StatefulRVEPool`, and pushing homogenized stresses back —
including that per-index RVE state *persists across messages* (path
dependence), which is the whole point of the single-stateful-worker design.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

pytest.importorskip("dolfinx")
pytest.importorskip("zmq")

from fenicsx_cosim.scatter_gather_communicator import (  # noqa: E402
    ScatterGatherCommunicator,
)


@pytest.mark.slow
@pytest.mark.integration
def test_scatter_gather_rve_roundtrip_is_path_dependent():
    env = dict(os.environ, PYTHONPATH=str(_EXAMPLES), FE2_RVE_N="4")
    worker = subprocess.Popen(
        [sys.executable, "-m", "fe2.micro_worker", "localhost"],
        cwd=str(_EXAMPLES), env=env,
    )
    sg = None
    try:
        sg = ScatterGatherCommunicator(
            role="master",
            push_endpoint="tcp://*:5556",
            pull_endpoint="tcp://*:5557",
            timeout_ms=120_000,
        )
        sg.slow_start(1.5)

        n = 3
        # Round 1: pure-shear loading past yield, committed.
        load = [np.array([0.0, 0.0, 8.0e-3]) for _ in range(n)]
        sg.scatter(load, [{"commit": True}] * n)
        loaded = sg.gather(n, ordered=True)

        # Round 2: unload to zero macro strain, committed. A correct stateful
        # worker returns a *nonzero residual* (it remembers each RVE yielded).
        unload = [np.zeros(3) for _ in range(n)]
        sg.scatter(unload, [{"commit": True}] * n)
        unloaded = sg.gather(n, ordered=True)

        sg.broadcast_shutdown(n_workers=1)

        assert all(abs(s[2]) > 1.0 for s in loaded)     # yielded under load
        assert all(abs(s[2]) > 1.0 for s in unloaded)   # residual => stateful
    finally:
        if sg is not None:
            sg.close()
        worker.wait(timeout=60)
        assert worker.returncode == 0
