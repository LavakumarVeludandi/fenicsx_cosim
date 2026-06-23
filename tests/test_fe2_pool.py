"""Tests for StatefulRVEPool — the single-worker brain for FE².

The pool keeps one independent, path-dependent RVE per macro quadrature-point
index. This is what lets a single worker correctly serve a path-dependent FE²
problem over the round-robin scatter-gather socket (which has no worker
affinity): all RVE state lives in one process, keyed by the QP index carried
in the message.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

pytest.importorskip("dolfinx")

from fe2.rve_solver import RVESolver  # noqa: E402
from fe2.stateful_pool import StatefulRVEPool  # noqa: E402


def _material():
    return dict(
        n=8, E_matrix=70.0e3, nu_matrix=0.3, sigma_y=200.0, H=500.0,
        E_incl=210.0e3, nu_incl=0.2, incl_radius=0.25,
    )


@pytest.mark.slow
@pytest.mark.integration
def test_pool_keeps_independent_state_per_index():
    """Two QP indices must not share plastic history."""
    pool = StatefulRVEPool(rve_kwargs=_material())

    # Index 0 is driven into plasticity; index 1 only ever sees zero strain.
    peak = np.array([0.0, 0.0, 8.0e-3])
    for k in range(1, 9):
        pool.solve(0, peak * k / 8)
    pool.solve(1, np.array([0.0, 0.0, 0.0]))

    # Unload index 0 -> residual (it yielded); index 1 stays stress-free.
    s0 = pool.solve(0, np.array([0.0, 0.0, 0.0]))
    s1 = pool.solve(1, np.array([0.0, 0.0, 0.0]))

    assert abs(s0[2]) > 1.0           # index 0 carries residual stress
    assert np.allclose(s1, 0.0, atol=1e-6)  # index 1 untouched


@pytest.mark.slow
@pytest.mark.integration
def test_pool_lazily_creates_one_rve_per_index():
    """A new index allocates a fresh RVE; a repeated index reuses it."""
    pool = StatefulRVEPool(rve_kwargs=_material())
    pool.solve(5, np.array([1.0e-4, 0.0, 0.0]))
    assert pool.num_rves == 1
    pool.solve(5, np.array([2.0e-4, 0.0, 0.0]))
    assert pool.num_rves == 1
    pool.solve(7, np.array([1.0e-4, 0.0, 0.0]))
    assert pool.num_rves == 2
    assert isinstance(pool.get(5), RVESolver)
