"""Behaviour test for the macro FE² driver (in-process, no ZeroMQ).

The macro driver is decoupled from transport: it calls a ``stress_callback``
that maps a list of per-cell macro strains to per-cell homogenized stresses.
Here we inject a :class:`StatefulRVEPool` directly, so the *real* two-scale
coupling runs in one process and is fast to test.

Falsifiable behaviour: under identical displacement control, a yielding micro
material produces a **softer** macro response (lower average stress) than an
effectively-elastic micro material — the multiscale feedback genuinely
changes the macro solution.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

pytest.importorskip("dolfinx")

from fe2.macro_driver import MacroFE2Problem  # noqa: E402
from fe2.stateful_pool import StatefulRVEPool  # noqa: E402


def _pool(sigma_y):
    return StatefulRVEPool(rve_kwargs=dict(
        n=4, E_matrix=70.0e3, nu_matrix=0.3, sigma_y=sigma_y, H=500.0,
        E_incl=210.0e3, nu_incl=0.2, incl_radius=0.25,
    ))


@pytest.mark.slow
@pytest.mark.integration
def test_macro_fe2_yielding_softer_than_elastic():
    E_eff, nu_eff = 85.0e3, 0.3  # approximate macro elastic tangent
    applied = 4.0e-3              # large enough to yield the soft micro
    n_steps = 3

    # Yielding micro (low yield stress).
    macro_y = MacroFE2Problem(n=2, E_eff=E_eff, nu_eff=nu_eff,
                              applied_strain_xx=applied, n_steps=n_steps)
    pool_y = _pool(sigma_y=150.0)
    sxx_yield = macro_y.solve(
        lambda strains, commit: [pool_y.solve(i, e, commit)
                                 for i, e in enumerate(strains)])

    # Effectively-elastic micro (huge yield stress), same macro problem.
    macro_e = MacroFE2Problem(n=2, E_eff=E_eff, nu_eff=nu_eff,
                              applied_strain_xx=applied, n_steps=n_steps)
    pool_e = _pool(sigma_y=1.0e30)
    sxx_elastic = macro_e.solve(
        lambda strains, commit: [pool_e.solve(i, e, commit)
                                 for i, e in enumerate(strains)])

    # Micro plasticity must have engaged...
    assert max(float(r.p.x.array.max()) for r in pool_y._rves.values()) > 1e-5
    # ...and softened the macro response.
    assert sxx_yield < 0.97 * sxx_elastic
    assert sxx_yield > 0.0


@pytest.mark.slow
@pytest.mark.integration
def test_macro_fe2_converges():
    """The modified-Newton FE² loop reports convergence each load step."""
    macro = MacroFE2Problem(n=2, E_eff=75.0e3, nu_eff=0.3,
                            applied_strain_xx=1.0e-3, n_steps=2)
    pool = _pool(sigma_y=1.0e30)  # elastic => should converge fast
    macro.solve(lambda strains, commit: [pool.solve(i, e, commit)
                                         for i, e in enumerate(strains)])
    assert all(macro.converged_flags)
