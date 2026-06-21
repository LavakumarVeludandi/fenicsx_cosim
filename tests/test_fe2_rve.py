"""Tests for the self-contained FE² RVE solver (examples/fe2/rve_solver.py).

These exercise the *real physics* that replaces the placeholder closed-form
damage formula in the old fe2 examples:

1. Homogeneous elastic RVE under KUBC reproduces the exact plane-strain
   Hooke law (uniform field => exact volume average).
2. Hill-Mandel macrohomogeneity holds for a two-phase elastic RVE.
3. J2 plasticity is genuinely path-dependent: load past yield then unload
   leaves a residual macro stress (the property that justifies FE²).

Requires the FEniCSx stack; marked slow + integration.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the self-contained example package importable.
_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

pytest.importorskip("dolfinx")

from fe2.rve_solver import RVESolver  # noqa: E402


def _plane_strain_C(E, nu):
    """Plane-strain stiffness in Voigt [xx, yy, xy] with engineering shear."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return np.array(
        [[lam + 2 * mu, lam, 0.0],
         [lam, lam + 2 * mu, 0.0],
         [0.0, 0.0, mu]]
    )


@pytest.mark.slow
@pytest.mark.integration
def test_homogeneous_elastic_matches_hooke():
    """Homogeneous RVE under KUBC => exact plane-strain Hooke response."""
    E, nu = 70.0e3, 0.3
    rve = RVESolver(
        n=6,
        E_matrix=E, nu_matrix=nu, sigma_y=1.0e9, H=0.0,  # yield huge => elastic
        E_incl=E, nu_incl=nu, incl_radius=0.0,           # no inclusion
    )
    eps_bar = np.array([1.0e-4, -0.5e-4, 0.3e-4])  # Voigt, engineering shear
    sigma_bar = rve.solve(eps_bar)

    expected = _plane_strain_C(E, nu) @ eps_bar
    np.testing.assert_allclose(sigma_bar, expected, rtol=1e-8, atol=1e-6)


@pytest.mark.slow
@pytest.mark.integration
def test_hill_mandel_two_phase_elastic():
    """Two-phase elastic RVE satisfies Hill-Mandel: <sigma:eps> = sigma_bar:eps_bar."""
    rve = RVESolver(
        n=12,
        E_matrix=70.0e3, nu_matrix=0.3, sigma_y=1.0e9, H=0.0,
        E_incl=210.0e3, nu_incl=0.2, incl_radius=0.25,
    )
    rve.solve(np.array([1.0e-4, 0.0, 0.0]))
    assert rve.hill_mandel_residual() < 1e-6


@pytest.mark.slow
@pytest.mark.integration
def test_j2_is_path_dependent():
    """Pure-shear loading past yield then unloading leaves residual stress.

    Pure shear is fully deviatoric, so the von Mises criterion engages
    cleanly (unlike confined uniaxial *strain*). This verifies the property
    that justifies FE²: the homogenized response is path-dependent, so it
    cannot be replaced by a precomputed ``C_eff``.
    """
    E, nu = 70.0e3, 0.3
    mu = E / (2 * (1 + nu))
    rve = RVESolver(
        n=10,
        E_matrix=E, nu_matrix=nu, sigma_y=200.0, H=500.0,
        E_incl=210.0e3, nu_incl=0.2, incl_radius=0.25,
    )
    n_steps = 8
    gamma_peak = 8.0e-3                       # engineering shear
    peak = np.array([0.0, 0.0, gamma_peak])
    for k in range(1, n_steps + 1):
        sigma_load = rve.solve(peak * k / n_steps)

    # Genuine plastic flow occurred (accumulated plastic strain > 0)...
    assert float(rve.p.x.array.max()) > 1e-4
    # ...and yielding caps macro shear below the elastic prediction σxy = μ·γ.
    elastic_shear = mu * gamma_peak
    assert sigma_load[2] < 0.8 * elastic_shear

    # Incrementally unload to zero macro strain.
    for k in range(n_steps - 1, -1, -1):
        sigma_unload = rve.solve(peak * k / n_steps)

    # Path dependence: nonzero residual stress at zero macro strain.
    assert abs(sigma_unload[2]) > 1.0
