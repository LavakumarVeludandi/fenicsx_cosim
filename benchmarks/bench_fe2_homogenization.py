"""Benchmark: FE2 homogenization of a homogeneous elastic RVE vs closed-form Hooke.

Validates RVE **homogenization correctness** (KUBC solve + volume averaging)
against an exact analytic answer. It does NOT exercise the FE² dispatch fabric
(``ScatterGatherCommunicator`` / ``QuadratureExtractor``) — that coupling path
needs its own strain->worker->stress round-trip benchmark (see P1/P2). For a
homogeneous linear-elastic RVE the unique KUBC solution is the affine field
``u = ε̄·x``, so the strain is constant ``= ε̄`` everywhere and the
volume-averaged (homogenized) stress must equal the plane-strain Hooke stress
``C : ε̄`` to solver precision. A degree>=1 displacement space represents the
affine field exactly, so there is no discretization error to hide behind.

Reports the relative error of ``σ̄`` against the closed form and the
Hill-Mandel energy residual (both should be ~machine zero).

Requires DOLFINx (run in the ``fenicsx-env``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from _harness import BenchmarkResult, report  # noqa: E402


def _hooke_plane_strain(E: float, nu: float, eps) -> np.ndarray:
    """Closed-form plane-strain stress for Voigt strain [εxx, εyy, γxy]."""
    exx, eyy, gxy = (float(c) for c in eps)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    sxx = lam * (exx + eyy) + 2 * mu * exx
    syy = lam * (exx + eyy) + 2 * mu * eyy
    sxy = mu * gxy
    return np.array([sxx, syy, sxy])


def run():
    from fe2.rve_solver import RVESolver

    E, nu = 70.0e3, 0.3
    eps = np.array([1.0e-3, -5.0e-4, 8.0e-4])

    # Homogeneous (incl_radius=0) and forced elastic (huge yield) => the RVE is
    # a single linear-elastic phase with an exact affine solution.
    rve = RVESolver(n=8, E_matrix=E, nu_matrix=nu,
                    sigma_y=1.0e30, H=0.0, incl_radius=0.0, degree=2)
    sigma_bar = rve.solve(eps, commit=True)
    exact = _hooke_plane_strain(E, nu, eps)

    rel = float(np.linalg.norm(sigma_bar - exact) / np.linalg.norm(exact))
    hm = float(rve.hill_mandel_residual())

    return [
        BenchmarkResult("homog elastic RVE: σ̄ vs Hooke", "rel error",
                        rel, 1.0e-7),
        BenchmarkResult("homog elastic RVE: Hill-Mandel", "energy residual",
                        hm, 1.0e-8),
    ]


if __name__ == "__main__":
    ok = report(run(), "FE2 Homogenization")
    sys.exit(0 if ok else 1)
