"""Benchmark: implicit-coupling accelerators on an added-mass-like fixed point.

Validates the strong-coupling accelerators (``Aitken``, ``IQNILS``) against an
exact analytic answer, and demonstrates the failure they exist to fix.

The coupled step is the affine fixed-point map ``S(x) = A x + b`` whose unique
fixed point is ``x* = (I - A)^{-1} b``. Choosing the spectral radius of ``A``
above 1 reproduces the added-mass instability: plain Gauss-Seidel iteration
(``x_{k+1} = S(x_k)``) **diverges**, while Aitken and IQN-ILS converge — IQN-ILS
in at most ``n`` sub-iterations on a linear problem.

Pure NumPy — runs anywhere (gated in CI).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from _harness import BenchmarkResult, report  # noqa: E402
from fenicsx_cosim.convergence import (  # noqa: E402
    Aitken, IQNILS, fixed_point_iterate,
)


def _affine(A, b):
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float)
    xstar = np.linalg.solve(np.eye(len(b)) - A, b)
    return (lambda x: A @ x + b), xstar


def run():
    results = []

    # --- scalar added-mass analogue: a = -1.5 => |a| > 1, GS diverges -------
    S1, xs1 = _affine([[-1.5]], [1.0])
    x0 = np.array([0.0])

    gs, gs_hist = fixed_point_iterate(S1, x0, None, tol=1e-10, max_iter=40)
    gs_ratio = gs_hist[0] / gs_hist[-1] if gs_hist[-1] > 0 else np.inf
    results.append(BenchmarkResult(
        "Gauss-Seidel (no accel): diverges", "init/final res (<0.1 ⇒ diverged)",
        float(gs_ratio), 0.1))

    a_x, a_hist = fixed_point_iterate(S1, x0, Aitken(omega0=0.5),
                                      tol=1e-10, max_iter=50)
    results.append(BenchmarkResult(
        "Aitken scalar: ‖x - x*‖", "abs error",
        float(np.linalg.norm(a_x - xs1)), 1e-8))
    results.append(BenchmarkResult(
        "Aitken scalar: final residual", "residual",
        a_hist[-1], 1e-8))

    # --- vector coupled problem, spectral radius > 1 ------------------------
    A = np.array([[0.0, 1.3, 0.2],
                  [0.4, 0.0, 1.1],
                  [0.3, 0.5, 0.0]])
    b = np.array([1.0, -2.0, 0.5])
    S3, xs3 = _affine(A, b)
    rho = max(abs(np.linalg.eigvals(A)))
    x0v = np.zeros(3)

    q_x, q_hist = fixed_point_iterate(S3, x0v, IQNILS(omega0=0.1),
                                      tol=1e-10, max_iter=50)
    results.append(BenchmarkResult(
        "IQN-ILS vector: ‖x - x*‖", "abs error",
        float(np.linalg.norm(q_x - xs3)), 1e-7,
        extra={"spectral_radius": float(rho)}))
    results.append(BenchmarkResult(
        "IQN-ILS vector: sub-iterations", "count (≤ n+2 = 5)",
        float(len(q_hist)), 5.0))

    return results


if __name__ == "__main__":
    ok = report(run(), "Implicit Coupling")
    sys.exit(0 if ok else 1)
