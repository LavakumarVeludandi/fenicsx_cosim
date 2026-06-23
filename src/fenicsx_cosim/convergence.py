"""Implicit (strong) coupling acceleration for partitioned fixed-point iteration.

Explicit / Gauss-Seidel partitioned coupling (run each solver once per step)
**diverges** for strongly coupled problems — the classic example is
incompressible FSI with a light, slender structure (the *added-mass effect*),
where the interface fixed-point map has spectral radius > 1. Strong coupling
sub-iterates within a time step until the interface residual vanishes, and needs
*acceleration* to converge in a useful number of iterations.

Convention
----------
The coupled step is a fixed-point map ``x̃ = S(x)`` (run solver A, map, run
solver B, map back → new interface value ``x̃``). The fixed point ``x*`` solves
``S(x*) = x*``. The interface residual is ``r = x̃ - x``. Each accelerator turns
the pair ``(x, x̃)`` at the current sub-iteration into the next input ``x``.

Implemented
-----------
* :class:`Aitken` — Irons & Tuck dynamic relaxation (scalar, adaptive ``ω``).
  Cheap, no history matrices; effective for mildly-coupled problems.
* :class:`IQNILS` — Degroote et al. (2009) interface quasi-Newton with an
  inverse-least-squares Jacobian from the residual/output history. On a linear
  coupled problem it converges in at most ``n`` sub-iterations.

References
----------
- B. Irons, R. Tuck, *A version of the Aitken accelerator for computer iteration*,
  Int. J. Numer. Methods Eng. (1969).
- J. Degroote, K.-J. Bathe, J. Vierendeels, *Performance of a new partitioned
  procedure versus a monolithic procedure in fluid–structure interaction*,
  Computers & Structures (2009) — IQN-ILS.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class Aitken:
    """Aitken (Irons-Tuck) dynamic under-relaxation.

    Parameters
    ----------
    omega0 : float
        Initial relaxation factor used on the first sub-iteration of each step.
    omega_max : float
        Clamp on |ω| to avoid runaway factors on near-orthogonal residuals.
    """

    def __init__(self, omega0: float = 0.1, omega_max: float = 2.0) -> None:
        self.omega0 = omega0
        self.omega_max = omega_max
        self.reset()

    def reset(self) -> None:
        """Forget history — call at the start of every new time step."""
        self._r_prev: Optional[np.ndarray] = None
        self._omega: Optional[float] = None
        self.last_residual_norm: float = np.inf

    def step(self, x_in: np.ndarray, x_out: np.ndarray) -> np.ndarray:
        """Return the next interface input from ``(x_in, x_out=S(x_in))``."""
        x_in = np.asarray(x_in, dtype=float)
        x_out = np.asarray(x_out, dtype=float)
        r = x_out - x_in
        self.last_residual_norm = float(np.linalg.norm(r))

        if self._r_prev is None:
            self._omega = self.omega0
        else:
            dr = r - self._r_prev
            denom = float(dr @ dr)
            if denom <= 1e-30:
                # Residual unchanged → already converged; keep ω.
                pass
            else:
                self._omega = -self._omega * float(self._r_prev @ dr) / denom
                self._omega = float(
                    np.clip(self._omega, -self.omega_max, self.omega_max)
                )

        self._r_prev = r
        return x_in + self._omega * r


class IQNILS:
    """Interface Quasi-Newton with Inverse Least-Squares (IQN-ILS).

    Builds an approximate inverse interface Jacobian from the history of
    residual and output differences within the current time step, and uses it to
    take a quasi-Newton interface step.

    Parameters
    ----------
    omega0 : float
        Relaxation factor for the very first sub-iteration (no history yet).
    filter_eps : float
        Relative threshold for QR-based filtering of near-collinear columns
        (drops information-poor history that would ill-condition the solve).
    max_cols : int, optional
        Cap on retained history columns (default: unlimited within a step).
    """

    def __init__(self, omega0: float = 0.1, filter_eps: float = 1e-10,
                 max_cols: Optional[int] = None) -> None:
        self.omega0 = omega0
        self.filter_eps = filter_eps
        self.max_cols = max_cols
        self.reset()

    def reset(self) -> None:
        """Forget history — call at the start of every new time step."""
        self._r_prev: Optional[np.ndarray] = None
        self._xt_prev: Optional[np.ndarray] = None
        self._V: list[np.ndarray] = []   # columns Δr_i (residual differences)
        self._W: list[np.ndarray] = []   # columns Δx̃_i (output differences)
        self.last_residual_norm: float = np.inf

    def step(self, x_in: np.ndarray, x_out: np.ndarray) -> np.ndarray:
        """Return the next interface input from ``(x_in, x_out=S(x_in))``."""
        x_in = np.asarray(x_in, dtype=float)
        x_out = np.asarray(x_out, dtype=float)
        r = x_out - x_in
        self.last_residual_norm = float(np.linalg.norm(r))

        if self._r_prev is not None:
            # Prepend most-recent difference (newest information first).
            self._V.insert(0, r - self._r_prev)
            self._W.insert(0, x_out - self._xt_prev)
            if self.max_cols is not None:
                self._V = self._V[: self.max_cols]
                self._W = self._W[: self.max_cols]

        self._r_prev = r
        self._xt_prev = x_out

        if not self._V:
            # First sub-iteration of the step → plain relaxation.
            return x_in + self.omega0 * r

        V = np.column_stack(self._V)
        W = np.column_stack(self._W)

        # Filter near-collinear columns via QR on V (Degroote-style).
        Q, R = np.linalg.qr(V)
        diag = np.abs(np.diag(R))
        keep = diag > self.filter_eps * (diag.max() if diag.size else 1.0)
        if not np.all(keep):
            V = V[:, keep]
            W = W[:, keep]
            Q, R = np.linalg.qr(V)

        # Solve V c ≈ -r  (least squares); update x_{k+1} = x̃_k + W c.
        c, *_ = np.linalg.lstsq(V, -r, rcond=None)
        return x_out + W @ c


def fixed_point_iterate(
    S: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    accelerator,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> tuple[np.ndarray, list[float]]:
    """Drive a coupled fixed-point ``S`` to convergence with an *accelerator*.

    This is the reference sub-iteration loop a strong-coupling time step runs.
    ``accelerator`` is :class:`Aitken`, :class:`IQNILS`, or ``None`` for plain
    Gauss-Seidel (``x_{k+1} = S(x_k)``).

    Returns
    -------
    (x, residual_history)
    """
    if accelerator is not None:
        accelerator.reset()
    x = np.asarray(x0, dtype=float)
    history: list[float] = []
    for _ in range(max_iter):
        x_tilde = np.asarray(S(x), dtype=float)
        res = float(np.linalg.norm(x_tilde - x))
        history.append(res)
        if res < tol:
            return x_tilde, history
        if accelerator is None:
            x = x_tilde
        else:
            x = accelerator.step(x, x_tilde)
    return x, history
