"""StatefulRVEPool — single-worker brain for path-dependent FE².

The scatter-gather socket round-robins work items with **no worker affinity**,
but J2 plasticity is **path-dependent**: each macro quadrature point owns an
RVE whose plastic history must persist across macro load increments. The
milestone-1 resolution is a single worker that holds *all* RVE states in
memory, keyed by the quadrature-point index carried in each message.

(Milestone 2 removes the single-worker limitation by shipping the RVE state in
a binary frame of the multipart message so any worker can serve any index.)
"""

from __future__ import annotations

import numpy as np

from fe2.rve_solver import RVESolver


class StatefulRVEPool:
    """Maps a macro quadrature-point index -> its persistent RVE solver.

    Parameters
    ----------
    rve_kwargs : dict
        Keyword arguments forwarded to :class:`RVESolver` for every RVE.
    """

    def __init__(self, rve_kwargs: dict) -> None:
        self._rve_kwargs = dict(rve_kwargs)
        self._rves: dict[int, RVESolver] = {}

    def solve(self, index: int, eps_bar_voigt, commit: bool = True) -> np.ndarray:
        """Advance (or trial-probe) the RVE for ``index`` by ``eps_bar_voigt``.

        Lazily creates the RVE on first encounter of ``index``. ``commit`` is
        forwarded to :meth:`RVESolver.solve` (``False`` => trial evaluation,
        state rolled back).

        Returns
        -------
        np.ndarray
            Homogenized macro stress in Voigt ``[xx, yy, xy]``.
        """
        rve = self._rves.get(index)
        if rve is None:
            rve = RVESolver(**self._rve_kwargs)
            self._rves[index] = rve
        return rve.solve(np.asarray(eps_bar_voigt, dtype=float), commit=commit)

    def get(self, index: int) -> RVESolver:
        """Return the RVE for ``index`` (raises ``KeyError`` if absent)."""
        return self._rves[index]

    @property
    def num_rves(self) -> int:
        """Number of distinct RVEs currently allocated."""
        return len(self._rves)
