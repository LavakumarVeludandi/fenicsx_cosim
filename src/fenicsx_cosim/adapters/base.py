"""
SolverAdapter — Abstract base class for multi-solver coupling.

Every external solver that participates in a fenicsx-cosim coupling must
provide a concrete :class:`SolverAdapter`.  The adapter's job is to
translate the solver's native data structures (e.g. Kratos ``ModelPart``,
Abaqus ODB, MOOSE variables) into plain NumPy arrays that the
``Communicator`` can serialize and transmit over ZeroMQ.

This is the "last mile" abstraction — everything before it (ZMQ transport,
nearest-neighbour mapping, synchronization) is solver-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class SolverAdapter(ABC):
    """Bridge between any solver's native data structures and fenicsx-cosim.

    Subclasses must implement the four abstract methods below.  The
    adapter is then passed to :meth:`CouplingInterface.from_adapter`
    (or used standalone with a raw ``Communicator``).

    Design principles
    -----------------
    * **No FEniCSx dependency** — adapters for external solvers must not
      ``import dolfinx``.  Only the ``FEniCSxAdapter`` needs DOLFINx.
    * **NumPy is the lingua franca** — all data is exchanged as NumPy
      arrays (coordinates as ``(N, 3)`` float64, scalar fields as
      ``(N,)`` float64, vector fields as ``(N, dim)`` float64).
    * **Thin wrappers** — adapters should delegate to solver APIs, not
      re-implement solver logic.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_boundary_coordinates(self) -> np.ndarray:
        """Return coordinates of coupling boundary nodes.

        Returns
        -------
        np.ndarray
            Shape ``(N, 3)``, dtype ``float64``.
            Each row is ``(x, y, z)`` for a boundary node.
        """

    @abstractmethod
    def extract_field(self, field_name: str) -> np.ndarray:
        """Extract a field's values at coupling boundary nodes.

        Parameters
        ----------
        field_name : str
            Solver-native field identifier (e.g. ``"TEMPERATURE"``).

        Returns
        -------
        np.ndarray
            Shape ``(N,)`` for scalar or ``(N, dim)`` for vector fields.
        """

    @abstractmethod
    def inject_field(self, field_name: str, values: np.ndarray) -> None:
        """Set field values at coupling boundary nodes.

        Parameters
        ----------
        field_name : str
            Solver-native field identifier.
        values : np.ndarray
            Same shape convention as :meth:`extract_field`.
        """

    @abstractmethod
    def advance(self) -> None:
        """Advance the solver by one time step.

        This is called by ``CouplingInterface.advance_in_time()`` after
        data exchange is complete.  Implement as a no-op if the solver's
        time-stepping is managed externally.
        """

    # ------------------------------------------------------------------
    # Optional hooks (concrete defaults provided)
    # ------------------------------------------------------------------

    def get_field_names(self) -> list[str]:
        """Return a list of available field names for export.

        Override to advertise which fields the solver can provide.  The
        default returns an empty list (metadata exchange is optional).
        """
        return []

    def get_metadata(self) -> dict[str, Any]:
        """Return solver metadata (mesh info, solver name, etc.).

        Override to provide metadata that the partner solver can inspect.
        """
        return {}
