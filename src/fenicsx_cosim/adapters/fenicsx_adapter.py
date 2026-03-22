"""
FEniCSxAdapter — wraps MeshExtractor in the SolverAdapter interface.

This adapter allows existing FEniCSx-based workflows to use the new
unified adapter API.  ``CouplingInterface`` uses this internally when
``register_interface()`` is called.

It also serves as the canonical reference implementation for writing new
adapters for other solvers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from fenicsx_cosim.adapters.base import SolverAdapter
from fenicsx_cosim.mesh_extractor import MeshExtractor
from fenicsx_cosim.utils import get_logger

logger = get_logger(__name__)

try:
    import dolfinx
    import dolfinx.fem
    import dolfinx.mesh
    _HAS_DOLFINX = True
except ImportError:
    _HAS_DOLFINX = False


class FEniCSxAdapter(SolverAdapter):
    """Adapter that bridges FEniCSx data structures to fenicsx-cosim.

    Wraps the existing :class:`~fenicsx_cosim.mesh_extractor.MeshExtractor`
    to provide the standardised :class:`SolverAdapter` interface.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The computational mesh.
    facet_tags : dolfinx.mesh.MeshTags, optional
        Mesh tags labelling boundary facets.
    marker_id : int, optional
        The coupling boundary marker. Default 1.
    function_space : dolfinx.fem.FunctionSpace, optional
        If ``None``, a Lagrange-1 scalar space is created.
    locator_fn : callable, optional
        Alternative to *facet_tags*: a geometric locator function.
    """

    def __init__(
        self,
        mesh: "dolfinx.mesh.Mesh",
        facet_tags: Optional["dolfinx.mesh.MeshTags"] = None,
        marker_id: int = 1,
        function_space: Optional["dolfinx.fem.FunctionSpace"] = None,
        locator_fn=None,
    ) -> None:
        if not _HAS_DOLFINX:
            raise ImportError(
                "DOLFINx is required for FEniCSxAdapter. "
                "Install with: pip install fenicsx-cosim[fenicsx]"
            )

        self._mesh = mesh
        self._extractor = MeshExtractor()
        self._functions: dict[str, "dolfinx.fem.Function"] = {}

        # Default function space
        if function_space is None:
            function_space = dolfinx.fem.functionspace(
                mesh, ("Lagrange", 1)
            )
        self._function_space = function_space

        # Register boundary
        if locator_fn is not None:
            self._extractor.register_from_locator(
                mesh, locator_fn, function_space, marker_id
            )
        elif facet_tags is not None:
            self._extractor.register(
                mesh, facet_tags, marker_id, function_space
            )
        else:
            raise ValueError(
                "Either facet_tags or locator_fn must be provided"
            )

        logger.info(
            "FEniCSxAdapter: registered %d boundary DoFs",
            len(self._extractor.boundary_dof_indices),
        )

    # ------------------------------------------------------------------
    # SolverAdapter interface
    # ------------------------------------------------------------------

    def get_boundary_coordinates(self) -> np.ndarray:
        """Return boundary DoF coordinates — shape ``(N, 3)``."""
        return self._extractor.boundary_coordinates

    def extract_field(self, field_name: str) -> np.ndarray:
        """Extract boundary values from a registered FEniCSx Function.

        The *field_name* is used to look up the function in the internal
        registry (set via :meth:`register_function`).
        """
        if field_name not in self._functions:
            raise KeyError(
                f"No function registered for field '{field_name}'. "
                f"Call register_function('{field_name}', fn) first."
            )
        return self._extractor.extract_boundary_values(
            self._functions[field_name]
        )

    def inject_field(self, field_name: str, values: np.ndarray) -> None:
        """Inject values into a registered FEniCSx Function at boundary DoFs."""
        if field_name not in self._functions:
            raise KeyError(
                f"No function registered for field '{field_name}'. "
                f"Call register_function('{field_name}', fn) first."
            )
        self._extractor.inject_boundary_values(
            self._functions[field_name], values
        )

    def advance(self) -> None:
        """No-op — FEniCSx time-stepping is managed by the user script."""
        pass

    # ------------------------------------------------------------------
    # FEniCSx-specific helpers
    # ------------------------------------------------------------------

    def register_function(
        self,
        field_name: str,
        function: "dolfinx.fem.Function",
    ) -> None:
        """Register a FEniCSx Function for a named field.

        Parameters
        ----------
        field_name : str
            Name to associate with this function (e.g. ``"TEMPERATURE"``).
        function : dolfinx.fem.Function
            The FEniCSx function.
        """
        self._functions[field_name] = function
        logger.debug("Registered function for field '%s'", field_name)

    def get_field_names(self) -> list[str]:
        """Return names of all registered fields."""
        return list(self._functions.keys())

    @property
    def extractor(self) -> MeshExtractor:
        """The underlying MeshExtractor."""
        return self._extractor

    @property
    def function_space(self) -> "dolfinx.fem.FunctionSpace":
        """The function space used for boundary extraction."""
        return self._function_space

    def extract_boundary_values(
        self, function: "dolfinx.fem.Function"
    ) -> np.ndarray:
        """Direct passthrough to MeshExtractor (for backward compat)."""
        return self._extractor.extract_boundary_values(function)

    def inject_boundary_values(
        self,
        function: "dolfinx.fem.Function",
        values: np.ndarray,
    ) -> None:
        """Direct passthrough to MeshExtractor (for backward compat)."""
        self._extractor.inject_boundary_values(function, values)
