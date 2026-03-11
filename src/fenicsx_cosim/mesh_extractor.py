"""
MeshExtractor — FEniCSx boundary data extraction engine.

This module handles the highly specific FEniCSx v0.10+ API calls required
to isolate boundary Degrees of Freedom (DoFs) and their spatial coordinates
from a ``dolfinx.mesh.Mesh``.

It corresponds to **Phase 1** of the development roadmap and is analogous
to the ``CoSimIO::ModelPart`` concept in Kratos, but works natively with
FEniCSx data structures.

Typical usage
-------------
>>> extractor = MeshExtractor()
>>> extractor.register(mesh, facet_tags, marker_id=1, function_space=V)
>>> coords = extractor.boundary_coordinates   # (N, 3) ndarray
>>> dof_indices = extractor.boundary_dof_indices  # (N,) ndarray
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fenicsx_cosim.utils import get_logger

logger = get_logger(__name__)

# We import dolfinx lazily so the module can be imported for type-checking
# even on machines without a full DOLFINx installation.
try:
    import dolfinx
    import dolfinx.fem
    import dolfinx.mesh

    _HAS_DOLFINX = True
except ImportError:  # pragma: no cover
    _HAS_DOLFINX = False


def _require_dolfinx() -> None:
    """Raise an informative error if DOLFINx is not installed."""
    if not _HAS_DOLFINX:
        raise ImportError(
            "DOLFINx is required for MeshExtractor but could not be imported. "
            "Install it with: pip install fenics-dolfinx"
        )


@dataclass
class BoundaryData:
    """Container for extracted boundary information.

    Attributes
    ----------
    dof_indices : np.ndarray
        Global DoF indices that live on the coupling boundary.  Shape ``(N,)``.
    coordinates : np.ndarray
        Spatial coordinates of the boundary DoFs.  Shape ``(N, 3)``.
    facet_indices : np.ndarray
        Mesh facet indices that form the coupling boundary.
    marker_id : int
        The marker value used to identify the coupling boundary.
    """

    dof_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    coordinates: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    facet_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    marker_id: int = 0


class MeshExtractor:
    """Extract boundary DoFs and coordinates from a FEniCSx mesh.

    This class wraps the FEniCSx v0.10+ API to:

    1. Identify boundary facets using mesh-tags or a locator function.
    2. Find the DoFs living on those facets via topological DoF location.
    3. Extract the ``(x, y, z)`` coordinates of the boundary DoFs.

    The extracted data is stored internally and can be queried as NumPy
    arrays for serialization and transmission via the ``Communicator``.
    """

    def __init__(self) -> None:
        _require_dolfinx()
        self._boundary_data: Optional[BoundaryData] = None
        self._function_space = None
        self._mesh = None
        logger.debug("MeshExtractor created")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        mesh: "dolfinx.mesh.Mesh",
        facet_tags: "dolfinx.mesh.MeshTags",
        marker_id: int,
        function_space: "dolfinx.fem.FunctionSpace",
    ) -> BoundaryData:
        """Register a coupling boundary and extract its DoF information.

        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
            The computational mesh.
        facet_tags : dolfinx.mesh.MeshTags
            Mesh tags that label boundary facets.
        marker_id : int
            The integer marker identifying the coupling boundary within
            *facet_tags*.
        function_space : dolfinx.fem.FunctionSpace
            The function space whose DoFs on the boundary are needed.

        Returns
        -------
        BoundaryData
            A dataclass holding the extracted boundary information.
        """
        self._mesh = mesh
        self._function_space = function_space

        # 1. Find facet indices on the coupling boundary
        facet_indices = facet_tags.find(marker_id)
        logger.info(
            "Found %d facets with marker_id=%d", len(facet_indices), marker_id
        )

        # 2. Get the topological dimension (facets are dim - 1)
        tdim = mesh.topology.dim
        fdim = tdim - 1

        # Ensure connectivity is available
        mesh.topology.create_connectivity(fdim, tdim)

        # 3. Locate DoFs on the boundary facets
        dof_indices = dolfinx.fem.locate_dofs_topological(
            function_space, fdim, facet_indices
        )
        logger.info(
            "Located %d boundary DoFs on marker_id=%d", len(dof_indices), marker_id
        )

        # 4. Extract the (x, y, z) coordinates of these DoFs
        #    In DOLFINx v0.10, dof coordinates come from the function space's
        #    tabulate_dof_coordinates() method.
        dof_coords = function_space.tabulate_dof_coordinates()
        boundary_coords = dof_coords[dof_indices]

        self._boundary_data = BoundaryData(
            dof_indices=np.asarray(dof_indices, dtype=np.int64),
            coordinates=np.asarray(boundary_coords, dtype=np.float64),
            facet_indices=np.asarray(facet_indices, dtype=np.int64),
            marker_id=marker_id,
        )
        return self._boundary_data

    def register_from_locator(
        self,
        mesh: "dolfinx.mesh.Mesh",
        locator_fn,
        function_space: "dolfinx.fem.FunctionSpace",
        marker_id: int = 0,
    ) -> BoundaryData:
        """Register a coupling boundary using a geometric locator function.

        This is a convenience method for cases where no ``MeshTags`` object
        exists.  The *locator_fn* must accept an ``(3, N)`` array of
        coordinates and return a boolean mask of length ``N``.

        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
            The computational mesh.
        locator_fn : callable
            ``locator_fn(x: ndarray) -> ndarray[bool]`` — identifies
            boundary facets.
        function_space : dolfinx.fem.FunctionSpace
            The function space whose DoFs on the boundary are needed.
        marker_id : int, optional
            A label for this boundary (default ``0``).

        Returns
        -------
        BoundaryData
        """
        self._mesh = mesh
        self._function_space = function_space

        tdim = mesh.topology.dim
        fdim = tdim - 1

        # Find boundary facets using the locator
        facet_indices = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, locator_fn
        )
        logger.info(
            "Locator found %d boundary facets", len(facet_indices)
        )

        mesh.topology.create_connectivity(fdim, tdim)

        dof_indices = dolfinx.fem.locate_dofs_topological(
            function_space, fdim, facet_indices
        )
        dof_coords = function_space.tabulate_dof_coordinates()
        boundary_coords = dof_coords[dof_indices]

        self._boundary_data = BoundaryData(
            dof_indices=np.asarray(dof_indices, dtype=np.int64),
            coordinates=np.asarray(boundary_coords, dtype=np.float64),
            facet_indices=np.asarray(facet_indices, dtype=np.int64),
            marker_id=marker_id,
        )
        return self._boundary_data

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def boundary_data(self) -> BoundaryData:
        """The most recently extracted :class:`BoundaryData`."""
        if self._boundary_data is None:
            raise RuntimeError(
                "No boundary registered yet. Call register() first."
            )
        return self._boundary_data

    @property
    def boundary_coordinates(self) -> np.ndarray:
        """Spatial coordinates of boundary DoFs — shape ``(N, 3)``."""
        return self.boundary_data.coordinates

    @property
    def boundary_dof_indices(self) -> np.ndarray:
        """Global DoF indices on the coupling boundary — shape ``(N,)``."""
        return self.boundary_data.dof_indices

    def extract_boundary_values(
        self, function: "dolfinx.fem.Function"
    ) -> np.ndarray:
        """Extract the values of a FEniCSx Function at the boundary DoFs.

        Parameters
        ----------
        function : dolfinx.fem.Function
            The FEniCSx function whose boundary values are needed.

        Returns
        -------
        np.ndarray
            Values at the boundary DoFs.
        """
        bd = self.boundary_data
        values = function.x.array[bd.dof_indices]
        return np.asarray(values, dtype=np.float64)

    def inject_boundary_values(
        self,
        function: "dolfinx.fem.Function",
        values: np.ndarray,
    ) -> None:
        """Inject values into a FEniCSx Function at the boundary DoFs.

        Parameters
        ----------
        function : dolfinx.fem.Function
            The target FEniCSx function.
        values : np.ndarray
            The values to inject (must match the number of boundary DoFs).

        Raises
        ------
        ValueError
            If *values* length does not match the number of boundary DoFs.
        """
        bd = self.boundary_data
        if len(values) != len(bd.dof_indices):
            raise ValueError(
                f"Expected {len(bd.dof_indices)} values, got {len(values)}"
            )
        function.x.array[bd.dof_indices] = values
