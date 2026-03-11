"""
QuadratureExtractor — Integration-point data extraction for FE² homogenization.

In multiscale FE² homogenization, data is **not** exchanged at the mesh
boundaries but at the *integration (Gauss) points* of every element.  The
macroscopic solver computes a strain tensor at each integration point and
dispatches it to a Representative Volume Element (RVE) solver.  The RVE
solver returns the homogenized stress tensor and tangent stiffness.

The ``QuadratureExtractor`` provides the FEniCSx v0.10+ API calls to:

1. Create and manage Quadrature function spaces::

       V_quad = dolfinx.fem.functionspace(mesh, ("Quadrature", degree, shape))

2. Extract tensor values (e.g. macroscopic strain) from Quadrature functions.
3. Inject tensor values (e.g. homogenized stress) back in, with correct
   element → global DoF mapping.

This corresponds to Section 1.2 of the Advanced Features Addendum.

Typical usage
-------------
>>> extractor = QuadratureExtractor()
>>> extractor.register(mesh, quadrature_degree=2, tensor_shape=(3, 3))
>>> strains = extractor.extract_values(macro_strain_function)
>>> extractor.inject_values(stress_function, homogenized_stresses)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np

from fenicsx_cosim.utils import get_logger

logger = get_logger(__name__)

# Lazy DOLFINx import
try:
    import basix
    import dolfinx
    import dolfinx.fem
    import dolfinx.mesh
    import ufl

    _HAS_DOLFINX = True
except ImportError:  # pragma: no cover
    _HAS_DOLFINX = False


def _require_dolfinx() -> None:
    """Raise an informative error if DOLFINx is not installed."""
    if not _HAS_DOLFINX:
        raise ImportError(
            "DOLFINx is required for QuadratureExtractor but could not be "
            "imported.  Install it with: pip install fenics-dolfinx"
        )


@dataclass
class QuadratureData:
    """Container for quadrature-point information.

    Attributes
    ----------
    num_cells : int
        Number of mesh cells (elements).
    points_per_cell : int
        Number of quadrature/integration points per cell.
    total_points : int
        ``num_cells * points_per_cell``.
    tensor_shape : tuple[int, ...]
        Shape of each tensor at each integration point (e.g. ``(3, 3)``
        for a 2D symmetric stress/strain tensor, or ``(6,)`` for Voigt).
    dof_per_value : int
        Number of scalar DoFs per integration point (product of
        ``tensor_shape``).
    coordinates : np.ndarray
        Physical coordinates of all integration points —
        shape ``(total_points, gdim)``.
    """

    num_cells: int = 0
    points_per_cell: int = 0
    total_points: int = 0
    tensor_shape: tuple = ()
    dof_per_value: int = 1
    coordinates: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )


class QuadratureExtractor:
    """Extract and inject tensor data at integration points.

    This class works with FEniCSx Quadrature function spaces to handle
    the data exchange required for FE² (computational homogenization).

    The user creates a ``QuadratureExtractor``, registers it with a mesh
    and quadrature degree, and then uses it to extract/inject tensors
    that are stored as ``dolfinx.fem.Function`` objects on the Quadrature
    space.
    """

    def __init__(self) -> None:
        _require_dolfinx()
        self._quad_data: Optional[QuadratureData] = None
        self._function_space = None
        self._mesh = None
        self._quadrature_degree: int = 0
        self._tensor_shape: tuple = ()
        self._cell_to_dof_map: Optional[np.ndarray] = None
        logger.debug("QuadratureExtractor created")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        mesh: "dolfinx.mesh.Mesh",
        quadrature_degree: int = 2,
        tensor_shape: Union[Tuple[int, ...], int] = (),
    ) -> QuadratureData:
        """Register a mesh and create a Quadrature function space.

        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
            The computational mesh (typically the macro-mesh in FE²).
        quadrature_degree : int, optional
            The polynomial degree for the quadrature rule.  Default = 2.
        tensor_shape : tuple[int, ...] or int, optional
            Shape of the tensor at each integration point.  Examples:

            * ``()`` or ``1`` — scalar field.
            * ``(3,)`` — vector in 2D (Voigt stress/strain).
            * ``(6,)`` — symmetric tensor in 3D (Voigt notation).
            * ``(3, 3)`` — full 2D tensor.

        Returns
        -------
        QuadratureData
            Metadata about the quadrature points.
        """
        self._mesh = mesh
        self._quadrature_degree = quadrature_degree

        if isinstance(tensor_shape, int):
            tensor_shape = (tensor_shape,) if tensor_shape > 1 else ()
        self._tensor_shape = tensor_shape

        # Determine the number of scalar DoFs per integration point
        dof_per_value = int(np.prod(tensor_shape)) if tensor_shape else 1

        # Create the Quadrature function space
        # In FEniCSx v0.10+ the Quadrature element is created via basix
        tdim = mesh.topology.dim
        gdim = mesh.geometry.dim

        # Create the Quadrature element
        cell_type = mesh.topology.cell_type
        basix_cell = getattr(basix.CellType, cell_type.name)

        q_element = basix.ufl.quadrature_element(
            basix_cell,
            value_shape=tensor_shape,
            degree=quadrature_degree,
        )

        self._function_space = dolfinx.fem.functionspace(mesh, q_element)

        # Get the number of cells
        mesh.topology.create_entities(tdim)
        num_cells = mesh.topology.index_map(tdim).size_local

        # Determine quadrature points per cell from the function space
        # The number of DoFs per cell gives us points_per_cell * dof_per_value
        dofmap = self._function_space.dofmap
        # Each cell maps to some DoFs; the number of DoFs / dof_per_value
        # = points per cell
        num_dofs_per_cell = dofmap.dof_layout.num_dofs
        points_per_cell = num_dofs_per_cell  # for scalar; each quad point is a DoF

        total_points = num_cells * points_per_cell

        # Compute physical coordinates of all integration points
        quad_coords = self._compute_quadrature_coordinates(
            mesh, quadrature_degree, num_cells, points_per_cell
        )

        # Build cell → global DoF mapping
        self._build_cell_to_dof_map(num_cells)

        self._quad_data = QuadratureData(
            num_cells=num_cells,
            points_per_cell=points_per_cell,
            total_points=total_points,
            tensor_shape=tensor_shape,
            dof_per_value=dof_per_value,
            coordinates=quad_coords,
        )

        logger.info(
            "QuadratureExtractor registered — %d cells × %d pts/cell = "
            "%d integration points, tensor_shape=%s",
            num_cells,
            points_per_cell,
            total_points,
            tensor_shape,
        )
        return self._quad_data

    def register_with_function_space(
        self,
        function_space: "dolfinx.fem.FunctionSpace",
        tensor_shape: Union[Tuple[int, ...], int] = (),
    ) -> QuadratureData:
        """Register using a pre-existing Quadrature function space.

        Parameters
        ----------
        function_space : dolfinx.fem.FunctionSpace
            An existing Quadrature function space.
        tensor_shape : tuple[int, ...] or int, optional
            Shape of the tensor at each integration point.

        Returns
        -------
        QuadratureData
        """
        self._function_space = function_space
        mesh = function_space.mesh
        self._mesh = mesh

        if isinstance(tensor_shape, int):
            tensor_shape = (tensor_shape,) if tensor_shape > 1 else ()
        self._tensor_shape = tensor_shape

        dof_per_value = int(np.prod(tensor_shape)) if tensor_shape else 1

        tdim = mesh.topology.dim
        mesh.topology.create_entities(tdim)
        num_cells = mesh.topology.index_map(tdim).size_local

        dofmap = function_space.dofmap
        num_dofs_per_cell = dofmap.dof_layout.num_dofs
        points_per_cell = num_dofs_per_cell

        total_points = num_cells * points_per_cell

        self._build_cell_to_dof_map(num_cells)

        self._quad_data = QuadratureData(
            num_cells=num_cells,
            points_per_cell=points_per_cell,
            total_points=total_points,
            tensor_shape=tensor_shape,
            dof_per_value=dof_per_value,
        )

        logger.info(
            "QuadratureExtractor registered (from function space) — "
            "%d cells × %d pts/cell = %d total points",
            num_cells,
            points_per_cell,
            total_points,
        )
        return self._quad_data

    # ------------------------------------------------------------------
    # Extraction / injection
    # ------------------------------------------------------------------

    def extract_values(
        self, function: "dolfinx.fem.Function"
    ) -> np.ndarray:
        """Extract all values from a Quadrature function.

        Parameters
        ----------
        function : dolfinx.fem.Function
            A function defined on the registered Quadrature space.

        Returns
        -------
        np.ndarray
            Flat array of all integration-point values in cell order.
            Shape ``(total_points * dof_per_value,)`` or reshaped
            to ``(total_points, *tensor_shape)`` if tensor_shape is set.
        """
        self._check_registered()
        values = np.asarray(function.x.array, dtype=np.float64).copy()
        qd = self._quad_data

        if qd.tensor_shape:
            # Reshape to (total_points, *tensor_shape)
            values = values.reshape(qd.total_points, *qd.tensor_shape)

        return values

    def extract_cell_values(
        self,
        function: "dolfinx.fem.Function",
        cell_index: int,
    ) -> np.ndarray:
        """Extract values for a single cell (element).

        Parameters
        ----------
        function : dolfinx.fem.Function
            A function on the Quadrature space.
        cell_index : int
            The local cell index.

        Returns
        -------
        np.ndarray
            Values at the integration points of the specified cell.
            Shape ``(points_per_cell,)`` for scalar, or
            ``(points_per_cell, *tensor_shape)`` for tensors.
        """
        self._check_registered()
        dofs = self._cell_to_dof_map[cell_index]
        values = np.asarray(function.x.array[dofs], dtype=np.float64)

        qd = self._quad_data
        if qd.tensor_shape:
            values = values.reshape(qd.points_per_cell, *qd.tensor_shape)

        return values

    def inject_values(
        self,
        function: "dolfinx.fem.Function",
        values: np.ndarray,
    ) -> None:
        """Inject values into a Quadrature function.

        Parameters
        ----------
        function : dolfinx.fem.Function
            Target function on the Quadrature space.
        values : np.ndarray
            Values to inject.  Can be flat
            ``(total_points * dof_per_value,)`` or shaped
            ``(total_points, *tensor_shape)``.
        """
        self._check_registered()
        flat = values.ravel()
        expected = len(function.x.array)
        if len(flat) != expected:
            raise ValueError(
                f"Expected {expected} values, got {len(flat)}.  "
                f"(total_points={self._quad_data.total_points}, "
                f"dof_per_value={self._quad_data.dof_per_value})"
            )
        function.x.array[:] = flat

    def inject_cell_values(
        self,
        function: "dolfinx.fem.Function",
        cell_index: int,
        values: np.ndarray,
    ) -> None:
        """Inject values for a single cell.

        Parameters
        ----------
        function : dolfinx.fem.Function
            Target function on the Quadrature space.
        cell_index : int
            The local cell index.
        values : np.ndarray
            Values for the integration points of this cell.
        """
        self._check_registered()
        dofs = self._cell_to_dof_map[cell_index]
        flat = values.ravel()
        if len(flat) != len(dofs):
            raise ValueError(
                f"Cell {cell_index}: expected {len(dofs)} values, "
                f"got {len(flat)}"
            )
        function.x.array[dofs] = flat

    # ------------------------------------------------------------------
    # Batch operations for FE² dispatch
    # ------------------------------------------------------------------

    def extract_for_dispatch(
        self, function: "dolfinx.fem.Function"
    ) -> list[np.ndarray]:
        """Extract per-cell integration-point values for dispatching to RVEs.

        Returns a list of ``num_cells`` arrays, each of shape
        ``(points_per_cell,)`` or ``(points_per_cell, *tensor_shape)``.
        Cell ``i`` corresponds to RVE ``i``.

        Parameters
        ----------
        function : dolfinx.fem.Function
            Source function on the Quadrature space.

        Returns
        -------
        list[np.ndarray]
            One array per cell.
        """
        self._check_registered()
        qd = self._quad_data
        result = []
        for i in range(qd.num_cells):
            result.append(self.extract_cell_values(function, i))
        return result

    def inject_from_gather(
        self,
        function: "dolfinx.fem.Function",
        cell_values: list[np.ndarray],
    ) -> None:
        """Inject per-cell values gathered from RVE solves.

        Parameters
        ----------
        function : dolfinx.fem.Function
            Target function on the Quadrature space.
        cell_values : list[np.ndarray]
            One array per cell, as returned by the RVE workers.
        """
        self._check_registered()
        qd = self._quad_data
        if len(cell_values) != qd.num_cells:
            raise ValueError(
                f"Expected {qd.num_cells} cell results, "
                f"got {len(cell_values)}"
            )
        for i, vals in enumerate(cell_values):
            self.inject_cell_values(function, i, vals)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def quadrature_data(self) -> QuadratureData:
        """Metadata about the registered quadrature configuration."""
        self._check_registered()
        return self._quad_data

    @property
    def function_space(self):
        """The Quadrature function space."""
        self._check_registered()
        return self._function_space

    @property
    def cell_to_dof_map(self) -> np.ndarray:
        """Cell → global DoF index mapping — shape ``(num_cells, dofs_per_cell)``."""
        self._check_registered()
        return self._cell_to_dof_map

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_cell_to_dof_map(self, num_cells: int) -> None:
        """Build the cell → global DoF index lookup table."""
        dofmap = self._function_space.dofmap
        num_dofs_per_cell = dofmap.dof_layout.num_dofs
        mapping = np.empty((num_cells, num_dofs_per_cell), dtype=np.int64)
        for i in range(num_cells):
            mapping[i, :] = dofmap.cell_dofs(i)
        self._cell_to_dof_map = mapping
        logger.debug(
            "Built cell→DoF map: %d cells × %d dofs/cell",
            num_cells,
            num_dofs_per_cell,
        )

    def _compute_quadrature_coordinates(
        self,
        mesh: "dolfinx.mesh.Mesh",
        degree: int,
        num_cells: int,
        points_per_cell: int,
    ) -> np.ndarray:
        """Compute physical coordinates of all integration points.

        Uses the mesh geometry to push-forward reference quadrature
        points to physical coordinates.

        Returns
        -------
        np.ndarray
            Shape ``(total_points, gdim)``.
        """
        gdim = mesh.geometry.dim
        tdim = mesh.topology.dim

        # Get the basix quadrature points for the reference cell
        cell_type = mesh.topology.cell_type
        basix_cell = getattr(basix.CellType, cell_type.name)
        q_points, _ = basix.make_quadrature(basix_cell, degree)

        # Map each cell's reference points to physical coordinates
        # using the mesh coordinate element
        x = mesh.geometry.x  # physical node coordinates
        dofmap_geom = mesh.geometry.dofmap

        # Tabulate the geometry basis functions at the quadrature points
        cmap = mesh.geometry.cmap
        tab = cmap.tabulate(0, q_points)
        # tab[0] has shape (num_q_points, num_geometry_dofs)
        basis_vals = tab[0, :, :, 0]  # (num_q_points, num_geom_dofs)

        total = num_cells * points_per_cell
        coords = np.empty((total, gdim), dtype=np.float64)

        for cell in range(num_cells):
            cell_nodes = dofmap_geom[cell]
            cell_node_coords = x[cell_nodes, :gdim]  # (num_geom_dofs, gdim)
            # Physical coords = basis_vals @ cell_node_coords
            phys = basis_vals @ cell_node_coords  # (points_per_cell, gdim)
            start = cell * points_per_cell
            coords[start : start + points_per_cell, :] = phys

        return coords

    def _check_registered(self) -> None:
        if self._quad_data is None:
            raise RuntimeError(
                "No quadrature space registered. Call register() first."
            )
