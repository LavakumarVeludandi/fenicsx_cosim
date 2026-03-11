"""
Tests for the QuadratureExtractor module.

These tests verify:
  - Registration of quadrature spaces on FEniCSx meshes
  - Extraction and injection of scalar and tensor values
  - Cell-wise dispatch/gather operations for FE²
  - Correct cell → DoF mapping
  - Quadrature coordinate computation

Requires: DOLFINx v0.10+, basix, mpi4py
"""

import numpy as np
import pytest

# Skip the entire module if DOLFINx is not available
dolfinx = pytest.importorskip("dolfinx")
basix = pytest.importorskip("basix")
from mpi4py import MPI  # noqa: E402

import dolfinx.fem  # noqa: E402
import dolfinx.mesh  # noqa: E402

from fenicsx_cosim.quadrature_extractor import QuadratureExtractor, QuadratureData  # noqa: E402


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def unit_square_mesh():
    """A simple 2×2 unit square mesh."""
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)


@pytest.fixture
def unit_cube_mesh():
    """A simple 2×2×2 unit cube mesh."""
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)


@pytest.fixture
def fine_square_mesh():
    """A finer 4×4 unit square mesh."""
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)


# ======================================================================
# Tests: Registration
# ======================================================================

class TestQuadratureExtractorRegistration:
    """Tests for quadrature space registration."""

    def test_register_scalar(self, unit_square_mesh):
        qe = QuadratureExtractor()
        qd = qe.register(unit_square_mesh, quadrature_degree=2)

        assert isinstance(qd, QuadratureData)
        assert qd.num_cells > 0
        assert qd.points_per_cell > 0
        assert qd.total_points == qd.num_cells * qd.points_per_cell
        assert qd.tensor_shape == ()
        assert qd.dof_per_value == 1

    def test_register_with_tensor_shape(self, unit_square_mesh):
        qe = QuadratureExtractor()
        qd = qe.register(
            unit_square_mesh,
            quadrature_degree=2,
            tensor_shape=(3,),
        )

        assert qd.tensor_shape == (3,)
        assert qd.dof_per_value == 3

    def test_register_matrix_tensor(self, unit_square_mesh):
        qe = QuadratureExtractor()
        qd = qe.register(
            unit_square_mesh,
            quadrature_degree=2,
            tensor_shape=(2, 2),
        )

        assert qd.tensor_shape == (2, 2)
        assert qd.dof_per_value == 4

    def test_register_3d_mesh(self, unit_cube_mesh):
        qe = QuadratureExtractor()
        qd = qe.register(unit_cube_mesh, quadrature_degree=2)

        assert qd.num_cells > 0
        assert qd.total_points > 0

    def test_coordinates_computed(self, unit_square_mesh):
        qe = QuadratureExtractor()
        qd = qe.register(unit_square_mesh, quadrature_degree=2)

        coords = qd.coordinates
        assert coords.shape == (qd.total_points, unit_square_mesh.geometry.dim)
        # All coordinates inside [0, 1] × [0, 1]
        assert np.all(coords >= -1e-10)
        assert np.all(coords <= 1.0 + 1e-10)

    def test_not_registered_raises(self):
        qe = QuadratureExtractor()
        with pytest.raises(RuntimeError, match="No quadrature space registered"):
            _ = qe.quadrature_data


# ======================================================================
# Tests: Cell-to-DoF mapping
# ======================================================================

class TestCellToDofMap:
    """Tests for the cell→global DoF index mapping."""

    def test_map_shape(self, unit_square_mesh):
        qe = QuadratureExtractor()
        qd = qe.register(unit_square_mesh, quadrature_degree=2)

        cell_map = qe.cell_to_dof_map
        assert cell_map.shape[0] == qd.num_cells
        assert cell_map.shape[1] > 0  # at least 1 DoF per cell

    def test_map_covers_all_dofs(self, unit_square_mesh):
        qe = QuadratureExtractor()
        qd = qe.register(unit_square_mesh, quadrature_degree=2)

        cell_map = qe.cell_to_dof_map
        all_dofs = np.unique(cell_map.ravel())
        # In Quadrature spaces, each DoF belongs to exactly one cell
        assert len(all_dofs) == qd.total_points


# ======================================================================
# Tests: Extraction and Injection
# ======================================================================

class TestExtractionInjection:
    """Tests for extract_values / inject_values."""

    def test_extract_flat_scalar(self, unit_square_mesh):
        qe = QuadratureExtractor()
        qd = qe.register(unit_square_mesh, quadrature_degree=2)

        func = dolfinx.fem.Function(qe.function_space)
        func.x.array[:] = np.arange(len(func.x.array), dtype=np.float64)

        values = qe.extract_values(func)
        assert values.shape == (qd.total_points,)
        np.testing.assert_array_equal(values, func.x.array)

    def test_inject_values(self, unit_square_mesh):
        qe = QuadratureExtractor()
        qd = qe.register(unit_square_mesh, quadrature_degree=2)

        func = dolfinx.fem.Function(qe.function_space)
        new_values = np.ones(qd.total_points) * 42.0
        qe.inject_values(func, new_values)

        np.testing.assert_array_almost_equal(func.x.array, 42.0)

    def test_inject_wrong_size_raises(self, unit_square_mesh):
        qe = QuadratureExtractor()
        qd = qe.register(unit_square_mesh, quadrature_degree=2)

        func = dolfinx.fem.Function(qe.function_space)
        with pytest.raises(ValueError, match="Expected"):
            qe.inject_values(func, np.zeros(7))

    def test_roundtrip(self, unit_square_mesh):
        """Extract → modify → inject → extract should be consistent."""
        qe = QuadratureExtractor()
        qd = qe.register(unit_square_mesh, quadrature_degree=2)

        func = dolfinx.fem.Function(qe.function_space)
        original = np.random.rand(qd.total_points)
        qe.inject_values(func, original)
        extracted = qe.extract_values(func)

        np.testing.assert_array_almost_equal(extracted, original)


# ======================================================================
# Tests: Cell-wise extraction / injection
# ======================================================================

class TestCellWiseOps:
    """Tests for extract_cell_values / inject_cell_values."""

    def test_extract_single_cell(self, unit_square_mesh):
        qe = QuadratureExtractor()
        qd = qe.register(unit_square_mesh, quadrature_degree=2)

        func = dolfinx.fem.Function(qe.function_space)
        func.x.array[:] = np.arange(len(func.x.array), dtype=np.float64)

        cell_vals = qe.extract_cell_values(func, 0)
        assert cell_vals.shape == (qd.points_per_cell,)

        # Values should match the DoFs for cell 0
        expected_dofs = qe.cell_to_dof_map[0]
        np.testing.assert_array_almost_equal(
            cell_vals, func.x.array[expected_dofs]
        )

    def test_inject_single_cell(self, unit_square_mesh):
        qe = QuadratureExtractor()
        qd = qe.register(unit_square_mesh, quadrature_degree=2)

        func = dolfinx.fem.Function(qe.function_space)
        func.x.array[:] = 0.0

        cell_vals = np.ones(qd.points_per_cell) * 99.0
        qe.inject_cell_values(func, 0, cell_vals)

        expected_dofs = qe.cell_to_dof_map[0]
        np.testing.assert_array_almost_equal(
            func.x.array[expected_dofs], 99.0
        )

    def test_inject_cell_wrong_size_raises(self, unit_square_mesh):
        qe = QuadratureExtractor()
        qd = qe.register(unit_square_mesh, quadrature_degree=2)

        func = dolfinx.fem.Function(qe.function_space)
        with pytest.raises(ValueError, match="Cell 0"):
            qe.inject_cell_values(func, 0, np.zeros(999))


# ======================================================================
# Tests: Dispatch / Gather for FE²
# ======================================================================

class TestDispatchGather:
    """Tests for extract_for_dispatch / inject_from_gather."""

    def test_dispatch_returns_per_cell(self, unit_square_mesh):
        qe = QuadratureExtractor()
        qd = qe.register(unit_square_mesh, quadrature_degree=2)

        func = dolfinx.fem.Function(qe.function_space)
        func.x.array[:] = np.random.rand(len(func.x.array))

        dispatch_list = qe.extract_for_dispatch(func)
        assert len(dispatch_list) == qd.num_cells
        for arr in dispatch_list:
            assert arr.shape == (qd.points_per_cell,)

    def test_gather_injects_all_cells(self, unit_square_mesh):
        qe = QuadratureExtractor()
        qd = qe.register(unit_square_mesh, quadrature_degree=2)

        func = dolfinx.fem.Function(qe.function_space)
        func.x.array[:] = 0.0

        # Create "results" for each cell
        cell_values = [
            np.ones(qd.points_per_cell) * (i + 1)
            for i in range(qd.num_cells)
        ]
        qe.inject_from_gather(func, cell_values)

        # Check each cell
        for i in range(qd.num_cells):
            dofs = qe.cell_to_dof_map[i]
            np.testing.assert_array_almost_equal(
                func.x.array[dofs], float(i + 1)
            )

    def test_gather_wrong_count_raises(self, unit_square_mesh):
        qe = QuadratureExtractor()
        qd = qe.register(unit_square_mesh, quadrature_degree=2)

        func = dolfinx.fem.Function(qe.function_space)
        with pytest.raises(ValueError, match="Expected"):
            qe.inject_from_gather(func, [np.zeros(3)])

    def test_roundtrip_dispatch_gather(self, unit_square_mesh):
        """dispatch → simulate RVE → gather should preserve data."""
        qe = QuadratureExtractor()
        qd = qe.register(unit_square_mesh, quadrature_degree=2)

        source_func = dolfinx.fem.Function(qe.function_space)
        source_func.x.array[:] = np.random.rand(len(source_func.x.array))

        dispatched = qe.extract_for_dispatch(source_func)

        # Simulate: multiply each cell by 2 (pretend RVE solve)
        results = [arr * 2.0 for arr in dispatched]

        target_func = dolfinx.fem.Function(qe.function_space)
        qe.inject_from_gather(target_func, results)

        np.testing.assert_array_almost_equal(
            target_func.x.array, source_func.x.array * 2.0
        )
