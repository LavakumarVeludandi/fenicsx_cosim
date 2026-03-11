"""
Tests for the MeshExtractor.

NOTE: These tests require a working DOLFINx installation.  They are
skipped automatically if DOLFINx is not available.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import dolfinx
    import dolfinx.fem
    import dolfinx.mesh
    from mpi4py import MPI

    _HAS_DOLFINX = True
except ImportError:
    _HAS_DOLFINX = False

pytestmark = pytest.mark.skipif(
    not _HAS_DOLFINX, reason="DOLFINx not available"
)


@pytest.fixture
def unit_square_mesh():
    """Create a simple unit square mesh for testing."""
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)


@pytest.fixture
def unit_square_with_tags(unit_square_mesh):
    """Create a unit square mesh with facet tags for the top boundary
    (y=1)."""
    mesh = unit_square_mesh
    tdim = mesh.topology.dim
    fdim = tdim - 1

    # Locate facets on y=1 boundary
    def top_boundary(x):
        return np.isclose(x[1], 1.0)

    facet_indices = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, top_boundary
    )

    # Create mesh tags
    marker_id = 1
    facet_values = np.full(len(facet_indices), marker_id, dtype=np.int32)
    facet_tags = dolfinx.mesh.meshtags(
        mesh, fdim, facet_indices, facet_values
    )

    return mesh, facet_tags, marker_id


class TestMeshExtractor:

    def test_register_boundary(self, unit_square_with_tags):
        """Test boundary registration with MeshTags."""
        from fenicsx_cosim.mesh_extractor import MeshExtractor

        mesh, facet_tags, marker_id = unit_square_with_tags
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

        extractor = MeshExtractor()
        bd = extractor.register(mesh, facet_tags, marker_id, V)

        # Should have boundary dofs
        assert len(bd.dof_indices) > 0
        assert bd.coordinates.shape[1] == 3

        # All boundary points should be at y ≈ 1.0
        np.testing.assert_allclose(bd.coordinates[:, 1], 1.0, atol=1e-12)

        # Accessors should work
        assert np.array_equal(
            extractor.boundary_coordinates, bd.coordinates
        )
        assert np.array_equal(
            extractor.boundary_dof_indices, bd.dof_indices
        )

    def test_register_from_locator(self, unit_square_mesh):
        """Test boundary registration with a locator function."""
        from fenicsx_cosim.mesh_extractor import MeshExtractor

        mesh = unit_square_mesh
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

        def bottom_boundary(x):
            return np.isclose(x[1], 0.0)

        extractor = MeshExtractor()
        bd = extractor.register_from_locator(
            mesh, bottom_boundary, V, marker_id=2
        )

        assert len(bd.dof_indices) > 0
        np.testing.assert_allclose(bd.coordinates[:, 1], 0.0, atol=1e-12)
        assert bd.marker_id == 2

    def test_extract_inject_values(self, unit_square_with_tags):
        """Test extracting and injecting values at boundary DoFs."""
        from fenicsx_cosim.mesh_extractor import MeshExtractor

        mesh, facet_tags, marker_id = unit_square_with_tags
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

        extractor = MeshExtractor()
        bd = extractor.register(mesh, facet_tags, marker_id, V)

        # Create a function with known boundary values
        u = dolfinx.fem.Function(V)
        u.interpolate(lambda x: x[0] + 2 * x[1])

        # Extract boundary values
        values = extractor.extract_boundary_values(u)
        assert len(values) == len(bd.dof_indices)

        # Since y=1 on the boundary: expected = x[0] + 2*1 = x[0] + 2
        expected = bd.coordinates[:, 0] + 2.0
        np.testing.assert_allclose(values, expected, atol=1e-12)

        # Inject new values
        new_values = np.ones(len(bd.dof_indices)) * 42.0
        extractor.inject_boundary_values(u, new_values)

        # Verify injection
        extracted = extractor.extract_boundary_values(u)
        np.testing.assert_allclose(extracted, 42.0, atol=1e-12)

    def test_inject_wrong_size_raises(self, unit_square_with_tags):
        """inject_boundary_values should raise if value count mismatches."""
        from fenicsx_cosim.mesh_extractor import MeshExtractor

        mesh, facet_tags, marker_id = unit_square_with_tags
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

        extractor = MeshExtractor()
        extractor.register(mesh, facet_tags, marker_id, V)

        u = dolfinx.fem.Function(V)
        with pytest.raises(ValueError, match="Expected"):
            extractor.inject_boundary_values(u, np.array([1.0, 2.0]))

    def test_no_registration_raises(self):
        """Accessing boundary_data before registration should raise."""
        from fenicsx_cosim.mesh_extractor import MeshExtractor

        extractor = MeshExtractor()
        with pytest.raises(RuntimeError, match="No boundary registered"):
            _ = extractor.boundary_data
