"""
Tests for the DataMapper (interpolation engine).

These tests are standalone — no DOLFINx needed.  They exercise the
NearestNeighborMapper with synthetic point clouds.
"""

from __future__ import annotations

import numpy as np
import pytest

from fenicsx_cosim.data_mapper import NearestNeighborMapper


class TestNearestNeighborMapper:
    """Tests for ``NearestNeighborMapper``."""

    def test_identical_grids(self):
        """When source and target grids are identical, mapping is exact."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        values = np.array([10.0, 20.0, 30.0, 40.0])

        mapper = NearestNeighborMapper()
        mapper.build(source_coords=coords, target_coords=coords)

        mapped = mapper.map(values)
        np.testing.assert_array_equal(mapped, values)
        assert mapper.max_distance == pytest.approx(0.0, abs=1e-12)

    def test_shifted_grid(self):
        """Source and target grids with slight offset — nearest neighbor
        should still find the correct point."""
        source = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        target = np.array([
            [0.1, 0.0, 0.0],   # closest to source[0]
            [0.9, 0.0, 0.0],   # closest to source[1]
            [2.1, 0.0, 0.0],   # closest to source[2]
        ])
        values = np.array([100.0, 200.0, 300.0])

        mapper = NearestNeighborMapper()
        mapper.build(source, target)

        mapped = mapper.map(values)
        np.testing.assert_array_equal(mapped, values)

    def test_different_sizes(self):
        """Source has more points than target."""
        source = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        target = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        mapper = NearestNeighborMapper()
        mapper.build(source, target)

        mapped = mapper.map(values)
        assert len(mapped) == 3
        np.testing.assert_array_equal(mapped, [10.0, 30.0, 50.0])

    def test_inverse_map(self):
        """Inverse mapping from target → source coordinates."""
        source = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        target = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        target_values = np.array([100.0, 200.0, 300.0])

        mapper = NearestNeighborMapper()
        mapper.build(source, target)

        inv = mapper.inverse_map(target_values)
        assert len(inv) == 2
        # source[0] closest to target[0] → 100
        # source[1] closest to target[2] → 300
        np.testing.assert_array_equal(inv, [100.0, 300.0])

    def test_vector_field(self):
        """Mapping a 3D vector field (N, 3) instead of scalar."""
        source = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        target = np.array([
            [0.1, 0.0, 0.0],
            [0.9, 0.0, 0.0],
        ])
        values = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])

        mapper = NearestNeighborMapper()
        mapper.build(source, target)

        mapped = mapper.map(values)
        assert mapped.shape == (2, 3)
        np.testing.assert_array_equal(mapped[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(mapped[1], [4.0, 5.0, 6.0])

    def test_not_built_raises(self):
        """Calling map() before build() should raise."""
        mapper = NearestNeighborMapper()
        with pytest.raises(RuntimeError, match="not built"):
            mapper.map(np.array([1.0, 2.0]))

    def test_forward_distances(self):
        """Check the forward distance diagnostic."""
        source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        target = np.array([[0.0, 0.1, 0.0]])

        mapper = NearestNeighborMapper()
        mapper.build(source, target)

        assert mapper.max_distance == pytest.approx(0.1, abs=1e-10)
        assert len(mapper.forward_distances) == 1

    def test_3d_cloud(self):
        """Mapping with a realistic 3D point cloud."""
        rng = np.random.default_rng(42)
        source = rng.random((50, 3))
        target = source + rng.normal(0, 0.01, source.shape)
        values = rng.random(50)

        mapper = NearestNeighborMapper()
        mapper.build(source, target)

        mapped = mapper.map(values)
        # With such small perturbations, nearest neighbor should recover
        # the original values exactly
        np.testing.assert_array_equal(mapped, values)
