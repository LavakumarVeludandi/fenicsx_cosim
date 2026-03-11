"""
Tests for the DynamicMapper module.

These tests verify:
  - Building and rebuilding of the DynamicMapper
  - Invalidation and stale-mapping detection
  - Negotiation protocol between two communicators
  - Correct mapping after mesh refinement (coordinate change)
"""

import threading
import time

import numpy as np
import pytest

from fenicsx_cosim.communicator import Communicator
from fenicsx_cosim.dynamic_mapper import (
    NO_UPDATE_SIGNAL,
    UPDATE_MESH_ACK,
    UPDATE_MESH_SIGNAL,
    DynamicMapper,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def source_coords():
    """5 points on a line (partner boundary)."""
    return np.array([
        [0.0, 1.0, 0.0],
        [0.25, 1.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.75, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])


@pytest.fixture
def target_coords():
    """5 points slightly offset (local boundary)."""
    return np.array([
        [0.0, 1.01, 0.0],
        [0.25, 1.01, 0.0],
        [0.5, 1.01, 0.0],
        [0.75, 1.01, 0.0],
        [1.0, 1.01, 0.0],
    ])


@pytest.fixture
def refined_target_coords():
    """9 points after 'refinement' — denser spacing."""
    return np.array([
        [0.0, 1.01, 0.0],
        [0.125, 1.01, 0.0],
        [0.25, 1.01, 0.0],
        [0.375, 1.01, 0.0],
        [0.5, 1.01, 0.0],
        [0.625, 1.01, 0.0],
        [0.75, 1.01, 0.0],
        [0.875, 1.01, 0.0],
        [1.0, 1.01, 0.0],
    ])


# ======================================================================
# Tests: Building and basic mapping
# ======================================================================

class TestDynamicMapperBuild:
    """Tests for build / map / inverse_map."""

    def test_build_and_map(self, source_coords, target_coords):
        dm = DynamicMapper()
        dm.build(source_coords, target_coords)

        assert dm.revision == 1
        assert not dm.needs_update

        source_values = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        mapped = dm.map(source_values)
        assert len(mapped) == len(target_coords)
        # Nearest-neighbor: values should match exactly (same count)
        np.testing.assert_array_almost_equal(mapped, source_values)

    def test_inverse_map(self, source_coords, target_coords):
        dm = DynamicMapper()
        dm.build(source_coords, target_coords)

        target_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        inv_mapped = dm.inverse_map(target_values)
        assert len(inv_mapped) == len(source_coords)
        np.testing.assert_array_almost_equal(inv_mapped, target_values)

    def test_rebuild_increments_revision(self, source_coords, target_coords):
        dm = DynamicMapper()
        dm.build(source_coords, target_coords)
        assert dm.revision == 1

        dm.build(source_coords, target_coords)
        assert dm.revision == 2

    def test_max_distance(self, source_coords, target_coords):
        dm = DynamicMapper()
        dm.build(source_coords, target_coords)
        assert dm.max_distance is not None
        assert dm.max_distance >= 0.0

    def test_coordinates_stored(self, source_coords, target_coords):
        dm = DynamicMapper()
        dm.build(source_coords, target_coords)
        np.testing.assert_array_equal(dm.partner_coordinates, source_coords)
        np.testing.assert_array_equal(dm.local_coordinates, target_coords)


# ======================================================================
# Tests: Invalidation
# ======================================================================

class TestDynamicMapperInvalidation:
    """Tests for invalidation and stale-mapping guards."""

    def test_invalidate_marks_stale(self, source_coords, target_coords):
        dm = DynamicMapper()
        dm.build(source_coords, target_coords)
        assert not dm.needs_update

        dm.invalidate()
        assert dm.needs_update

    def test_map_raises_when_stale(self, source_coords, target_coords):
        dm = DynamicMapper()
        dm.build(source_coords, target_coords)
        dm.invalidate()

        with pytest.raises(RuntimeError, match="stale"):
            dm.map(np.zeros(5))

    def test_inverse_map_raises_when_stale(self, source_coords, target_coords):
        dm = DynamicMapper()
        dm.build(source_coords, target_coords)
        dm.invalidate()

        with pytest.raises(RuntimeError, match="stale"):
            dm.inverse_map(np.zeros(5))

    def test_map_raises_when_not_built(self):
        dm = DynamicMapper()
        with pytest.raises(RuntimeError, match="not built"):
            dm.map(np.zeros(5))

    def test_rebuild_clears_stale(self, source_coords, target_coords):
        dm = DynamicMapper()
        dm.build(source_coords, target_coords)
        dm.invalidate()
        assert dm.needs_update

        dm.build(source_coords, target_coords)
        assert not dm.needs_update
        # Should work without error
        result = dm.map(np.ones(5))
        assert len(result) == 5


# ======================================================================
# Tests: Mapping after refinement
# ======================================================================

class TestDynamicMapperRefinement:
    """Tests for rebuilding after simulated mesh refinement."""

    def test_refined_mapping(self, source_coords, refined_target_coords):
        dm = DynamicMapper()
        # Initial build: 5 source → 5 target
        initial_target = source_coords.copy()
        initial_target[:, 1] += 0.01
        dm.build(source_coords, initial_target)
        assert dm.revision == 1

        # "Refine" — now we have 9 target points
        dm.invalidate()
        dm.build(source_coords, refined_target_coords)
        assert dm.revision == 2

        # Map: 5 source values → 9 target values
        src_vals = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        mapped = dm.map(src_vals)
        assert len(mapped) == 9
        # Boundary points should map to nearest source values
        assert mapped[0] == pytest.approx(100.0)
        assert mapped[-1] == pytest.approx(500.0)
        assert mapped[4] == pytest.approx(300.0)


# ======================================================================
# Tests: Negotiation protocol (with live ZeroMQ)
# ======================================================================

class TestDynamicMapperNegotiation:
    """Tests for the negotiate_update protocol using real ZeroMQ."""

    @pytest.fixture
    def endpoint(self):
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return f"tcp://127.0.0.1:{port}"

    def _create_pair(self, endpoint):
        """Create a bind/connect communicator pair."""
        c_bind = Communicator(
            name="SolverA",
            partner_name="SolverB",
            role="bind",
            endpoint=endpoint,
            timeout_ms=10_000,
            handshake=False,
        )
        c_connect = Communicator(
            name="SolverB",
            partner_name="SolverA",
            role="connect",
            endpoint=endpoint,
            timeout_ms=10_000,
            handshake=False,
        )
        return c_bind, c_connect

    def test_negotiate_no_update(self, endpoint, source_coords, target_coords):
        """When neither side needs an update, nothing happens."""
        c_bind, c_connect = self._create_pair(endpoint)

        dm_a = DynamicMapper()
        dm_a.build(source_coords, target_coords)

        dm_b = DynamicMapper()
        dm_b.build(target_coords, source_coords)

        results = [None, None]

        def side_a():
            results[0] = dm_a.negotiate_update(c_bind, "bind", None)

        def side_b():
            results[1] = dm_b.negotiate_update(c_connect, "connect", None)

        t_a = threading.Thread(target=side_a)
        t_b = threading.Thread(target=side_b)
        t_a.start()
        t_b.start()
        t_a.join(timeout=10)
        t_b.join(timeout=10)

        # No update needed → both return None
        assert results[0] is None
        assert results[1] is None

        c_bind.close()
        c_connect.close()

    def test_negotiate_one_side_updates(self, endpoint, source_coords,
                                         target_coords, refined_target_coords):
        """When one side refines, both exchange and rebuild."""
        c_bind, c_connect = self._create_pair(endpoint)

        dm_a = DynamicMapper()
        dm_a.build(source_coords, target_coords)
        dm_a.invalidate()  # Side A refined

        dm_b = DynamicMapper()
        dm_b.build(target_coords, source_coords)
        # Side B did NOT refine

        results = [None, None]

        def side_a():
            results[0] = dm_a.negotiate_update(
                c_bind, "bind", refined_target_coords
            )

        def side_b():
            results[1] = dm_b.negotiate_update(
                c_connect, "connect", None
            )

        t_a = threading.Thread(target=side_a)
        t_b = threading.Thread(target=side_b)
        t_a.start()
        t_b.start()
        t_a.join(timeout=10)
        t_b.join(timeout=10)

        # Both sides got partner's coordinates
        assert results[0] is not None  # A got B's (old) coords
        assert results[1] is not None  # B got A's refined coords

        # B should now have A's refined coordinates as partner coords
        assert len(dm_b.partner_coordinates) == len(refined_target_coords)
        np.testing.assert_array_almost_equal(
            dm_b.partner_coordinates, refined_target_coords
        )

        # Both mappers should be valid
        assert not dm_a.needs_update
        assert not dm_b.needs_update
        assert dm_a.revision == 2
        assert dm_b.revision == 2

        c_bind.close()
        c_connect.close()
