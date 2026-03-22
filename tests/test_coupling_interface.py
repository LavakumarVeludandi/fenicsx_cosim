"""
Tests for the CouplingInterface (integration tests).

NOTE: These tests require a working DOLFINx installation and run two
coupled solvers in threads.  They are skipped automatically if DOLFINx
is not available.
"""

from __future__ import annotations

import threading
import time

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


def _create_mesh_and_tags():
    """Create a unit square mesh with top-boundary facet tags."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    tdim = mesh.topology.dim
    fdim = tdim - 1

    def top_boundary(x):
        return np.isclose(x[1], 1.0)

    facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top_boundary)
    tags = dolfinx.mesh.meshtags(
        mesh, fdim, facets, np.full(len(facets), 1, dtype=np.int32)
    )
    return mesh, tags


def _run_solver_a(results: dict, endpoint: str) -> None:
    """Thermal solver (bind side)."""
    from fenicsx_cosim.coupling_interface import CouplingInterface

    try:
        mesh, facet_tags = _create_mesh_and_tags()
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

        cosim = CouplingInterface(
            name="SolverA",
            partner_name="SolverB",
            role="bind",
            endpoint=endpoint,
            timeout_ms=15_000,
        )

        cosim.register_interface(mesh, facet_tags, marker_id=1, function_space=V)

        # Create a temperature-like function
        temperature = dolfinx.fem.Function(V)
        temperature.interpolate(lambda x: 100.0 + x[0] * 50.0)

        # Export temperature
        cosim.export_data("TemperatureField", temperature)

        # Import displacement
        displacement = dolfinx.fem.Function(V)
        cosim.import_data("DisplacementField", displacement)

        # Verify we received something
        bd_vals = cosim._extractor.extract_boundary_values(displacement)
        results["solver_a_received"] = bd_vals.copy()

        cosim.advance_in_time()
        results["solver_a_step"] = cosim.step_count

        cosim.disconnect()
        results["solver_a_ok"] = True

    except Exception as e:
        results["solver_a_error"] = str(e)
        import traceback
        results["solver_a_tb"] = traceback.format_exc()


def _run_solver_b(results: dict, endpoint: str) -> None:
    """Mechanical solver (connect side)."""
    from fenicsx_cosim.coupling_interface import CouplingInterface

    try:
        time.sleep(0.3)  # Let solver A bind first

        mesh, facet_tags = _create_mesh_and_tags()
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

        cosim = CouplingInterface(
            name="SolverB",
            partner_name="SolverA",
            role="connect",
            endpoint=endpoint,
            timeout_ms=15_000,
        )

        cosim.register_interface(mesh, facet_tags, marker_id=1, function_space=V)

        # Import temperature
        temperature = dolfinx.fem.Function(V)
        cosim.import_data("TemperatureField", temperature)

        # Verify we received temperature data
        bd_vals = cosim._extractor.extract_boundary_values(temperature)
        results["solver_b_received_temp"] = bd_vals.copy()

        # Export displacement
        displacement = dolfinx.fem.Function(V)
        displacement.interpolate(lambda x: 0.001 * x[0])
        cosim.export_data("DisplacementField", displacement)

        cosim.advance_in_time()
        results["solver_b_step"] = cosim.step_count

        cosim.disconnect()
        results["solver_b_ok"] = True

    except Exception as e:
        results["solver_b_error"] = str(e)
        import traceback
        results["solver_b_tb"] = traceback.format_exc()


class TestCouplingIntegration:
    """Full integration test of two coupled solvers."""

    def test_two_solver_coupling(self):
        """Run two solvers in threads and verify data exchange."""
        results: dict = {}
        endpoint = "tcp://127.0.0.1:15560"

        t_a = threading.Thread(
            target=_run_solver_a, args=(results, endpoint)
        )
        t_b = threading.Thread(
            target=_run_solver_b, args=(results, endpoint)
        )

        t_a.start()
        t_b.start()
        t_a.join(timeout=30)
        t_b.join(timeout=30)

        # Check no errors
        if "solver_a_error" in results:
            pytest.fail(
                f"Solver A failed: {results['solver_a_error']}\n"
                f"{results.get('solver_a_tb', '')}"
            )
        if "solver_b_error" in results:
            pytest.fail(
                f"Solver B failed: {results['solver_b_error']}\n"
                f"{results.get('solver_b_tb', '')}"
            )

        assert results.get("solver_a_ok") is True
        assert results.get("solver_b_ok") is True

        # Both should have advanced one step
        assert results["solver_a_step"] == 1
        assert results["solver_b_step"] == 1

        # Solver B should have received temperature data
        temp_vals = results["solver_b_received_temp"]
        assert len(temp_vals) > 0
        # Temperature was 100 + x*50, at y=1 boundary
        # All values should be in [100, 150]
        assert np.all(temp_vals >= 100.0 - 1e-10)
        assert np.all(temp_vals <= 150.0 + 1e-10)
