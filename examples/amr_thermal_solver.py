"""
Example: AMR Thermal Solver — demonstrating Adaptive Mesh Refinement coupling.

This script simulates the thermal side of a thermo-mechanical coupling
where the thermal solver uses Adaptive Mesh Refinement (AMR) to
refine the mesh near regions of high temperature gradient, while the
mechanical solver maintains a static coarse mesh.

This corresponds to **Scenario A** and **Test Example 1 (Breathing
Cylinder)** from the Advanced Features Addendum.

Usage
-----
    Terminal 1:  python amr_thermal_solver.py
    Terminal 2:  python amr_mechanical_solver.py

The thermal solver:
  1. Solves a heat equation on a domain that gets AMR-refined as the
     heat pulse propagates.
  2. Calls ``cosim.update_interface_geometry()`` whenever the mesh changes.
  3. The DynamicMapper handles re-interpolation onto the partner's
     static mesh automatically.
"""

import dolfinx
import dolfinx.fem
import dolfinx.mesh
import numpy as np
from mpi4py import MPI

from fenicsx_cosim import CouplingInterface

# ========================================================================
# 1. FEniCSx Mesh & Function Space Setup
# ========================================================================
comm = MPI.COMM_WORLD
mesh = dolfinx.mesh.create_unit_square(comm, 10, 10)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

# ========================================================================
# 2. Define the Coupling Boundary (top edge, y = 1)
# ========================================================================
tdim = mesh.topology.dim
fdim = tdim - 1


def top_boundary(x):
    return np.isclose(x[1], 1.0)


def create_facet_tags(mesh):
    """Create facet tags marking the top boundary with marker_id=1."""
    boundary_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, top_boundary
    )
    return dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim - 1,
        boundary_facets,
        np.full(len(boundary_facets), 1, dtype=np.int32),
    )


facet_tags = create_facet_tags(mesh)

# ========================================================================
# 3. Set Up Co-Simulation
# ========================================================================
cosim = CouplingInterface(
    name="AMR_ThermalSolver",
    partner_name="StaticMechanicalSolver",
    connection_type="tcp",
)
cosim.register_interface(mesh, facet_tags, marker_id=1, function_space=V)

# ========================================================================
# 4. Create functions
# ========================================================================
temperature = dolfinx.fem.Function(V, name="Temperature")
displacement = dolfinx.fem.Function(V, name="Displacement")

# Initial temperature: uniform baseline
temperature.interpolate(lambda x: 300.0 + 100.0 * x[1])

from dolfinx import io
xdmf = io.XDMFFile(comm, "amr_thermal_output.xdmf", "w")
xdmf.write_mesh(mesh)

# ========================================================================
# 5. AMR + Time-Stepping Loop
# ========================================================================
t = 0.0
T_final = 0.5
dt = 0.1
step = 0

# AMR parameters
refine_threshold = 0.3  # Refine when t > this value (simulating error)
has_refined = False

print(f"[AMR_Thermal] Starting co-simulation — T_final={T_final}, dt={dt}")

while t < T_final - 1e-10:
    t += dt
    step += 1
    print(f"\n[AMR_Thermal] === Step {step}, t = {t:.3f} ===")

    # --- Simulate AMR: refine after a certain time ---
    if t > refine_threshold and not has_refined:
        print("  🔄 Triggering Adaptive Mesh Refinement...")

        # In a real application, you'd evaluate an error estimator
        # and use dolfinx.mesh.refine(mesh, cell_markers)
        # Here we simulate by creating a finer mesh
        mesh = dolfinx.mesh.create_unit_square(comm, 20, 20)
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
        temperature = dolfinx.fem.Function(V, name="Temperature")
        displacement = dolfinx.fem.Function(V, name="Displacement")

        facet_tags = create_facet_tags(mesh)

        # Crucial API Call: Tell the coupling package the mesh changed
        cosim.update_interface_geometry(
            mesh, facet_tags, marker_id=1, function_space=V
        )

        has_refined = True
        xdmf.close()
        xdmf = io.XDMFFile(comm, f"amr_thermal_output_refined.xdmf", "w")
        xdmf.write_mesh(mesh)
        print(f"  ✅ Mesh refined — now {mesh.topology.index_map(tdim).size_local} cells")
    else:
        # Crucial API Call: Perform the AMR negotiation handshake
        # to confirm to the partner that we did NOT refine.
        cosim.check_mesh_update()

    # --- "Solve" the thermal problem ---
    temperature.interpolate(
        lambda x, _t=t: 300.0 + 100.0 * x[1] + 20.0 * np.sin(2 * np.pi * _t)
    )
    print(
        f"  Temperature range: [{temperature.x.array.min():.1f}, "
        f"{temperature.x.array.max():.1f}]"
    )

    # --- Export temperature to mechanical solver ---
    cosim.export_data("TemperatureField", temperature)
    print("  Exported TemperatureField")

    # --- Import displacement from mechanical solver ---
    cosim.import_data("DisplacementField", displacement)
    print("  Imported DisplacementField")

    # --- Write Results to XDMF ---
    xdmf.write_function(temperature, t)
    xdmf.write_function(displacement, t)

    # --- Synchronize ---
    cosim.advance_in_time()
    print(f"  Synchronized (step {cosim.step_count})")

# ========================================================================
# 6. Teardown
# ========================================================================
cosim.disconnect()
xdmf.close()
print(f"\n[AMR_Thermal] Co-simulation complete after {step} steps.")
