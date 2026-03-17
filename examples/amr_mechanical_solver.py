"""
Example: Static Mechanical Solver — partner for the AMR thermal solver.

This script maintains a static (non-refined) mesh and receives the
temperature field from the AMR thermal solver. It demonstrates how
the DynamicMapper handles re-interpolation seamlessly when the partner
mesh changes.

Usage
-----
    Terminal 1:  python amr_thermal_solver.py
    Terminal 2:  python amr_mechanical_solver.py
"""

import dolfinx
import dolfinx.fem
import dolfinx.mesh
import numpy as np
from mpi4py import MPI

from fenicsx_cosim import CouplingInterface

# ========================================================================
# 1. FEniCSx Mesh & Function Space Setup (static, coarse mesh)
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


boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top_boundary)
facet_tags = dolfinx.mesh.meshtags(
    mesh, fdim, boundary_facets,
    np.full(len(boundary_facets), 1, dtype=np.int32),
)

# ========================================================================
# 3. Set Up Co-Simulation
# ========================================================================
cosim = CouplingInterface(
    name="StaticMechanicalSolver",
    partner_name="AMR_ThermalSolver",
    connection_type="tcp",
)
cosim.register_interface(mesh, facet_tags, marker_id=1, function_space=V)

# ========================================================================
# 4. Create functions
# ========================================================================
temperature = dolfinx.fem.Function(V, name="Temperature")
displacement = dolfinx.fem.Function(V, name="Displacement")

alpha = 1.2e-5  # Thermal expansion coefficient

from dolfinx import io
xdmf = io.XDMFFile(comm, "amr_mechanical_output.xdmf", "w")
xdmf.write_mesh(mesh)

# ========================================================================
# 5. Time-Stepping Loop
# ========================================================================
t = 0.0
T_final = 0.5
dt = 0.1
step = 0

print(f"[StaticMech] Starting co-simulation — T_final={T_final}, dt={dt}")

while t < T_final - 1e-10:
    t += dt
    step += 1
    print(f"\n[StaticMech] === Step {step}, t = {t:.3f} ===")

    # First check for any AMR updates from the partner
    cosim.check_mesh_update()

    # --- Import temperature from thermal solver ---
    cosim.import_data("TemperatureField", temperature)
    print("  Imported TemperatureField")

    # --- Compute thermal displacement ---
    T_ref = 300.0
    bd = cosim.extractor.boundary_data
    temp_values = temperature.x.array[bd.dof_indices]
    disp_values = alpha * (temp_values - T_ref)
    displacement.x.array[bd.dof_indices] = disp_values

    print(
        f"  Displacement range: [{disp_values.min():.6e}, "
        f"{disp_values.max():.6e}]"
    )

    # --- Export displacement to thermal solver ---
    cosim.export_data("DisplacementField", displacement)
    print("  Exported DisplacementField")

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
print(f"\n[StaticMech] Co-simulation complete after {step} steps.")
