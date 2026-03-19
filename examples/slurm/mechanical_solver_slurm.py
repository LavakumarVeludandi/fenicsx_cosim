"""
Example: Mechanical Solver (Solver B) — demonstrating fenicsx-cosim coupling.

This script simulates the mechanics side of a thermo-mechanical coupling.
It should be run in one terminal while ``thermal_solver.py`` runs in
another.

Usage
-----
    Terminal 1:  python thermal_solver.py
    Terminal 2:  python mechanical_solver.py

The mechanical solver:
  1. Receives the temperature field from the thermal solver on the top
     boundary (y=1).
  2. Computes a thermal displacement (simplified).
  3. Exports the displacement back to the thermal solver.
"""

import os
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
mesh = dolfinx.mesh.create_unit_square(comm, 20, 20)
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
master_ip = os.environ.get("COSIM_MASTER_IP", "localhost")
cosim = CouplingInterface(
    name="MechanicalSolver",
    partner_name="ThermalSolver",
    connection_type="tcp",
    endpoint=f"tcp://{master_ip}:5555"
)
cosim.register_interface(mesh, facet_tags, marker_id=1, function_space=V)

# ========================================================================
# 4. Create functions for the simulation
# ========================================================================
temperature = dolfinx.fem.Function(V, name="Temperature")
displacement = dolfinx.fem.Function(V, name="Displacement")

from dolfinx import io
xdmf = io.XDMFFile(comm, "mechanical_output.xdmf", "w")
xdmf.write_mesh(mesh)

# Thermal expansion coefficient
alpha = 1.2e-5  # 1/K

# ========================================================================
# 5. Time-Stepping Loop
# ========================================================================
t = 0.0
T_final = 0.5
dt = 0.1
step = 0

print(f"[MechanicalSolver] Starting co-simulation — T_final={T_final}, dt={dt}")

while t < T_final - 1e-10:
    t += dt
    step += 1
    print(f"\n[MechanicalSolver] === Step {step}, t = {t:.3f} ===")

    # --- Import temperature from thermal solver ---
    cosim.import_data("TemperatureField", temperature)
    print("  Imported TemperatureField")

    # --- "Solve" the mechanical problem ---
    # (Simplified: displacement = alpha * (T - T_ref) * L)
    # In a real application, this would be a FEniCSx elasticity solve
    # using the temperature as a body load.
    T_ref = 300.0
    bd = cosim._extractor.boundary_data
    temp_values = temperature.x.array[bd.dof_indices]
    disp_values = alpha * (temp_values - T_ref)
    displacement.x.array[bd.dof_indices] = disp_values

    print(f"  Displacement range: [{disp_values.min():.6e}, "
          f"{disp_values.max():.6e}]")

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
print(f"\n[MechanicalSolver] Co-simulation complete after {step} steps.")
