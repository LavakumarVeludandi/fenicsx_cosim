"""
Example: Thermal Solver (Solver A) — demonstrating fenicsx-cosim coupling.

This script simulates the thermal side of a thermo-mechanical coupling.
It should be run in one terminal while ``mechanical_solver.py`` runs in
another.

Usage
-----
    Terminal 1:  python thermal_solver.py
    Terminal 2:  python mechanical_solver.py

The thermal solver:
  1. Solves a simple heat equation on a unit square.
  2. Exports the temperature field on the top boundary (y=1) to the
     mechanical solver.
  3. Receives displacement feedback and uses it to update the boundary
     condition (demonstrating two-way coupling).
"""

import dolfinx
import dolfinx.fem
import dolfinx.mesh
import sys
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
host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
endpoint = "tcp://*:5555" if host == "0.0.0.0" else f"tcp://{host}:5555"

cosim = CouplingInterface(
    name="ThermalSolver",
    partner_name="MechanicalSolver",
    role="bind",
    endpoint=endpoint,
)
cosim.register_interface(mesh, facet_tags, marker_id=1, function_space=V)

# ========================================================================
# 4. Create functions for the simulation
# ========================================================================
temperature = dolfinx.fem.Function(V, name="Temperature")
displacement = dolfinx.fem.Function(V, name="Displacement")

# Initial temperature distribution: linear gradient
temperature.interpolate(lambda x: 300.0 + 100.0 * x[1])

from dolfinx import io
xdmf = io.XDMFFile(comm, "thermal_output.xdmf", "w")
xdmf.write_mesh(mesh)

# ========================================================================
# 5. Time-Stepping Loop
# ========================================================================
t = 0.0
T_final = 0.5
dt = 0.1
step = 0

print(f"[ThermalSolver] Starting co-simulation — T_final={T_final}, dt={dt}")

while t < T_final - 1e-10:
    t += dt
    step += 1
    print(f"\n[ThermalSolver] === Step {step}, t = {t:.3f} ===")

    # --- "Solve" the thermal problem ---
    # (In a real application, this would be a FEniCSx PDE solve.)
    # Here we simulate temperature evolution with a simple update.
    temperature.interpolate(
        lambda x, _t=t: 300.0 + 100.0 * x[1] + 20.0 * np.sin(2 * np.pi * _t)
    )
    print(f"  Temperature range: [{temperature.x.array.min():.1f}, "
          f"{temperature.x.array.max():.1f}]")

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
print(f"\n[ThermalSolver] Co-simulation complete after {step} steps.")
