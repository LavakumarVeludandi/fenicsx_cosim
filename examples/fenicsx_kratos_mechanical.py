"""
Example: FEniCSx Mechanical Solver coupled with Kratos.

This script simulates the mechanical side of a thermo-mechanical coupling.
It expects a Kratos thermal solver (like kratos_thermal_solver.py)
to be running and acting as the server ("bind" role).

Usage
-----
    Terminal 1:  python kratos_thermal_solver.py
    Terminal 2:  python fenicsx_kratos_mechanical.py
"""

import sys
import numpy as np

import dolfinx
import dolfinx.fem
import dolfinx.mesh
from mpi4py import MPI

from fenicsx_cosim import CouplingInterface


def main():
    comm = MPI.COMM_WORLD
    
    # 1. Mesh & Function Space (Unit Square)
    print("[FEniCSxSolver] Creating mesh...")
    mesh = dolfinx.mesh.create_unit_square(comm, 15, 15)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    
    # 2. Define Coupling Boundary (Top edge, y=1)
    tdim = mesh.topology.dim
    fdim = tdim - 1

    def top_boundary(x):
        return np.isclose(x[1], 1.0)
        
    facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top_boundary)
    facet_tags = dolfinx.mesh.meshtags(
        mesh, fdim, facets, np.full(len(facets), 1, dtype=np.int32)
    )
    
    # 3. Setup Co-Simulation
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    print(f"[FEniCSxSolver] Connecting to Kratos at tcp://{host}:5555...")
    
    cosim = CouplingInterface(
        name="FEniCSxMechanical",
        partner_name="KratosThermal",
        role="connect",  # FEniCSx connects to Kratos
        endpoint=f"tcp://{host}:5555",
        enable_mapping=True, # Handles non-matching nodes between Kratos & FEniCSx
    )
    
    # Register boundary using FEniCSx native method
    cosim.register_interface(mesh, facet_tags, marker_id=1, function_space=V)
    
    # 4. Create FEniCSx Functions
    # Instead of full elasticity vector space, we just use 3 scalar spaces
    # to mimic mapping a 3D Kratos vector (X, Y, Z) to our 2D mesh, 
    # but for simplicity we'll just exchange scalar X displacement
    
    temperature = dolfinx.fem.Function(V, name="Temperature")
    disp_x = dolfinx.fem.Function(V, name="Displacement_X")
    disp_y = dolfinx.fem.Function(V, name="Displacement_Y")
    disp_z = dolfinx.fem.Function(V, name="Displacement_Z")
    
    # Pack into (N, 3) for Kratos
    def get_disp_vector():
        bd_indices = cosim._extractor.boundary_dof_indices
        N = len(bd_indices)
        vec = np.zeros((N, 3), dtype=np.float64)
        vec[:, 0] = disp_x.x.array[bd_indices]
        vec[:, 1] = disp_y.x.array[bd_indices]
        vec[:, 2] = disp_z.x.array[bd_indices]
        return vec
        
    T_final = 0.5
    dt = 0.1
    t = 0.0
    step = 0
    
    while t < T_final - 1e-10:
        t += dt
        step += 1
        print(f"\n[FEniCSxSolver] === Step {step}, t={t:.2f} ===")
        
        # 1. Import Temperature from Kratos
        cosim.import_data("TEMPERATURE", temperature)
        max_t = temperature.x.array[cosim._extractor.boundary_dof_indices].max()
        print(f"  Imported TEMPERATURE (max={max_t:.2f})")
        
        # 2. Solve Mechanical Problem (Mocked)
        # Displacement grows with time and temperature
        disp_x.interpolate(lambda x, _t=t: 0.001 * x[0] * _t)
        disp_y.interpolate(lambda x: 0.002 * x[1])
        
        # 3. Export Displacement Vector to Kratos
        # We manually map the FEniCSx values to an (N,3) array and send it
        disp_vec = get_disp_vector()
        cosim.export_raw("DISPLACEMENT", disp_vec)
        print("  Exported DISPLACEMENT vector")
        
        # 4. Sync
        cosim.advance_in_time()
        print(f"  Synchronized (step {cosim.step_count})")

    cosim.disconnect()
    print("\n[FEniCSxSolver] Co-simulation complete.")

if __name__ == "__main__":
    main()
