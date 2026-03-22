"""
Example: FEniCSx Thermal Solver coupled with Abaqus.

This script simulates the FEniCSx side of a staggered thermo-mechanical
coupling. It calculates a temperature field and sends it to the Abaqus
wrapper script (`abaqus_coupling_wrapper.py`), then waits for the Abaqus
displacement response.

Usage
-----
    Terminal 1:  python fenicsx_abaqus_thermal.py
    Terminal 2:  python abaqus_coupling_wrapper.py
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
    
    # 1. Mesh & Function Space
    print("[FEniCSxSolver] Creating mesh...")
    mesh = dolfinx.mesh.create_unit_square(comm, 10, 10)
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
    print(f"[FEniCSxSolver] Connecting to AbaqusWrapper at tcp://{host}:5556...")
    
    cosim = CouplingInterface(
        name="FEniCSxThermal",
        partner_name="AbaqusMechanical",
        role="connect",  # FEniCSx connects to the wrapper
        endpoint=f"tcp://{host}:5556",
        enable_mapping=True, 
    )
    
    # Register boundary
    cosim.register_interface(mesh, facet_tags, marker_id=1, function_space=V)
    
    # 4. Create FEniCSx Functions
    temperature = dolfinx.fem.Function(V, name="Temperature")
    
    T_final = 0.5
    dt = 0.1
    t = 0.0
    step = 0
    
    while t < T_final - 1e-10:
        t += dt
        step += 1
        print(f"\n[FEniCSxSolver] === Step {step}, t={t:.2f} ===")
        
        # 1. Solve Thermal Problem (Mocked)
        # Heat concentrates in the center over time
        temperature.interpolate(
            lambda x, _t=t: 300.0 + 50.0 * np.exp(-10.0 * ((x[0] - 0.5)**2)) * (_t / 0.5)
        )
        
        # 2. Export Temperature to Abaqus
        cosim.export_data("TEMPERATURE", temperature)
        max_t = temperature.x.array[cosim._extractor.boundary_dof_indices].max()
        print(f"  Exported TEMPERATURE (max={max_t:.2f})")
        
        # 3. Import Displacement from Abaqus (Abaqus wrapper returns (N, 3))
        # Since we just want to prove mapping works, we accept raw array
        # and print its contents
        name, disp_array = cosim._communicator.receive_array()
        if cosim.mapper is not None:
            disp_array = cosim.mapper.map(disp_array)
            
        print(f"  Imported {name} vector. Max Y-disp: {disp_array[:, 1].max():.4f}")
        
        # 4. Sync
        cosim.advance_in_time()
        print(f"  Synchronized (step {cosim.step_count})")

    cosim.disconnect()
    print("\n[FEniCSxSolver] Co-simulation complete.")


if __name__ == "__main__":
    main()
