"""
Example: Abaqus coupling wrapper script.

This script demonstrates how an Abaqus simulation can participate in
a real-time staggered coupling loop using fenicsx-cosim. Since Abaqus
lacks a native Python API for live socket communication during a run,
this wrapper script sits between FEniCSx and Abaqus.

Workflow
--------
1. Creates the exchange directory.
2. Writes mock coordinates (which an Abaqus pre-processor would normally do).
3. Connects to the partner solver (e.g. FEniCSx) via ZMQ.
4. Enters a time-stepping loop:
   a. Waits for FEniCSx to send its boundary values.
   b. Writes them to `.npy` files for Abaqus.
   c. (Mocks) calling Abaqus to run a single step.
   d. (Mocks) reading Abaqus output `.csv` and converting it to `.npy`.
   e. Emits the results to FEniCSx.

Usage
-----
    Terminal 1:  python fenicsx_abaqus_thermal.py
    Terminal 2:  python abaqus_coupling_wrapper.py
"""

import time
import os
import shutil
from pathlib import Path

import numpy as np

from fenicsx_cosim.adapters import AbaqusFileAdapter
from fenicsx_cosim import CouplingInterface


def create_mock_mesh_data(exchange_dir: Path) -> np.ndarray:
    """Mocks an Abaqus pre-processor writing boundary coordinates."""
    # 5 nodes on a 1D interface
    coords = np.zeros((5, 3))
    coords[:, 0] = np.linspace(0, 1, 5)  # X from 0 to 1
    coords[:, 1] = 1.0                   # Y = 1.0
    
    np.save(exchange_dir / "boundary_coords.npy", coords)
    return coords


def main():
    exchange_dir = Path("abaqus_exchange")
    if exchange_dir.exists():
        shutil.rmtree(exchange_dir)
    exchange_dir.mkdir()
    
    print(f"[AbaqusWrapper] Created exchange dir: {exchange_dir.absolute()}")
    
    # 1. Pre-processing: mock Abaqus writing coordinates
    print("[AbaqusWrapper] Extracting boundary coordinates...")
    coords = create_mock_mesh_data(exchange_dir)
    num_nodes = len(coords)
    
    # 2. Setup coupling adapter
    print("[AbaqusWrapper] Initializing adapter and connecting to FEniCSx...")
    adapter = AbaqusFileAdapter(exchange_dir)
    
    cosim = CouplingInterface.from_adapter(
        adapter=adapter,
        name="AbaqusSolver",
        partner_name="FEniCSxSolver",
        role="bind",  # Wrapper acts as server
        endpoint="tcp://*:5556"
    )
    
    # Exchange coordinates with FEniCSx
    cosim.register_adapter_interface()
    
    T_final = 0.5
    dt = 0.1
    t = 0.0
    step = 0
    
    while t < T_final - 1e-10:
        t += dt
        step += 1
        print(f"\n[AbaqusWrapper] === Step {step}, t={t:.2f} ===")
        
        # 3. Wait for FEniCSx to send boundary conditions (TEMPERATURE)
        cosim.import_via_adapter("TEMPERATURE")
        print("  Imported TEMPERATURE from FEniCSx")
        
        # MOCK ABAQUS EXECUTION
        # Real Abaqus wrapper would:
        # a) Generate an Abaqus input file fragment reading TEMPERATURE_in.npy
        # b) Call `os.system("abaqus job=... interact")` or similar
        # c) Extract the ODB results to DISPLACEMENT_out.npy
        print("  (Mocking Abaqus execution taking 1s...)")
        time.sleep(1.0)
        
        # We read the injected temperature to simulate calculating displacement
        temp_in = np.load(exchange_dir / "TEMPERATURE_in.npy")
        
        # Mock Abaqus outputting displacement (expanding outwards)
        disp_out = np.zeros((num_nodes, 3))
        disp_out[:, 1] = temp_in * 0.01  # Y-displacement depends on Temp
        
        np.save(exchange_dir / "DISPLACEMENT_out.npy", disp_out)
        print("  Abaqus step finished. Results written to DISPLACEMENT_out.npy")
        
        # 4. Export Abaqus results back to FEniCSx
        name, _ = cosim._communicator.receive_array(flags=1) # check non-blocking if needed...
        # Wait, actually we just send because FEniCSx is expecting it
        cosim.export_via_adapter("DISPLACEMENT")
        print("  Exported DISPLACEMENT to FEniCSx")
        
        # 5. Sync
        cosim.advance_adapter()
        print(f"  Synchronized (step {cosim.step_count})")

    cosim.disconnect()
    print("\n[AbaqusWrapper] Co-simulation complete.")


if __name__ == "__main__":
    main()
