"""
Example: FE² Micro-Worker — simulating Representative Volume Elements (RVEs).

This script simulates a pool of microscopic (RVE) workers in an FE²
multiscale problem. It pulls macroscopic strain tensors from the Master
(``fe2_macro_solver.py``), simulates the damage mechanics or plastic
flow at the microscopic scale, and pushes the homogenized stress tensors
back to the Master.

This corresponds to **Scenario B** and **Test Example 2** from the
Advanced Features Addendum.

Usage
-----
    Terminal 1 (Master):    python fe2_macro_solver.py
    Terminals 2-N (Workers): python fe2_micro_worker.py
"""

import os
import numpy as np

from fenicsx_cosim import CouplingInterface

# ========================================================================
# 1. RVE Solver Placeholder
# ========================================================================
def solve_rve(index_in_batch: int, strain: np.ndarray, metadata: dict) -> np.ndarray:
    """Simulate a Representative Volume Element (RVE) solve.

    In a real FE² system, this function sets up a new FEniCSx mesh (or
    loads a pre-computed grid) representing the microscopic structure
    (e.g., a square with a central void), applies the `strain` tensor
    as periodic boundary conditions, solves for displacement, and computes
    the volume average of the microscopic stress.
    
    Here, we simulate a simple elastic material with damage onset.
    """
    E = metadata.get("E", 210e9)  # Young's Modulus
    nu = 0.3                       # Poisson's ratio
    
    # 2D Plane strain constitutive matrix (Voigt notation)
    C = E / ((1 + nu) * (1 - 2 * nu)) * np.array([
        [1 - nu, nu, 0],
        [nu, 1 - nu, 0],
        [0, 0, (1 - 2 * nu) / 2]
    ])
    
    # Simple scalar damage model based on strain invariants
    strain_norm = np.linalg.norm(strain)
    damage = 0.0
    if strain_norm > 0.005:  # Arbitrary yield criteria
        damage = 1.0 - (0.005 / strain_norm)
        # Cap damage to avoid zero stiffness
        damage = min(damage, 0.99)
        print(f"    RVE {index_in_batch:03d} yielding! (Damage: {damage:.2f})")
    
    # Homogenized stress = (1 - D) * C * strain
    stress = (1.0 - damage) * (C @ strain.T).T
    
    # Simulate computation time of an RVE
    # time.sleep(0.01)
    
    return stress

# ========================================================================
# 2. Worker Setup & Loop
# ========================================================================
# Get the master's IP from the environment (default to localhost for local runs)
master_ip = os.environ.get("COSIM_MASTER_IP", "localhost")

print(f"[FE2_Worker] Starting Scatter/Gather Worker...")
cosim_worker = CouplingInterface(
    name="MicroWorker",
    role="Worker",
    topology="scatter-gather",
    push_endpoint=f"tcp://{master_ip}:5557",  # Connect to Master's Pull
    pull_endpoint=f"tcp://{master_ip}:5556",  # Connect to Master's Push
)

print(f"[FE2_Worker] Connected to Master.")
print(f"[FE2_Worker] Listening for macroscopic strains...")

# Start the continuous work loop
processed_count = cosim_worker._sg_communicator.work_loop(solve_rve)

# The work loop exits when the Master sends a SHUTDOWN_SIGNAL
print(f"\n[FE2_Worker] Received shutdown signal.")
print(f"[FE2_Worker] RVEs solved this session: {processed_count}")

cosim_worker.disconnect()
print("[FE2_Worker] Exit.")
