import time
import numpy as np
import scipy.sparse as sps

from fenicsx_cosim import CouplingInterface

def solve_shakedown_mock(A_mat: sps.csr_matrix, f_vec: np.ndarray) -> float:
    """
    Mock optimizer representing Gurobi/Mosek.
    In a real scenario, this formulates the kinematic or static shakedown
    optimization problem using the supplied matrices.
    """
    print(f"[Optimizer] Received sparse matrix: {A_mat.shape} with {A_mat.nnz} non-zeros.")
    print(f"[Optimizer] Received load vector: {f_vec.shape}")
    
    print("[Optimizer] Setting up optimization problem in Gurobi/Mosek...")
    time.sleep(1) # Fake solver time
    
    # Mock computation: Norm of load over matrix trace as a dummy safety factor
    trace_val = A_mat.diagonal().sum()
    load_norm = np.linalg.norm(f_vec)
    
    safety_factor = trace_val / (load_norm + 1e-9)
    print(f"[Optimizer] Optimization successful. Kinematic multiplier: {safety_factor:.4f}")
    
    return safety_factor

def run_worker():
    print("[Shakedown/Optimizer] Starting Optimizer Worker...")
    
    # Initialize the coupling interface
    cosim = CouplingInterface(name="Optimizer_Worker", partner_name="FEniCSx_Master", role="connect", endpoint="tcp://localhost:5559")

    print("[Shakedown/Optimizer] Waiting for matrices from FEniCSx...")
    
    # Receive the sparse matrix
    A_scipy = cosim.import_raw("StiffnessMatrix")
    
    # Receive the load vector
    f_vec = cosim.import_raw("LoadVector")

    # Pass to the mock optimization engine
    safety_factor = solve_shakedown_mock(A_scipy, f_vec)
    
    print("[Shakedown/Optimizer] Sending Shakedown Safety Factor back to FEniCSx...")
    safety_factor_array = np.array([safety_factor], dtype=np.float64)
    cosim.export_raw("SafetyFactor", safety_factor_array)

    cosim._communicator.close()
    print("[Shakedown/Optimizer] Done.")

if __name__ == "__main__":
    run_worker()
