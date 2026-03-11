import time
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl

import scipy.sparse as sps

from fenicsx_cosim import CouplingInterface

def run_master():
    print("[Shakedown/FEniCSx] Starting FEniCSx Master solver...")
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [10, 10])

    # Initialize the coupling interface
    cosim = CouplingInterface(name="FEniCSx_Master", partner_name="Optimizer_Worker", role="bind", endpoint="tcp://*:5559")

    # In a real shakedown analysis, we would compute elastic stresses under multiple load vertices.
    # Here, to prove the sparse matrix transport, we will assemble the global stiffness matrix
    # and extract it as a scipy.sparse CSR matrix, then send it to the optimizer!

    print("[Shakedown/FEniCSx] Assembling sparse stiffness matrix...")
    V = fem.functionspace(domain, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    
    # Assemble FEniCSx matrix
    A_petsc = petsc.assemble_matrix(fem.form(a))
    A_petsc.assemble()
    
    # Convert PETSc matrix to SciPy CSR matrix
    ai, aj, av = A_petsc.getValuesCSR()
    A_scipy = sps.csr_matrix((av, aj, ai), shape=A_petsc.getSize())
    
    print(f"[Shakedown/FEniCSx] Stiffness Matrix shape: {A_scipy.shape}, non-zeros: {A_scipy.nnz}")

    print("[Shakedown/FEniCSx] Exporting Sparse Matrix to Optimizer...")
    # Send the sparse matrix over ZeroMQ!
    cosim.export_raw("StiffnessMatrix", A_scipy)

    # We also send a mock load vector
    f_vec = np.ones(A_scipy.shape[0]) * 0.5
    cosim.export_raw("LoadVector", f_vec)

    print("[Shakedown/FEniCSx] Waiting for Optimizer to compute Shakedown Safety Factor...")
    safety_factor_array = cosim.import_raw("SafetyFactor")
    
    print(f"\n✅ [Shakedown/FEniCSx] Received Shakedown Safety Factor: {safety_factor_array[0]:.4f}")
    
    cosim._communicator.close()
    print("[Shakedown/FEniCSx] Done.")

if __name__ == "__main__":
    run_master()
