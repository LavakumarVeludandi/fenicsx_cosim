"""
Example: FE² Macro-Solver (Master) — demonstrating Multiscale Homogenization.

This script simulates the macroscopic scale of an FE² multiscale problem.
Instead of evaluating a constitutive law directly, it evaluates the
macroscopic strain tensor at every integration (Gauss) point of the mesh,
and then uses the ``ScatterGatherCommunicator`` via ``CouplingInterface``
to dispatch these strain tensors to a pool of microscopic (RVE) workers.

This corresponds to **Scenario B** and **Test Example 2** from the
Advanced Features Addendum.

Usage
-----
    Terminal 1 (Master):    python fe2_macro_solver.py
    Terminals 2-N (Workers): python fe2_micro_worker.py
"""

import time

import dolfinx
import dolfinx.fem
import dolfinx.mesh
import numpy as np
from mpi4py import MPI
import basix
import ufl

from fenicsx_cosim import CouplingInterface

# ========================================================================
# 1. FEniCSx Mesh & Function Space Setup (Macro-scale)
# ========================================================================
comm = MPI.COMM_WORLD
# A simple 2x2 grid = 8 triangular elements
mesh = dolfinx.mesh.create_unit_square(comm, 2, 2)

# Macroscopic displacement field
V_macro = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
u_macro = dolfinx.fem.Function(V_macro, name="MacroDisplacement")

# Create a dummy displacement (e.g., simple tension)
u_macro.interpolate(lambda x: np.vstack((0.1 * x[0], -0.02 * x[1])))

# ========================================================================
# 2. Quadrature Space for Integration Points
# ========================================================================
quad_degree = 2

# To extract strains, we need a Quadrature space that holds 2D symmetric
# tensors (e.g., Voigt notation shape=(3,) or full tensor shape=(2,2)).
# Here we use shape (3,) for [eps_xx, eps_yy, 2*eps_xy]
tensor_shape = (3,)

# Setup basix quadrature element manually as ufl isn't fully integrated here
cell_type = mesh.topology.cell_type
basix_cell = getattr(basix.CellType, cell_type.name)
qe = basix.ufl.quadrature_element(basix_cell, value_shape=tensor_shape, degree=quad_degree)
V_quad = dolfinx.fem.functionspace(mesh, qe)

macro_strain = dolfinx.fem.Function(V_quad, name="MacroStrain")
homogenized_stress = dolfinx.fem.Function(V_quad, name="HomogenizedStress")

# --- Simulate computing the macroscopic strain ---
# (In a real FEniCSx solve, this would be computed via projection
#  or local element assembly: eps = sym(grad(u)))
# We populate it with dummy values for the example
macro_strain.x.array[:] = np.random.rand(len(macro_strain.x.array)) * 0.01

# ========================================================================
# 3. Set Up Co-Simulation (FE² Scatter/Gather)
# ========================================================================
print(f"[FE2_Macro] Initializing Scatter/Gather topology...")
cosim_fe2 = CouplingInterface(
    name="MacroSolver",
    role="Master",
    topology="scatter-gather",
    push_endpoint="tcp://*:5556",
    pull_endpoint="tcp://*:5557",
)

# Register the Quadrature space instead of a boundary
cosim_fe2.register_quadrature_space(
    function_space_or_mesh=V_quad,
    tensor_shape=tensor_shape
)

qd = cosim_fe2.quadrature_extractor.quadrature_data
n_cells = qd.num_cells
pts_per_cell = qd.points_per_cell

print(f"[FE2_Macro] System has {n_cells} macro-elements.")
print(f"[FE2_Macro] Expecting {n_cells} RVE solves (one per macro-element).")
print(f"[FE2_Macro] Waiting for workers to spin up (run fe2_micro_worker.py)...")

# Give workers a moment to connect
time.sleep(2.0)

# ========================================================================
# 4. Dispatch and Gather
# ========================================================================
print("\n[FE2_Macro] --- Commencing FE² Dispatch ---")

# Step 1: Scatter strains to the pool of RVE workers
materials = [{"material": "matrix", "E": 210e9} for _ in range(n_cells)]
cosim_fe2.scatter_data(
    data_name="StrainTensor",
    function=macro_strain,
    metadata=materials
)

# Step 2: Gather the homogenized stresses from the RVEs
print(f"[FE2_Macro] Waiting for RVE results...")
try:
    cosim_fe2.gather_data(
        data_name="StressTensor",
        function=homogenized_stress,
        n_expected=n_cells
    )
    print("\n[FE2_Macro] ✅ Successfully gathered all homogenized stresses!")
    
    # Inspect one of the element's resulting stresses
    dofs = cosim_fe2.quadrature_extractor.cell_to_dof_map[0]
    cell_stress = homogenized_stress.x.array[dofs].reshape(pts_per_cell, *tensor_shape)
    print(f"[FE2_Macro] Example stress at element 0:\n{cell_stress}")

except TimeoutError as e:
    print(f"\n[FE2_Macro] ❌ Timeout: {e}")
    print("[FE2_Macro] Did you forget to start the worker scripts?")

# ========================================================================
# 5. Teardown
# ========================================================================
# Tell workers to gracefully exit
cosim_fe2._sg_communicator.broadcast_shutdown(5)  # Send a few shutdown signals
cosim_fe2.disconnect()
print("\n[FE2_Macro] Master shut down successfully.")
