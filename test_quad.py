import numpy as np
import dolfinx
from mpi4py import MPI
import basix
import basix.ufl

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 1, 1)

# scalar
e_scalar = basix.ufl.quadrature_element(basix.CellType.triangle, degree=2)
V_scalar = dolfinx.fem.functionspace(mesh, e_scalar)

# tensor
e_tensor = basix.ufl.quadrature_element(basix.CellType.triangle, degree=2, value_shape=(2, 2))
V_tensor = dolfinx.fem.functionspace(mesh, e_tensor)

print("SCALAR:")
print("dofmap num dofs per cell:", V_scalar.dofmap.dof_layout.num_dofs)
print("index_map_bs:", V_scalar.dofmap.index_map_bs)
f_s = dolfinx.fem.Function(V_scalar)
print("function array len:", len(f_s.x.array))
print("cell_dofs(0):", V_scalar.dofmap.cell_dofs(0))

print("\nTENSOR:")
print("dofmap num dofs per cell:", V_tensor.dofmap.dof_layout.num_dofs)
print("index_map_bs:", V_tensor.dofmap.index_map_bs)
f_t = dolfinx.fem.Function(V_tensor)
print("function array len:", len(f_t.x.array))
print("cell_dofs(0):", V_tensor.dofmap.cell_dofs(0))
