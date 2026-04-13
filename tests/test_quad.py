import pytest


dolfinx = pytest.importorskip("dolfinx")
basix = pytest.importorskip("basix")
pytest.importorskip("basix.ufl")
MPI = pytest.importorskip("mpi4py").MPI


def test_quadrature_functionspace_layout():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 1, 1)

    scalar_element = basix.ufl.quadrature_element(basix.CellType.triangle, degree=2)
    scalar_space = dolfinx.fem.functionspace(mesh, scalar_element)

    tensor_element = basix.ufl.quadrature_element(
        basix.CellType.triangle, degree=2, value_shape=(2, 2)
    )
    tensor_space = dolfinx.fem.functionspace(mesh, tensor_element)

    scalar_function = dolfinx.fem.Function(scalar_space)
    tensor_function = dolfinx.fem.Function(tensor_space)

    assert scalar_space.dofmap.dof_layout.num_dofs > 0
    assert tensor_space.dofmap.dof_layout.num_dofs > 0
    assert len(scalar_function.x.array) > 0
    assert len(tensor_function.x.array) > 0
