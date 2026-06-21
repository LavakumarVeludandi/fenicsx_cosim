"""Macro FE² driver — 2D elasticity whose constitutive law is an RVE pool.

The driver is **transport-agnostic**. It calls a user-supplied
``stress_callback(strains, commit) -> stresses`` that maps a list of per-cell
macro strains (Voigt ``[xx, yy, γxy]``) to per-cell homogenized stresses. Inject
an in-process :class:`~fe2.stateful_pool.StatefulRVEPool` for tests, or wire the
ZeroMQ scatter/gather of ``fenicsx_cosim`` for a distributed run — the macro
physics is identical. This is also the "bring-your-own-solver" rapid-prototyper
pattern: the macro side never needs to know how the micro stress is produced.

Macro problem: unit-square plate, left edge clamped, right edge pulled in x by
``applied_strain_xx`` over ``n_steps`` increments. One macro quadrature point
per cell (P1 displacement => cell-wise constant strain). The macro tangent is
the fixed elastic ``C_eff`` (modified Newton); each load step iterates micro
*trial* evaluations (``commit=False``) to equilibrium, then commits the micro
state once.
"""

from __future__ import annotations

import basix
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc


def _plane_strain_C(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return np.array(
        [[lam + 2 * mu, lam, 0.0],
         [lam, lam + 2 * mu, 0.0],
         [0.0, 0.0, mu]]
    )


def _voigt_strain(u):
    """2D Voigt strain [xx, yy, γxy] with engineering shear."""
    return ufl.as_vector([u[0].dx(0), u[1].dx(1), u[0].dx(1) + u[1].dx(0)])


class MacroFE2Problem:
    def __init__(self, n: int, E_eff: float, nu_eff: float,
                 applied_strain_xx: float, n_steps: int = 5) -> None:
        self.comm = MPI.COMM_WORLD
        self.domain = mesh.create_unit_square(self.comm, n, n,
                                              mesh.CellType.triangle)
        tdim = self.domain.topology.dim
        self.applied = applied_strain_xx
        self.n_steps = n_steps
        self.converged_flags: list[bool] = []

        self.V = fem.functionspace(self.domain, ("Lagrange", 1, (tdim,)))
        self.u = fem.Function(self.V, name="macro_displacement")
        self.du = fem.Function(self.V)

        # One quadrature point per cell (P1 => constant strain per triangle).
        Wq_e = basix.ufl.quadrature_element(
            self.domain.basix_cell(), value_shape=(3,), degree=0
        )
        self.Wq = fem.functionspace(self.domain, Wq_e)
        self.sig_macro = fem.Function(self.Wq, name="macro_stress")

        self.dx = ufl.Measure(
            "dx", domain=self.domain,
            metadata={"quadrature_degree": 0, "quadrature_scheme": "default"},
        )
        self._vol = self.comm.allreduce(
            fem.assemble_scalar(fem.form(1.0 * self.dx)), op=MPI.SUM
        )

        # Fixed elastic macro tangent (modified Newton).
        C = fem.Constant(self.domain, _plane_strain_C(E_eff, nu_eff))
        v = ufl.TestFunction(self.V)
        u_ = ufl.TrialFunction(self.V)
        self._a = fem.form(
            ufl.inner(_voigt_strain(v), ufl.dot(C, _voigt_strain(u_))) * self.dx
        )
        self._L = fem.form(
            -ufl.inner(_voigt_strain(v), self.sig_macro) * self.dx
        )

        # quadrature evaluation points for strain extraction
        self._quad_points, _ = basix.make_quadrature(
            self.domain.basix_cell(), 0
        )
        self._ncells = self.domain.topology.index_map(tdim).size_local
        self._cells = np.arange(self._ncells, dtype=np.int32)
        self._strain_expr = fem.Expression(_voigt_strain(self.u),
                                            self._quad_points)

        self._setup_bcs()

    def _setup_bcs(self):
        Vx, _ = self.V.sub(0).collapse()

        def left(x):
            return np.isclose(x[0], 0.0)

        def right(x):
            return np.isclose(x[0], 1.0)

        self._left_dofs = fem.locate_dofs_geometrical(self.V, left)
        # (V dofs, Vx dofs) pair for the x-component on the right edge.
        self._right_pair = fem.locate_dofs_geometrical((self.V.sub(0), Vx), right)
        self._Vx = Vx

        # Homogeneous BCs used for the Newton correction (du = 0 on essentials).
        zero_vec = fem.Function(self.V)
        zero_x = fem.Function(Vx)
        self._bc_hom = [
            fem.dirichletbc(zero_vec, self._left_dofs),
            fem.dirichletbc(zero_x, self._right_pair, self.V.sub(0)),
        ]

    def _apply_essential(self, ux_value: float):
        """Write the prescribed displacement directly into the solution."""
        self.u.x.array[self._left_dofs] = 0.0
        self.u.x.array[self._right_pair[0]] = ux_value

    def _extract_strains(self) -> list[np.ndarray]:
        vals = self._strain_expr.eval(self.domain, self._cells)
        vals = vals.reshape(self._ncells, 3)
        return [vals[i].copy() for i in range(self._ncells)]

    def _inject_stresses(self, stresses: list[np.ndarray]) -> None:
        arr = np.asarray(stresses, dtype=PETSc.ScalarType).reshape(-1)
        self.sig_macro.x.array[:] = arr

    def solve(self, stress_callback, tol: float = 1e-8, max_it: int = 30):
        """Run the load-stepped FE² solve. Returns volume-averaged macro σxx.

        Modified Newton: the macro tangent ``C_eff`` is constant, so the
        stiffness matrix is assembled and LU-factored **once** and reused for
        every correction across all load steps (the matrix sparsity pattern
        and essential-dof set never change). Each load step writes the
        prescribed displacement into ``u`` and iterates homogeneous Newton
        corrections, probing the micro RVEs with ``commit=False`` until
        equilibrium, then commits the micro state once.
        """
        # Constant tangent: assemble + factor a single time.
        A = petsc.assemble_matrix(self._a, bcs=self._bc_hom)
        A.assemble()
        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        b = petsc.assemble_vector(self._L)  # persistent RHS, reused below

        for step in range(1, self.n_steps + 1):
            ux_target = self.applied * step / self.n_steps
            self._apply_essential(ux_target)

            converged = False
            for it in range(max_it):
                strains = self._extract_strains()
                self._inject_stresses(stress_callback(strains, False))

                with b.localForm() as bloc:
                    bloc.set(0.0)
                petsc.assemble_vector(b, self._L)  # L = -R(u)
                fem.apply_lifting(b, [self._a], bcs=[self._bc_hom])
                b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                              mode=PETSc.ScatterMode.REVERSE)
                petsc.set_bc(b, self._bc_hom)        # homogeneous: du = 0 on BC

                ksp.solve(b, self.du.x.petsc_vec)
                self.du.x.scatter_forward()
                self.u.x.array[:] += self.du.x.array

                if self.du.x.petsc_vec.norm() < tol:
                    converged = True
                    break

            # Commit micro state for this converged load step.
            strains = self._extract_strains()
            self._inject_stresses(stress_callback(strains, True))
            self.converged_flags.append(converged)

        ksp.destroy(); A.destroy(); b.destroy()
        return self._avg_sxx()

    def _avg_sxx(self) -> float:
        return self.comm.allreduce(
            fem.assemble_scalar(fem.form(self.sig_macro[0] * self.dx)),
            op=MPI.SUM,
        ) / self._vol
