"""Real heat-conduction and thermoelasticity subdomains (self-contained).

Used by the partitioned thermo-mechanical example. Each subdomain performs a
genuine FEniCSx PDE solve; coupling happens only through interface values, so
the same objects drive both the in-process test and the ZeroMQ scripts.
"""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import default_scalar_type as _ds
from dolfinx import fem, geometry, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI


def _eval_scalar(fn, mesh_, x, y):
    """Evaluate a scalar/vector Function at the point (x, y)."""
    tdim = mesh_.topology.dim
    tree = geometry.bb_tree(mesh_, tdim)
    p = np.array([[x, y, 0.0]], dtype=np.float64)
    cand = geometry.compute_collisions_points(tree, p)
    cells = geometry.compute_colliding_cells(mesh_, cand, p)
    cell = cells.links(0)[0]
    return fn.eval(p, [cell])


class HeatSubdomain:
    """Steady heat conduction on a rectangle [x0, x1] x [0, 1] (k = 1).

    Dirichlet temperature is imposed on the left (x = x0) and right (x = x1)
    edges; these are the partitioned-coupling interface values.
    """

    def __init__(self, x0: float, x1: float, n: int = 12) -> None:
        self.comm = MPI.COMM_WORLD
        self.x0, self.x1 = x0, x1
        self.domain = mesh.create_rectangle(
            self.comm, [[x0, 0.0], [x1, 1.0]], [n, n], mesh.CellType.triangle
        )
        self.V = fem.functionspace(self.domain, ("Lagrange", 1))
        self.T = fem.Function(self.V, name="temperature")

        self._left_dofs = fem.locate_dofs_geometrical(
            self.V, lambda x: np.isclose(x[0], x0)
        )
        self._right_dofs = fem.locate_dofs_geometrical(
            self.V, lambda x: np.isclose(x[0], x1)
        )

    def solve(self, left_val: float, right_val: float):
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(fem.Constant(self.domain, _ds(0.0)), v) * ufl.dx

        bcs = [
            fem.dirichletbc(_ds(left_val), self._left_dofs, self.V),
            fem.dirichletbc(_ds(right_val), self._right_dofs, self.V),
        ]
        problem = LinearProblem(
            a, L, bcs=bcs, u=self.T,
            petsc_options_prefix="heat_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        problem.solve()
        return self.T

    def eval_at_x(self, x: float, y: float = 0.5) -> float:
        return float(_eval_scalar(self.T, self.domain, x, y)[0])


class ThermoElasticSubdomain:
    """Linear thermoelasticity on [x0, x1] x [0, 1] (plane strain).

    Solves C:(ε(u) - ε_th) in equilibrium with no body force, where the
    thermal strain ε_th = α (T - T_ref) I comes from a prescribed temperature
    field. One edge is clamped; the opposite edge is free.
    """

    def __init__(self, x0: float, x1: float, n: int = 12,
                 E: float = 70e3, nu: float = 0.3,
                 alpha: float = 1e-4, T_ref: float = 0.0) -> None:
        self.comm = MPI.COMM_WORLD
        self.x0, self.x1 = x0, x1
        self.domain = mesh.create_rectangle(
            self.comm, [[x0, 0.0], [x1, 1.0]], [n, n], mesh.CellType.triangle
        )
        self.V = fem.functionspace(self.domain, ("Lagrange", 1, (2,)))
        self.Vt = fem.functionspace(self.domain, ("Lagrange", 1))
        self.u = fem.Function(self.V, name="displacement")
        self.alpha = alpha
        self.T_ref = T_ref
        self.lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu = E / (2 * (1 + nu))

    def _eps(self, w):
        return ufl.sym(ufl.grad(w))

    def _sigma(self, w):
        return (self.lmbda * ufl.tr(self._eps(w)) * ufl.Identity(2)
                + 2 * self.mu * self._eps(w))

    def solve(self, temperature_fn, clamp_x: float):
        Tfield = fem.Function(self.Vt)
        Tfield.interpolate(temperature_fn)
        dT = Tfield - fem.Constant(self.domain, _ds(self.T_ref))
        eps_th = self.alpha * dT * ufl.Identity(2)
        sig_th = (self.lmbda * ufl.tr(eps_th) * ufl.Identity(2)
                  + 2 * self.mu * eps_th)

        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        a = ufl.inner(self._sigma(u), self._eps(v)) * ufl.dx
        L = ufl.inner(sig_th, self._eps(v)) * ufl.dx

        clamp_dofs = fem.locate_dofs_geometrical(
            self.V, lambda x: np.isclose(x[0], clamp_x)
        )
        bc = fem.dirichletbc(np.zeros(2, dtype=_ds), clamp_dofs, self.V)
        problem = LinearProblem(
            a, L, bcs=[bc], u=self.u,
            petsc_options_prefix="mech_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        problem.solve()
        return self.u

    def eval_ux_at_x(self, x: float, y: float = 0.5) -> float:
        return float(_eval_scalar(self.u, self.domain, x, y)[0])
