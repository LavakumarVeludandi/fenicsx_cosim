"""Self-contained 2D plane-strain RVE solver for FE² homogenization.

Microstructure: a stiff (effectively elastic) circular inclusion embedded in
an elastoplastic matrix obeying **flow-theory J2 plasticity** (von Mises,
linear isotropic hardening, radial-return integration).

Boundary conditions: **KUBC** — the macro strain ``ε̄`` is imposed as the
affine Dirichlet displacement ``u = ε̄ · x`` on the whole RVE boundary. KUBC
needs no multi-point constraints, so a plain custom Newton solve suffices.
(PBC is the milestone-2 accuracy upgrade and *would* require the MPC solver.)

The solver is **stateful**: each :meth:`RVESolver.solve` call advances the
internal plastic state (stress + cumulative plastic strain) from the previous
converged increment, which is exactly the path-dependence that makes FE²
necessary instead of a precomputed ``C_eff``.

Conventions
-----------
Macro strain / stress are exchanged in Voigt form ``[xx, yy, xy]`` with the
**engineering shear** ``γ_xy = 2 ε_xy``. Internally, tensors are carried as a
4-vector ``[xx, yy, zz, xy]`` (plane strain has ``σ_zz ≠ 0``) using the true
(tensorial) shear component, following Bleyer's elastoplasticity formulation.
"""

from __future__ import annotations

import basix
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc


def _ppos(x):
    """Macaulay bracket <x>+ = (x + |x|) / 2."""
    return (x + abs(x)) / 2.0


def _as_3d_tensor(v):
    """4-vector [xx, yy, zz, xy] -> symmetric 3x3 tensor (true shear)."""
    return ufl.as_tensor(
        [[v[0], v[3], 0.0],
         [v[3], v[1], 0.0],
         [0.0, 0.0, v[2]]]
    )


class RVESolver:
    """Stateful two-phase J2 RVE under KUBC.

    Parameters
    ----------
    n : int
        Cells per edge of the unit-square RVE mesh.
    E_matrix, nu_matrix : float
        Matrix elastic moduli.
    sigma_y : float
        Matrix initial yield stress (von Mises). Set huge to force elasticity.
    H : float
        Matrix linear isotropic hardening modulus.
    E_incl, nu_incl : float
        Inclusion elastic moduli (kept elastic via a very high yield stress).
    incl_radius : float
        Inclusion radius (centred at (0.5, 0.5)). ``0`` => homogeneous matrix.
    degree : int
        Displacement element degree (default 2).
    """

    def __init__(
        self,
        n: int = 12,
        E_matrix: float = 70.0e3,
        nu_matrix: float = 0.3,
        sigma_y: float = 250.0,
        H: float = 1.0e3,
        E_incl: float = 210.0e3,
        nu_incl: float = 0.2,
        incl_radius: float = 0.25,
        degree: int = 2,
    ) -> None:
        self.comm = MPI.COMM_WORLD
        self.domain = mesh.create_unit_square(
            self.comm, n, n, mesh.CellType.triangle
        )
        tdim = self.domain.topology.dim
        self._deg_q = 2 * (degree - 1) if degree > 1 else 1

        # --- material parameter fields (DG0, two-phase) ------------------
        DG0 = fem.functionspace(self.domain, ("DG", 0))
        self.mu = fem.Function(DG0, name="mu")
        self.lmbda = fem.Function(DG0, name="lmbda")
        self.sig0 = fem.Function(DG0, name="sig0")
        self.Hmod = fem.Function(DG0, name="H")

        ncells = self.domain.topology.index_map(tdim).size_local + \
            self.domain.topology.index_map(tdim).num_ghosts
        cell_ids = np.arange(ncells, dtype=np.int32)
        midpoints = mesh.compute_midpoints(self.domain, tdim, cell_ids)
        r = np.sqrt(
            (midpoints[:, 0] - 0.5) ** 2 + (midpoints[:, 1] - 0.5) ** 2
        )
        in_incl = r < incl_radius

        def _lame(E, nu):
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))
            return lam, mu

        lam_m, mu_m = _lame(E_matrix, nu_matrix)
        lam_i, mu_i = _lame(E_incl, nu_incl)

        self.mu.x.array[:] = np.where(in_incl, mu_i, mu_m)
        self.lmbda.x.array[:] = np.where(in_incl, lam_i, lam_m)
        # Inclusion kept elastic via an enormous yield stress.
        self.sig0.x.array[:] = np.where(in_incl, 1.0e30, sigma_y)
        self.Hmod.x.array[:] = np.where(in_incl, 0.0, H)

        # --- function spaces --------------------------------------------
        self.V = fem.functionspace(self.domain, ("Lagrange", degree, (tdim,)))
        We = basix.ufl.quadrature_element(
            self.domain.basix_cell(), value_shape=(4,), degree=self._deg_q
        )
        W0e = basix.ufl.quadrature_element(
            self.domain.basix_cell(), value_shape=(), degree=self._deg_q
        )
        self.W = fem.functionspace(self.domain, We)
        self.W0 = fem.functionspace(self.domain, W0e)

        # --- state -------------------------------------------------------
        self.sig = fem.Function(self.W, name="stress")
        self.p = fem.Function(self.W0, name="cumulative_plastic_strain")
        self.u = fem.Function(self.V, name="total_displacement")
        self.du = fem.Function(self.V, name="iteration_correction")
        self.Du = fem.Function(self.V, name="increment")

        # quadrature evaluation points (reference cell)
        self._quad_points, _ = basix.make_quadrature(
            self.domain.basix_cell(), self._deg_q
        )
        self._cells = np.arange(
            self.domain.topology.index_map(tdim).size_local, dtype=np.int32
        )

        self.dx = ufl.Measure(
            "dx",
            domain=self.domain,
            metadata={"quadrature_degree": self._deg_q,
                      "quadrature_scheme": "default"},
        )
        self._vol = self.comm.allreduce(
            fem.assemble_scalar(fem.form(1.0 * self.dx)), op=MPI.SUM
        )

        # --- KUBC boundary dofs -----------------------------------------
        self.domain.topology.create_connectivity(tdim - 1, tdim)
        boundary_facets = mesh.exterior_facet_indices(self.domain.topology)
        self._boundary_dofs = fem.locate_dofs_topological(
            self.V, tdim - 1, boundary_facets
        )
        self._u_bc = fem.Function(self.V, name="kubc_value")

        self._build_forms()
        self._last_hill_mandel = None

    # ------------------------------------------------------------------
    def _eps(self, v):
        e = ufl.sym(ufl.grad(v))
        return ufl.as_vector([e[0, 0], e[1, 1], 0.0, e[0, 1]])

    def _sigma_el(self, eps_v):
        e = _as_3d_tensor(eps_v)
        return self.lmbda * ufl.tr(e) * ufl.Identity(3) + 2 * self.mu * e

    def _proj_sig(self, deps, old_sig, old_p):
        """Radial return: trial stress from strain increment -> projected."""
        sig_n = _as_3d_tensor(old_sig)
        sig_elas = sig_n + self._sigma_el(deps)
        s = ufl.dev(sig_elas)
        sig_eq = ufl.sqrt(3.0 / 2.0 * ufl.inner(s, s))
        f_elas = sig_eq - self.sig0 - self.Hmod * old_p
        dp = _ppos(f_elas) / (3.0 * self.mu + self.Hmod)
        n_elas = s / (sig_eq + 1.0e-30) * _ppos(f_elas) / (f_elas + 1.0e-30)
        beta = 3.0 * self.mu * dp / (sig_eq + 1.0e-30)
        new_sig = sig_elas - beta * s
        new_sig_v = ufl.as_vector(
            [new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 1]]
        )
        n_elas_v = ufl.as_vector(
            [n_elas[0, 0], n_elas[1, 1], n_elas[2, 2], n_elas[0, 1]]
        )
        return new_sig_v, n_elas_v, beta, dp

    def _sigma_tang(self, e, n_elas, beta):
        N = _as_3d_tensor(n_elas)
        et = _as_3d_tensor(e)
        return (
            self._sigma_el(e)
            - 3.0 * self.mu
            * (3.0 * self.mu / (3.0 * self.mu + self.Hmod) - beta)
            * ufl.inner(N, et) * N
            - 2.0 * self.mu * beta * ufl.dev(et)
        )

    def _build_forms(self):
        v = ufl.TestFunction(self.V)
        u_ = ufl.TrialFunction(self.V)

        deps = self._eps(self.Du)
        self._sig_new, self._n_elas, self._beta, self._dp = self._proj_sig(
            deps, self.sig, self.p
        )

        a = ufl.inner(
            _as_3d_tensor(self._eps(v)),
            self._sigma_tang(self._eps(u_), self._n_elas, self._beta),
        ) * self.dx
        # Residual uses the *projected* trial stress (a function of Du),
        # so each Newton reassembly reflects the return-mapping update.
        L = -ufl.inner(
            _as_3d_tensor(self._eps(v)), _as_3d_tensor(self._sig_new)
        ) * self.dx

        self._a_form = fem.form(a)
        self._L_form = fem.form(L)

    def _interp_quadrature(self, ufl_expr, fn):
        expr = fem.Expression(ufl_expr, self._quad_points)
        vals = expr.eval(self.domain, self._cells)
        fn.x.array[:] = vals.reshape(-1)

    # ------------------------------------------------------------------
    def solve(self, eps_bar_voigt, tol: float = 1.0e-8, max_it: int = 50,
              commit: bool = True):
        """Solve the RVE under macro strain ``eps_bar_voigt`` = [xx, yy, γxy].

        Returns the homogenized macro stress ``σ̄`` as Voigt ``[xx, yy, xy]``.

        Parameters
        ----------
        commit : bool
            If ``True`` (default) the converged plastic state is committed
            (the RVE advances along its load path). If ``False`` the call is a
            *trial* evaluation from the last committed state and the internal
            state is rolled back — required when a macro Newton loop probes
            multiple trial strains within one load step.
        """
        exx, eyy, gxy = (float(c) for c in eps_bar_voigt)
        exy = 0.5 * gxy  # tensorial shear

        # KUBC affine displacement u = ε̄ · x
        def kubc_expr(x):
            ux = exx * x[0] + exy * x[1]
            uy = exy * x[0] + eyy * x[1]
            return np.vstack([ux, uy])

        self._u_bc.interpolate(kubc_expr)

        # Increment of boundary displacement to reach the new total state.
        self.Du.x.array[:] = 0.0
        bc_vals = fem.Function(self.V)
        bc_vals.x.array[:] = self._u_bc.x.array - self.u.x.array
        bc = fem.dirichletbc(bc_vals, self._boundary_dofs)

        A = petsc.assemble_matrix(self._a_form, bcs=[bc])
        A.assemble()
        b = petsc.assemble_vector(self._L_form)
        fem.apply_lifting(b, [self._a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver = PETSc.KSP().create(self.comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")

        nRes0 = b.norm()
        nRes0 = nRes0 if nRes0 > 0 else 1.0
        nRes = nRes0
        it = 0
        # Homogeneous BC for subsequent Newton corrections.
        zero_bc = fem.dirichletbc(fem.Function(self.V), self._boundary_dofs)

        while nRes / nRes0 > tol and it < max_it:
            solver.solve(b, self.du.x.petsc_vec)
            self.du.x.scatter_forward()
            self.Du.x.array[:] += self.du.x.array

            # recompute residual + tangent at updated increment
            A.zeroEntries()
            petsc.assemble_matrix(A, self._a_form, bcs=[zero_bc])
            A.assemble()
            with b.localForm() as b_loc:
                b_loc.set(0.0)
            petsc.assemble_vector(b, self._L_form)
            fem.apply_lifting(b, [self._a_form], bcs=[[zero_bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                          mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [zero_bc])
            solver.setOperators(A)
            nRes = b.norm()
            it += 1

        # Evaluate the converged trial stress field (needed for homogenization
        # whether or not we commit). proj_sig reads the *committed* self.sig /
        # self.p, so self.sig still holds the old state during this eval.
        old_sig = self.sig.x.array.copy()
        self._interp_quadrature(self._sig_new, self.sig)
        self._last_eps_bar = np.array([exx, eyy, 0.0, exy])
        sigma_bar = self._homogenized_stress()

        if commit:
            self.u.x.array[:] += self.Du.x.array
            p_new = fem.Function(self.W0)
            self._interp_quadrature(self.p + self._dp, p_new)
            self.p.x.array[:] = p_new.x.array
        else:
            self.sig.x.array[:] = old_sig  # roll back trial stress

        solver.destroy()
        A.destroy()
        b.destroy()

        return sigma_bar

    # ------------------------------------------------------------------
    def _homogenized_stress(self):
        comps = []
        for i in range(4):
            val = self.comm.allreduce(
                fem.assemble_scalar(fem.form(self.sig[i] * self.dx)),
                op=MPI.SUM,
            )
            comps.append(val / self._vol)
        sxx, syy, szz, sxy = comps
        self._last_sigma_bar = np.array([sxx, syy, szz, sxy])
        # Voigt [xx, yy, xy] for the 2D macro solver (szz is plane-strain only)
        return np.array([sxx, syy, sxy])

    def hill_mandel_residual(self) -> float:
        """Relative Hill-Mandel residual |⟨σ:ε⟩ - σ̄:ε̄| / |σ̄:ε̄| for last solve."""
        micro_work = self.comm.allreduce(
            fem.assemble_scalar(
                fem.form(
                    ufl.inner(_as_3d_tensor(self.sig),
                              _as_3d_tensor(self._eps(self.u))) * self.dx
                )
            ),
            op=MPI.SUM,
        ) / self._vol
        sb = _as_3d_tensor(ufl.as_vector(list(self._last_sigma_bar)))
        eb = _as_3d_tensor(ufl.as_vector(list(self._last_eps_bar)))
        macro_work = float(
            self._last_sigma_bar[0] * self._last_eps_bar[0]
            + self._last_sigma_bar[1] * self._last_eps_bar[1]
            + self._last_sigma_bar[2] * self._last_eps_bar[2]
            + 2.0 * self._last_sigma_bar[3] * self._last_eps_bar[3]
        )
        denom = abs(macro_work) if abs(macro_work) > 1e-30 else 1.0
        return abs(micro_work - macro_work) / denom
