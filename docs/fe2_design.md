# FE² Real Use-Case Design (honest path-dependent homogenization)

Status: in progress. Supersedes the placeholder physics in
`examples/fe2_macro_solver.py` / `examples/fe2_micro_worker.py`
(which faked the RVE with a closed-form damage formula).

## Why FE² is the justified use case

`fenicsx-cosim` co-simulation is only justified when (a) two codes **cannot
be fused** into one program (connector), or (b) the problem is **genuinely
many independent subproblems** (FE²). Two FEniCSx solvers exchanging boundary
fields over a socket is an anti-pattern — you would solve that monolithically.

FE² earns its existence **only if the micro material is path-dependent**.
A linear-elastic RVE collapses to a single precomputed `C_eff`; no workers
needed. Therefore the micro law here is **flow-theory J2 plasticity**
(radial return), not deformation theory and not linear elasticity.

## Locked architecture (Milestone 1)

| Decision | Choice | Rationale |
|---|---|---|
| Micro constitutive | flow-theory J2, isotropic hardening, radial return | path-dependent → FE² genuinely needed |
| RVE boundary cond. | **KUBC** `u = ε̄·x` (Dirichlet) | no MPC → plain `NewtonSolver`. PBC is the M2 accuracy upgrade |
| RVE microstructure | 2D plane-strain two-phase: stiff elastic inclusion + J2 matrix | minimal real heterogeneity |
| Parallelism | **single stateful worker**, RVE state dict keyed by QP index | sacrifice parallelism, not physics. Correct & serial |
| RVE code | self-contained in `examples/fe2/` | keep `fenicsx-cosim` standalone — **no `sdlab` import** |
| Macro driver | 2D elasticity, quad-point strains via `QuadratureExtractor` | exercises scatter/gather API |
| Verification | **Hill-Mandel** `⟨σ:ε⟩_RVE = σ̄:ε̄` + uniaxial response | Voigt/Reuss bounds are linear-only, invalid for J2 |

### Data contract (per macro quadrature point)
- Master → worker: macro strain `ε̄` (Voigt, 2D = `[εxx, εyy, γxy]`), QP index.
- Worker → master: homogenized stress `σ̄` (Voigt). Worker retains the RVE
  plastic state internally, keyed by the QP `index` carried in the message.

### State / parallelism note
The scatter-gather protocol carries `[json_header, binary_payload]` with
**round-robin, no worker affinity**. Path-dependent J2 needs per-QP state
persisted across macro increments. M1 sidesteps this with **one** worker that
owns all RVE states. Scaling to N workers (M2) needs a binary **state frame**
added to the multipart message (`[header, strain, state_in] → [header,
stress, state_out]`, ~30 lines in `scatter_gather_communicator.py`) so workers
stay stateless-by-message and round-robin-safe.

## Milestones

- **M1 (this work):** self-contained J2/KUBC RVE solver + single stateful
  worker + macro FE² driver + Hill-Mandel test. Serial, correct.
- **M2:** binary state-frame → N parallel workers. PBC option for accuracy.
  Consistent macro tangent (Miehe condensation) for fast macro Newton.
- **M3 (connector variant):** Kratos as the microscale RVE solver via the
  existing `KratosAdapter`, FEniCSx at the macroscale.
- **GPU variant:** macro CPU → GPU workers running a **batched constitutive
  surrogate** (JAX/CuPy/PyTorch) trained on RVE responses. NOTE: stock
  DOLFINx 0.10 has **no GPU assembly** (only `la` cast-copy); GPU lives at the
  worker/surrogate layer, never inside the DOLFINx solve.

## Thermal↔mechanical example — DONE (real)
Built `examples/thermomech/`: a genuine partitioned coupling — two FEniCSx
subdomains run an **overlapping-Schwarz** steady heat solve exchanging interface
temperature over fenicsx-cosim (`export_raw`/`import_raw`), converging to the
**exact** linear profile (interface T(1)=50.0, verified analytically), then
subdomain B runs a real **thermoelastic** solve on the converged field (free end
expands outward, u_x<0). Importable physics in `thermomech/subdomains.py`
(`HeatSubdomain`, `ThermoElasticSubdomain`), thin ZeroMQ scripts `thermal_a.py`
/`thermal_b.py`, `run_thermomech.sh`, tests `tests/test_thermomech.py`. Old
`thermal_solver.py`/`mechanical_solver.py` now shim to these. Each script
carries the honesty note: two FEniCSx subdomains would normally be solved
monolithically — partitioning is justified only across different codes.

## Mesh + material middleware (FEniCSx ↔ Kratos) — DONE (bidirectional; linear-elastic family Kratos-load-verified)
`src/fenicsx_cosim/interop/` is the generalized middleware. It is **pure NumPy**
(no dolfinx/Kratos/meshio at import) so it loads in either conda env, and it works
**both directions** — the repeated requirement, not FEniCSx→Kratos only.

- **`NeutralMesh`/`Region`** (`neutral_mesh.py`): code-neutral pivot. 1-based ids
  (gmsh+Kratos convention), named regions, optional `material` per region.
- **`mdpa_io.py`**: writes/reads real Kratos `.mdpa` — registered element/condition
  type names, the per-element Properties id, and named **SubModelParts**. These are
  exactly what meshio's `.msh→.mdpa` drops (it emits geometry name `Triangle2D3`,
  no Properties linkage, no SubModelParts), which is why meshio output is not
  Kratos-loadable and a custom writer is required.
- **`element_map.py`**: family×dim×cell_type ↔ Kratos element/condition name tables.
- **`dolfinx_bridge.py`**: `NeutralMesh ↔ dolfinx` (lazy dolfinx import).
- **`material.py`**: neutral `MaterialModel` ↔ Kratos (element family +
  constitutive_law + Variable names). Curated registry, no silent fallback. The
  Kratos element name encodes kinematics, so the element family is **derived from
  the material**, not hardcoded — `write_mdpa` reads `Region.material` to pick it.
- **sdlab binding** lives in `sdlab_fem` (`constitutive/cosim_material.py`,
  dependency `sdlab_fem → fenicsx_cosim`, keeping cosim standalone/physics-free):
  sdlab ConstitutiveLaw registry name + kwargs ↔ neutral `MaterialModel`, both
  ways. Plane is sourced from the analysis for 3D-capable laws (J2, hyperelastic),
  read from the name for `LinearElasticPlaneStrain/Stress`.

**Proof, not substring checks.** meshio passed string-shaped tests yet Kratos
rejected the result. `tests/_kratos_load_check.py` (run by `test_kratos_load.py`,
base env) is the real gate: for each case Kratos `ModelPartIO` reads the written
`.mdpa` and `ReadMaterialsUtility` reads the written materials.json — both raise
on an unregistered element/law, so a clean load proves the element name (from
`element_map`) and the constitutive-law name (from `material._REGISTRY`) are real.
Asserts SubModelPart + element count + `CONSTITUTIVE_LAW`/`YOUNG_MODULUS`. (Child
process — Kratos segfaults pytest's interpreter on teardown; `element.Info()`/
`law.Info()` segfault this pybind build, so the successful read *is* the
assertion.) Covers three registry rows, each a distinct element family AND law:
`SmallDisplacementElement2D3N`+`LinearElasticPlaneStrain2DLaw` (2D triangle),
`...PlaneStress2DLaw` (2D triangle), `SmallDisplacementElement3D4N`+
`LinearElastic3DLaw` (3D tetra).

**Scope of "verified".** This Kratos build has only **StructuralMechanicsApplication**
(no **ConstitutiveLawsApplication**). So the **linear-elastic family** (2D
plane-strain / plane-stress + 3D) is genuinely load-verified end-to-end. The
**J2 plasticity and hyperelastic** (`neo_hookean`, `st_venant_kirchhoff`) registry
entries require ConstitutiveLawsApplication; their element families exist in
StructuralMechanics but the constitutive-law names are **curated, not load-tested**
(flagged via `requires_app` in `material._REGISTRY`). Re-run the load check on a
build with the app to promote them.

Limitation: the reverse *material* path is wired in `material.py`/`cosim_material.py`
but `read_mdpa` does not yet parse materials.json back into `MaterialModel`s
(regions read back with `material=None`).

## Still TODO
- **Kratos-as-microscale** FE² variant (user wants next). Blocked: KratosCore
  not in `fenicsx-env` conda; Kratos is in system Python → the conda/system
  split is itself the connector case (separate processes over ZeroMQ).
- **GPU surrogate worker** (batched constitutive eval on GPU workers).
- M2 parallel stateful FE² (binary state frame), PBC, consistent macro tangent.
