# fenicsx-cosim Roadmap

A standalone pure-Python partitioned multiphysics coupling library for FEniCSx v0.10+.
This roadmap tracks planned improvements by release milestone.

---

## v0.2.0 — Packaging & Repo Health *(In Progress)*

- [ ] Fix author metadata and add project URLs in `pyproject.toml`
- [ ] Publish package to PyPI (`pip install fenicsx-cosim`)
- [ ] Add GitHub Actions CI workflow to run `pytest` on every push
- [ ] Add code coverage reporting via Codecov
- [ ] Connect repo to Zenodo for a citable DOI
- [ ] Create first tagged release `v0.1.0` on GitHub
- [ ] Add `CONTRIBUTING.md` with dev setup and PR guidelines
- [ ] Add `CHANGELOG.md` starting from v0.1.0
- [ ] Add GitHub issue templates (bug report, feature request)
- [ ] Enable GitHub Discussions for community Q&A
- [ ] Add repository topics: `fenicsx`, `dolfinx`, `cosimulation`, `multiphysics`, `partitioned-coupling`
- [ ] Add README badges: CI status, PyPI version, DOI, license

---

## v0.3.0 — Strong (Implicit) Coupling

Currently `CouplingInterface` supports only **explicit coupling** — each solver runs once per
time step. For problems with strong physical feedback (e.g. fluid-structure interaction,
large-deformation thermo-mechanics), solvers must iterate within each time step until
they reach a converged state.

- [ ] Add Aitken Δ² relaxation for fixed-point acceleration (`convergence.py`)
- [ ] Add `advance_implicit()` method to `CouplingInterface` with sub-iteration loop
- [ ] Add convergence tolerance check with ZeroMQ synchronisation handshake
- [ ] Add convergence monitoring: iteration count and residual norm reporting
- [ ] Add example: `examples/implicit_thermal_mechanical.py` with convergence plot
- [ ] Add test: verify implicit loop converges on a known coupled problem

---

## v0.4.0 — Multi-Solver (N > 2) Topology

The current `Communicator` uses ZeroMQ **PAIR** sockets which are strictly 1-to-1.
Coupling three or more solvers (e.g. fluid + thermal + structural) requires a different
socket pattern.

- [ ] Implement `BrokerCommunicator` using ZeroMQ ROUTER/DEALER for N-solver topologies
- [ ] Update `CouplingInterface` API to support named solver routing
- [ ] Add example: three-way coupling (fluid + thermal + mechanical)
- [ ] Add test: N=3 solver synchronisation and data exchange

---

## v0.5.0 — Validation & Benchmarks

Before broader community use, the library needs verified correctness against
known analytical solutions.

- [ ] Add `benchmarks/` directory with automated result generation
- [ ] Benchmark: 1D heat equation split across two subdomains — compare against analytical solution
- [ ] Benchmark: Neumann-Neumann flux conservation test at coupling interface
- [ ] Benchmark: FE² macro-micro convergence study
- [ ] Benchmark: AMR remapping accuracy before and after mesh refinement event
- [ ] All benchmarks produce plots saved to `benchmarks/results/`

---

## v0.6.0 — Documentation & Tutorials

- [ ] Jupyter notebook: `01_basic_thermal_mechanical.ipynb` — end-to-end coupled solve with plots
- [ ] Jupyter notebook: `02_fe2_multiscale.ipynb` — macro-RVE dispatch and homogenisation
- [ ] Jupyter notebook: `03_amr_coupling.ipynb` — mid-simulation mesh refinement and remapping
- [ ] Add `docs/comparison_precice.md` — honest comparison vs. preCICE FEniCSx adapter
  - Installation complexity (pure Python vs. compiled C++ runtime)
  - Supported coupling schemes (explicit vs. implicit)
  - Use cases where each is better suited
- [ ] API reference: docstrings on all public methods in `CouplingInterface`
- [ ] Improve HPC / SLURM guide with tested example configurations

---

## Completed ✅

- [x] `CouplingInterface` — single class API hiding all networking and mapping complexity
- [x] `Communicator` — ZeroMQ PAIR sockets for bidirectional data exchange
- [x] `MeshExtractor` — boundary DoF and coordinate extraction from FEniCSx meshes
- [x] `DataMapper` — nearest-neighbor mapping via `scipy.spatial.KDTree`
- [x] `DynamicMapper` — AMR mesh-remapping with ZeroMQ mesh-update negotiation
- [x] `QuadratureExtractor` — FE² integration point data extraction (basix + UFL)
- [x] `ScatterGatherCommunicator` — ZeroMQ PUSH/PULL for parallel RVE dispatch
- [x] `DemandDrivenBroker` — REQ/REP pattern for colocated runs
- [x] `KratosAdapter` — coupling FEniCSx to Kratos Multiphysics via ZeroMQ
- [x] `AbaqusFileAdapter` — file-based staggered coupling with Abaqus (.npy exchange)
- [x] Docker support for containerised coupling
- [x] SLURM job script examples for HPC clusters
- [x] 10 test files covering all core components
- [x] MIT License
