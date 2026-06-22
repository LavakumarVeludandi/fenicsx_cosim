
# fenicsx-cosim

<p align="center">
  <img src="docs/media/fenicsx-cosim.svg" alt="fenicsx-cosim logo">
</p>

[![CI](https://github.com/LavakumarVeludandi/fenicsx_cosim/actions/workflows/ci.yml/badge.svg)](https://github.com/LavakumarVeludandi/fenicsx_cosim/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/LavakumarVeludandi/fenicsx_cosim/badge.svg)](https://codecov.io/gh/LavakumarVeludandi/fenicsx_cosim)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Documentation Status](https://img.shields.io/badge/docs-online-blue)](https://lavakumarveludandi.github.io/fenicsx_cosim/)

**Documentation:** [https://lavakumarveludandi.github.io/fenicsx_cosim/](https://lavakumarveludandi.github.io/fenicsx_cosim/)

**The coupling layer for people who already live in FEniCSx.**

`fenicsx-cosim` is a standalone, pure-Python package for partitioned
multiphysics with [FEniCSx](https://fenicsproject.org/) (v0.10+). It is *not*
trying to out-feature [preCICE](https://precice.org/) on generic FSI/CHT — it
wins where preCICE is heavy or absent **for a FEniCSx-native workflow**:

- **Zero-setup ergonomics** — `pip install`, pure Python, operate directly on
  `dolfinx.fem.Function` / `Mesh` / `MeshTags`. No C++ runtime, no XML config,
  no adapter compilation. Time-to-first-coupling is minutes.
- **FE² task-farming** — scatter quadrature-point strains to a worker pool and
  gather homogenized stresses (`PUSH/PULL`, `REQ/REP` broker). preCICE couples a
  fixed set of participants over a shared interface mesh; it does **not** do
  many-subproblem RVE dispatch. This one is ours.
- **FEniCSx ↔ Kratos mesh/material translation** — write/read real Kratos
  `.mdpa` + `StructuralMaterials.json` (registered element families, Properties,
  SubModelParts) that Kratos actually loads. A *translator*, orthogonal to
  field coupling — nobody else does it.

### When to use which

| You want… | Use |
|---|---|
| FEniCSx FE² homogenization, RVE worker pools | **fenicsx-cosim** |
| Move a mesh + material between FEniCSx and Kratos | **fenicsx-cosim** |
| Quick partitioned coupling between FEniCSx and one other code, minimal setup | **fenicsx-cosim** |
| Production FSI/CHT across many codes (OpenFOAM, SU2, CalculiX), IQN quasi-Newton, RBF/waveform mapping | **preCICE** |

See [`docs/comparison_precice.md`](docs/comparison_precice.md) for the honest,
specific breakdown.

> **Note.** Coupling *two FEniCSx solvers* that could be one program is an
> anti-pattern — solve it monolithically. Partitioning earns its keep only
> across **different codes** (the connector case) or for **many independent
> subproblems** (FE²). The examples below are framed accordingly.

## Features

- **Clean API** — A single `CouplingInterface` class hides networking and mapping
- **ZeroMQ IPC** — PyZMQ inter-process communication that doesn't interfere with FEniCSx's internal MPI
- **FEniCSx Native** — Works directly with `dolfinx.fem.Function`, `dolfinx.mesh.Mesh`, and `MeshTags`
- **Mesh Mapping** — nearest-neighbor and consistent (inverse-distance) interpolation for non-conforming boundaries
- **Implicit coupling** — Aitken dynamic relaxation and IQN-ILS quasi-Newton for strongly-coupled (added-mass) problems
- **Multiple Coupling Topologies** — `PAIR` (1-to-1), `PUSH/PULL` scatter-gather FE², and `REQ/REP` demand-driven FE² broker
- **Adapter-Based Integration** — adapter abstractions for FEniCSx, Kratos, and Abaqus workflows
- **Validated benchmarks** — analytic correctness gates in [`benchmarks/`](benchmarks/), enforced in CI

## Installation

```bash
git clone https://github.com/LavakumarVeludandi/fenicsx_cosim.git
cd fenicsx_cosim
pip install -e .
```

### Dependencies

| Package | Type | Purpose |
|---|---|
| `numpy >= 1.24.0` | Core | Array manipulation |
| `pyzmq >= 25.0.0` | Core | Inter-process communication |
| `scipy >= 1.10.0` | Core | KDTree-based mapping |
| `fenics-dolfinx >= 0.10.0` | Optional (`fenicsx`) | FEniCSx coupling backend |
| `mpi4py` | Optional (`fenicsx`) | MPI support for FEniCSx runs |

## Quick Start

### FE² homogenization — the flagship (one macro solver, a pool of RVE workers)

```python
from fenicsx_cosim import CouplingInterface

# Macro solver: scatter quadrature-point strains, gather homogenized stresses.
macro = CouplingInterface(name="Macro", role="Master", topology="scatter-gather")
macro.register_quadrature_space(V_quad)
macro.scatter_gather_data("Strain", strain_fn, "Stress", stress_fn)
```

Start as many RVE workers as you want; each pulls a strain, solves its micro
problem, and pushes back a stress. See `examples/fe2/`.

### N > 2 coupling via the broker hub

```python
from fenicsx_cosim import CouplingBroker, CouplingInterface

# One hub process:
CouplingBroker("tcp://*:5560", expected=3).start()

# Each participant (Fluid / Thermal / Structure):
ci = CouplingInterface(name="Fluid", topology="broker",
                       endpoint="tcp://localhost:5560")
ci.register_broker()                       # blocks until all 3 have joined
ci.send_to("Thermal", "pressure", p_values)
src, field, values = ci.receive_from()
ci.barrier()
```

### Connector: FEniCSx ↔ another code (e.g. Kratos)

> Two *FEniCSx* solvers that could be one program → solve monolithically.
> Partitioning earns its keep only across **different codes**. Here the partner
> is an external solver, and strong (implicit) coupling is wrapped with an
> accelerator so added-mass problems converge.

```python
from fenicsx_cosim import CouplingInterface, IQNILS, fixed_point_iterate

cosim = CouplingInterface(name="FEniCSxSolver", partner_name="ExternalSolver")
cosim.register_interface(mesh, facet_tags, marker_id=1, function_space=V)

accel = IQNILS()                           # or Aitken() for cheap relaxation
for _ in range(n_steps):
    # one strong-coupling step: sub-iterate to a converged interface state
    def coupled(x):                        # x = interface field (numpy)
        cosim.export_data("Traction", x_to_function(x))
        cosim.import_data("Displacement", disp)
        return function_to_array(disp)
    x_star, residuals = fixed_point_iterate(coupled, x0, accel)
    cosim.advance_in_time()
```

## Architecture

```
┌─────────────────────────┐         ZeroMQ          ┌─────────────────────────┐
│     Thermal Solver      │ ◄════════════════════► │   Mechanical Solver     │
│                         │     TCP / IPC            │                         │
│  ┌───────────────────┐  │                         │  ┌───────────────────┐  │
│  │ CouplingInterface │  │                         │  │ CouplingInterface │  │
│  │  ├─ MeshExtractor │  │                         │  │  ├─ MeshExtractor │  │
│  │  ├─ Communicator  │  │                         │  │  ├─ Communicator  │  │
│  │  └─ DataMapper    │  │                         │  │  └─ DataMapper    │  │
│  └───────────────────┘  │                         │  └───────────────────┘  │
└─────────────────────────┘                         └─────────────────────────┘
```

### Core Components

| Component | Description |
|---|---|
| `CouplingInterface` | User-facing API — orchestrates everything |
| `MeshExtractor` | Extracts boundary DoFs and coordinates from FEniCSx meshes |
| `Communicator` | ZeroMQ `PAIR` sockets for bidirectional data exchange |
| `DataMapper` | `scipy.spatial.KDTree` nearest-neighbor mapping for non-conforming meshes |
| `DynamicMapper` | Handles AMR mesh-remapping via ZeroMQ mesh-update negotiation |
| `QuadratureExtractor` | FE² integration point data extraction (via basix & ufl) |
| `ScatterGatherCommunicator` | ZeroMQ `PUSH/PULL` sockets for parallel RVE dispatch |
| `DemandDrivenBroker` | ZeroMQ `REQ/REP` demand-driven scheduling for dynamically balanced FE² workers |
| `SolverAdapter` + adapters | External solver bridge abstractions (`FEniCSxAdapter`, `KratosAdapter`, `AbaqusFileAdapter`) |

## Advanced Examples

### 1. Adaptive Mesh Refinement (AMR)
Demonstrates a thermal solver that refines its mesh mid-simulation, seamlessly negotiating the new interpolation mapping with a static-mesh mechanical solver.

Terminal 1:
```bash
export PYTHONPATH=src
python examples/amr_thermal_solver.py
```

Terminal 2:
```bash
export PYTHONPATH=src
python examples/amr_mechanical_solver.py
```

### 2. Multiscale FE² Homogenization
Demonstrates an FE² macro-solver dispatching quadrature-point strains to a pool of microscopic (RVE) workers in parallel, and gathering homogenized stresses.

Terminal 1 (Master):
```bash
export PYTHONPATH=src
python examples/fe2_macro_solver.py
```

Terminals 2+ (Workers):
```bash
# Run this in as many terminals as you want workers!
export PYTHONPATH=src
python examples/fe2_micro_worker.py
```

### 3. Shakedown Verification Loop
Demonstrates extracting whole sparse stiffness matrices from FEniCSx and beaming them to a mock Gurobi/Mosek optimizer on another process to compute a kinematic safety factor.

Terminal 1 (FEniCSx Master):
```bash
export PYTHONPATH=src
python examples/shakedown_fenicsx_master.py
```

Terminal 2 (Optimizer Worker):
```bash
export PYTHONPATH=src
python examples/shakedown_optimizer_worker.py
```

### 4. Coupling with Kratos Multiphysics (via ZeroMQ)
Demonstrates the `KratosAdapter` coupling a FEniCSx mechanical solver to a native Kratos thermal solver in real-time.

Terminal 1 (Kratos thermal Server):
```bash
export PYTHONPATH=src
python examples/kratos_thermal_solver.py
```

Terminal 2 (FEniCSx mechanical Client):
```bash
export PYTHONPATH=src
python examples/fenicsx_kratos_mechanical.py
```

### 5. File-Based Staggered Coupling with Abaqus
Demonstrates the `AbaqusFileAdapter` syncing a FEniCSx thermal solver with an Abaqus Python wrapper using shared NumPy `.npy` files.

Terminal 1 (FEniCSx thermal Client):
```bash
export PYTHONPATH=src
python examples/fenicsx_abaqus_thermal.py
```

Terminal 2 (Abaqus wrapper Server):
```bash
export PYTHONPATH=src
python examples/abaqus_coupling_wrapper.py
```

## Running Tests

Install development dependencies and run tests from the repository root:

```bash
pip install -e ".[dev]"
export PYTHONPATH=src
pytest tests/ -v
```

Note: tests that require `fenics-dolfinx` are automatically skipped when DOLFINx is unavailable.

## Development Roadmap

For current and planned work, see:

- [`ROADMAP.md`](ROADMAP.md)
- [`CHANGELOG.md`](CHANGELOG.md)
- [`CONTRIBUTING.md`](CONTRIBUTING.md)

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Lavakumar Veludandi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the conditions stated in the [LICENSE](LICENSE) file.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED. See the [LICENSE](LICENSE) file for details.
