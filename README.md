# fenicsx-cosim

**A Native Partitioned Multiphysics Coupling Library for FEniCSx**

`fenicsx-cosim` is a standalone Python package that enables partitioned multiphysics co-simulation for [FEniCSx](https://fenicsproject.org/) (v0.10+). Inspired by the architecture of [Kratos CoSimIO](https://github.com/KratosMultiphysics/CoSimIO), it provides a non-intrusive API for connecting independent FEniCSx solvers across different processes.

## Features

- **Clean API** — A single `CouplingInterface` class hides all networking and mapping complexity
- **ZeroMQ IPC** — Uses PyZMQ for inter-process communication that doesn't interfere with FEniCSx's internal MPI
- **Automatic Mesh Mapping** — Nearest-neighbor interpolation via `scipy.spatial.KDTree` for non-conforming boundaries
- **FEniCSx Native** — Works directly with `dolfinx.fem.Function`, `dolfinx.mesh.Mesh`, and `MeshTags`

## Installation

```bash
pip install -e .
```

### Dependencies

| Package | Purpose |
|---|---|
| `fenics-dolfinx >= 0.10.0` | Core finite element backend |
| `mpi4py` | MPI parallel support |
| `numpy >= 1.24.0` | Array manipulation |
| `pyzmq >= 25.0.0` | Inter-Process Communication |
| `scipy >= 1.10.0` | KDTree for mesh mapping |

## Quick Start

### Thermal Solver (Terminal 1)

```python
import dolfinx
from mpi4py import MPI
from fenicsx_cosim import CouplingInterface

# Standard FEniCSx setup
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 20, 20)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
temperature = dolfinx.fem.Function(V)

# Initialize co-simulation
cosim = CouplingInterface(name="ThermalSolver", partner_name="MechanicalSolver")
cosim.register_interface(mesh, facet_tags, marker_id=1, function_space=V)

# Time loop
while t < T:
    # ... solve thermal problem ...
    cosim.export_data("TemperatureField", temperature)
    cosim.import_data("DisplacementField", displacement)
    cosim.advance_in_time()
```

### Mechanical Solver (Terminal 2)

```python
from fenicsx_cosim import CouplingInterface

cosim = CouplingInterface(name="MechanicalSolver", partner_name="ThermalSolver")
cosim.register_interface(mesh, facet_tags, marker_id=1, function_space=V)

while t < T:
    cosim.import_data("TemperatureField", temperature)
    # ... solve mechanical problem ...
    cosim.export_data("DisplacementField", displacement)
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

## Running Tests

```bash
# Tests that don't require DOLFINx
pytest tests/test_communicator.py tests/test_data_mapper.py -v

# Full test suite (requires DOLFINx)
pytest tests/ -v
```

## Development Roadmap

| Phase | Status | Description |
|---|---|---|
| Phase 1 — Core Data Extraction | ✅ | `MeshExtractor` for boundary DoF isolation |
| Phase 2 — IPC Layer | ✅ | `Communicator` with PyZMQ |
| Phase 3 — Integration & API | ✅ | `CouplingInterface` combining all components |
| Phase 4 — Mapping | ✅ | `NearestNeighborMapper` with KDTree |
| Phase 5 — Validation | 🔄 | Benchmark examples & packaging |

## License

MIT
