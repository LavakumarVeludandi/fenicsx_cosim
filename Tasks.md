Here is a comprehensive Software Architecture and Business Requirements Document for building your FEniCSx coupling package from the ground up.

For the sake of this document, I have given the project a working title: **`fenicsx-cosim`**.

You can save the following block directly as a `.md` file.

---

# Project Specification: `fenicsx-cosim`

## A Native Partitioned Multiphysics Coupling Library for FEniCSx

### 1. Executive Summary

`fenicsx-cosim` is a standalone, installable Python package designed to enable partitioned multiphysics co-simulation exclusively for FEniCSx (v0.10+). Inspired by the architecture of Kratos CoSimIO, it provides a non-intrusive API for researchers and engineers to connect independent FEniCSx solvers (e.g., thermal, mechanical, multiscale FE²) across different processes. By handling inter-process communication (IPC), boundary data extraction, and mesh mapping natively within the FEniCSx ecosystem, it eliminates the need for monolithic solver development.

### 2. Dependency & Requirements (`pyproject.toml` / `requirements.txt`)

To ensure high performance and modern packaging, the project will use a modern `pyproject.toml` build system.

**Core Dependencies:**

* `fenics-dolfinx >= 0.10.0`: The core finite element backend.
* `mpi4py`: Required for FEniCSx parallelism and distributed data handling.
* `numpy >= 1.24.0`: For array manipulation and buffer transfers.
* `pyzmq >= 25.0.0`: (ZeroMQ) The engine for Inter-Process Communication (IPC). It is lightweight, handles TCP/IP or local sockets, and crucially, *does not interfere with FEniCSx's internal MPI communicators*.
* `scipy >= 1.10.0`: specifically `scipy.spatial.KDTree` for fast nearest-neighbor searches when mapping non-conforming mesh boundaries.

### 3. High-Level System Architecture

The system relies on a **Peer-to-Peer** or **Remote Control** topology.

1. **The Solvers (Workers):** Standard FEniCSx scripts. They initialize a `CouplingInterface` object.
2. **The Communication Layer:** ZeroMQ passes serialized NumPy arrays (representing boundary coordinates or field values) between the solvers.
3. **The Mapping Layer:** Re-interpolates data if Solver A's mesh does not perfectly align with Solver B's mesh.

---

### 4. Core Classes and Interfaces

#### 4.1. `CouplingInterface` (The User API)

This is the main entry point for the user. It abstracts away all the networking and mapping.

* **Methods:**
* `__init__(self, name: str, partner_name: str, connection_type: str = "tcp")`
* `register_interface(self, mesh: dolfinx.mesh.Mesh, facet_tags: dolfinx.mesh.meshtags, marker_id: int)`: Tells the package *where* the coupling happens.
* `export_data(self, data_name: str, function: dolfinx.fem.Function)`: Extracts the boundary values and sends them over the network.
* `import_data(self, data_name: str, function: dolfinx.fem.Function)`: Receives data and injects it into the FEniCSx Function.
* `advance_in_time(self)`: Synchronizes the solvers.



#### 4.2. `MeshExtractor` (FEniCSx Engine)

Handles the highly specific FEniCSx v0.10 API calls to isolate boundary data.

* **Responsibilities:**
* Use `dolfinx.mesh.locate_entities_boundary` to find physical facets.
* Use `dolfinx.fem.locate_dofs_topological` to get the Degrees of Freedom (DoFs) living on those facets.
* Extract the `(x, y, z)` coordinates of these specific DoFs to build a point cloud.



#### 4.3. `Communicator` (Network Engine)

Wraps ZeroMQ (`pyzmq`) to handle the actual bytes sent over the wire.

* **Responsibilities:**
* Establish `REQ/REP` (Request/Reply) or `PAIR` sockets between the Python scripts.
* Serialize NumPy arrays (using `.tobytes()` or `pickle` for complex metadata).
* Handle timeouts and handshake protocols to ensure Solver A doesn't run ahead of Solver B.



#### 4.4. `DataMapper` (Interpolation Engine)

If the thermo-mechanical meshes are different, data must be mapped.

* **Responsibilities:**
* **Nearest Neighbor Mapping:** Takes coordinates from Solver A, finds the closest DoF on Solver B using `scipy.spatial.KDTree`, and assigns the value.
* **Projection Mapping:** (Future scope) Galerkin projection between non-matching boundary meshes.



---

### 5. API Design: User Experience (How it looks in practice)

The ultimate goal is to keep the user's FEniCSx script as clean as possible. Here is a mock-up of what the user will write in their Thermal Solver script:

```python
import dolfinx
from mpi4py import MPI
from fenicsx_cosim import CouplingInterface

# 1. Standard FEniCSx Setup
domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
temperature = dolfinx.fem.Function(V)

# 2. Initialize Co-Simulation
cosim = CouplingInterface(name="ThermalSolver", partner_name="MechanicalSolver")

# Define the coupling boundary (e.g., the top edge of the square)
# (Assuming facet_tags and marker_id=1 are previously defined)
cosim.register_interface(domain, facet_tags, marker_id=1)

# 3. Time Loop
while t < T:
    # ... Solve FEniCSx Thermal Problem ...
    
    # Send Temperature to Mechanics
    cosim.export_data("TemperatureField", temperature)
    
    # Receive Displacements back from Mechanics
    cosim.import_data("DisplacementField", displacement_func)
    
    cosim.advance_in_time()

```

---

### 6. Development Roadmap

| Phase | Milestone | Deliverables |
| --- | --- | --- |
| **Phase 1** | **Core Data Extraction** | Build the `MeshExtractor`. Successfully isolate boundary DoFs and coordinates from a FEniCSx v0.10 mesh. |
| **Phase 2** | **IPC Layer** | Build the `Communicator` using PyZMQ. Successfully send a dummy NumPy array between two running Python terminals. |
| **Phase 3** | **Integration & API** | Combine Phase 1 and 2 into the `CouplingInterface`. Write values from one solver directly into the `dolfinx.fem.Function.x.array` of another. |
| **Phase 4** | **Mapping** | Implement `scipy.spatial.KDTree` to allow for non-matching meshes. |
| **Phase 5** | **Validation** | Run a two-way thermo-mechanical benchmark. Package using `pyproject.toml` and publish internally or on PyPI. |

---

### Next Steps

This document gives you the blueprint. The hardest technical hurdle to clear first is **Phase 1**: cleanly extracting the boundary DoF coordinates and their corresponding indices in the FEniCSx memory array.

Would you like me to write the prototype Python code for the `MeshExtractor` class using the FEniCSx v0.10 API so you can test isolating the boundary data?