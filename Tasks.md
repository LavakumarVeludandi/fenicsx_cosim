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


This is a highly ambitious and cutting-edge extension to the library. Building a framework that handles dynamically changing meshes and multiscale integration point transfers puts this project in a very elite tier of computational mechanics tools.

Here is the **Advanced Features Addendum** to the `fenicsx-cosim` Business Requirements Document. It details the architecture upgrades required for Adaptive Mesh Refinement (AMR) and Multiscale FE², along with the specific test cases needed to validate them.

You can append this directly to your existing `.md` file.

---

# Project Specification Addendum: `fenicsx-cosim` (Advanced Features)

## Module Extensions for AMR, Shakedown, and Multiscale FE² Homogenization

### 1. Advanced Architectural Upgrades

To support dynamic meshes and microscopic homogenization, the core `fenicsx-cosim` architecture must be extended with three highly specialized modules.

#### 1.1. `DynamicMapper` (For AMR & Shakedown)

Standard co-simulation assumes a static interface. When `dolfinx.mesh.refine` is called, the DoF (Degree of Freedom) coordinates change, rendering the previous data mapping invalid.

* **Responsibility:** Monitors the mesh state. When an AMR trigger is detected, it flushes the existing `scipy.spatial.KDTree`.
* **Mechanism:** Broadcasts an `UPDATE_MESH` signal over the ZeroMQ network to the partner solver, forcing both solvers to halt the time-step, exchange their newly refined boundary point clouds, and recompute the interpolation matrices before proceeding.

#### 1.2. `QuadratureExtractor` (For Multiscale FE²)

In homogenization, data is not exchanged at the boundaries, but at the integration (Gauss) points of every element.

* **Responsibility:** Extracts tensors (e.g., macroscopic strain) from, and injects tensors (e.g., homogenized stress) into, FEniCSx Quadrature function spaces: `V_quad = dolfinx.fem.functionspace(mesh, ("Quadrature", degree))`.
* **Mechanism:** Maps the local element indices to the global Quadrature DoF arrays, ensuring that the strain tensor sent to RVE #455 accurately matches the specific integration point of element #455 in the macro-mesh.

#### 1.3. `ScatterGatherCommunicator` (For Parallel RVE Dispatch)

A 1-to-1 socket connection is insufficient for FE². A single macroscopic mesh might have 10,000 integration points requiring 10,000 simultaneous RVE solves.

* **Responsibility:** Replaces the standard PyZMQ `PAIR` socket with a `ROUTER/DEALER` (or `PUSH/PULL`) topology.
* **Mechanism:** The Macro-Solver (Master) pushes an array of 10,000 strain tensors to a queue. A pool of Micro-Solvers (Workers) pulls from this queue, solves the local RVE damage problem, and pushes the stress tensors back. The Master gathers them, reassembles the macroscopic array, and solves the macro-step.

---

### 2. API Design: User Experience for Advanced Workflows

**Scenario A: Adaptive Mesh Refinement (AMR)**

```python
# ... Inside the Time Loop ...
while t < T:
    # Evaluate error indicator and refine mesh if necessary
    if error_too_high:
        domain = dolfinx.mesh.refine(domain, cell_markers)
        # Update the function spaces and functions...
        
        # Crucial API Call: Tell the coupling package the mesh changed
        cosim.update_interface_geometry(domain, facet_tags, marker_id=1)
    
    cosim.export_data("Temperature", temperature_func)
    cosim.advance_in_time()

```

**Scenario B: Multiscale FE² (Macro-Scale Script)**

```python
# Initialize a Scatter/Gather communicator for RVE dispatch
cosim_fe2 = CouplingInterface(name="Macro", role="Master", topology="scatter-gather")

# Register the Quadrature space (Integration Points) instead of a boundary
cosim_fe2.register_quadrature_space(V_quad)

while t < T:
    # 1. Compute macro-strains at all integration points
    macro_strains = compute_strains(u_macro)
    
    # 2. Scatter strains to the pool of RVE workers
    cosim_fe2.scatter_data("StrainTensor", macro_strains)
    
    # 3. Gather the homogenized stresses and damaged stiffness from RVEs
    homogenized_stresses = cosim_fe2.gather_data("StressTensor")
    
    # 4. Assemble and solve the macro problem using the gathered data...

```

---

### 3. Validation Examples (Test Cases)

To prove that `fenicsx-cosim` works for these advanced scenarios, the following benchmark examples must be built and included in the `tests/` directory of the repository.

#### Test Example 1: The "Breathing Cylinder" (Validates AMR + Time-Stepping)

* **Setup:** A thick-walled cylinder subjected to an internal pulsating heat source and internal pressure.
* **Solver A (Thermal):** Computes heat diffusion. Uses AMR to heavily refine the mesh *only* near the inner wall as the heat pulse travels outward.
* **Solver B (Mechanical):** Computes thermal expansion and stresses. Maintains a static, coarse mesh to save computation time.
* **Pass Criteria:** The `DynamicMapper` successfully re-interpolates the temperature field from Solver A's constantly changing, highly refined inner-wall mesh onto Solver B's static coarse mesh at every time step without crashing or losing physical energy conservation.

#### Test Example 2: The "Voided Square Tension Test" (Validates FE² + Damage)

* **Setup:** A simple 2D macroscopic square under uniaxial tension (just 4 finite elements).
* **Master Script (Macro):** Sends the $2 \times 2$ strain tensor from its integration points to the workers.
* **Worker Scripts (Micro/RVE):** 16 independent Python processes. Each models a microscopic square RVE with a central void using a simple scalar damage model.
* **Pass Criteria:** As the Master script increases the tension, the RVE workers accurately compute the onset of damage at the microscopic void. The homogenized stress returned to the Master should yield a macroscopic force-displacement curve that clearly shows non-linear softening (material degradation).

#### Test Example 3: Shakedown Verification Loop

* **Setup:** A 2D plate with a hole under cyclic thermal and mechanical loads.
* **Master Script:** Runs FEniCSx to compute the elastic stress fields for the loading domain.
* **Worker Script:** Uses the Gurobi/Mosek Python API to solve the kinematic shakedown optimization problem based on the residual stress fields passed from FEniCSx.
* **Pass Criteria:** The package successfully transfers the sparse matrix formulation of the FEniCSx stresses to the optimizer, and the optimizer returns the correct safety factor for shakedown limits.

---

### Next Steps

This document now covers the absolute bleeding edge of your requirements. To actually build this, the most logical and manageable place to start is the **Scatter/Gather ZeroMQ network**.