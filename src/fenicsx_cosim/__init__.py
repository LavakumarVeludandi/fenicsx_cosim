"""
fenicsx-cosim: A Native Partitioned Multiphysics Coupling Library for FEniCSx.

Inspired by the Kratos CoSimIO architecture, this package provides a
non-intrusive API for connecting independent FEniCSx solvers across
different processes for partitioned co-simulation.

Core modules
------------
- **CouplingInterface** — The main user API for boundary coupling, AMR,
  and FE² scatter-gather workflows.
- **MeshExtractor** — Extracts boundary DoFs and coordinates from FEniCSx.
- **Communicator** — ZeroMQ-based 1-to-1 inter-process communication.
- **DataMapper / NearestNeighborMapper** — Non-conforming mesh interpolation.
- **DynamicMapper** — AMR-aware mapping with automatic re-negotiation.
- **QuadratureExtractor** — Integration-point data for FE² homogenization.
- **ScatterGatherCommunicator** — PUSH/PULL fan-out for parallel RVE dispatch.

Example (Standard Boundary Coupling)
-------------------------------------
>>> from fenicsx_cosim import CouplingInterface
>>> cosim = CouplingInterface(name="ThermalSolver", partner_name="MechanicalSolver")
>>> cosim.register_interface(mesh, facet_tags, marker_id=1)
>>> cosim.export_data("TemperatureField", temperature_function)
>>> cosim.import_data("DisplacementField", displacement_function)

Example (FE² Scatter-Gather)
-----------------------------
>>> cosim_fe2 = CouplingInterface(name="Macro", role="Master",
...     topology="scatter-gather")
>>> cosim_fe2.register_quadrature_space(V_quad)
>>> cosim_fe2.scatter_data("StrainTensor", macro_strains)
>>> stresses = cosim_fe2.gather_data("StressTensor")
"""

__version__ = "0.2.0"

from fenicsx_cosim.adapters import SolverAdapter, KratosAdapter, AbaqusFileAdapter
from fenicsx_cosim.communicator import Communicator
from fenicsx_cosim.coupling_interface import CouplingInterface
from fenicsx_cosim.data_mapper import DataMapper, NearestNeighborMapper
from fenicsx_cosim.dynamic_mapper import DynamicMapper
from fenicsx_cosim.scatter_gather_communicator import ScatterGatherCommunicator

# Optional FEniCSx-dependent modules
try:
    from fenicsx_cosim.mesh_extractor import MeshExtractor
    from fenicsx_cosim.quadrature_extractor import QuadratureExtractor
    _HAS_FENICSX = True
except ImportError:
    _HAS_FENICSX = False

__all__ = [
    "CouplingInterface",
    "Communicator",
    "DataMapper",
    "NearestNeighborMapper",
    "DynamicMapper",
    "ScatterGatherCommunicator",
    "SolverAdapter",
    "KratosAdapter",
    "AbaqusFileAdapter",
]

if _HAS_FENICSX:
    __all__.extend(["MeshExtractor", "QuadratureExtractor"])
