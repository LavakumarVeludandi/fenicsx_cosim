"""
fenicsx-cosim: A Native Partitioned Multiphysics Coupling Library for FEniCSx.

Inspired by the Kratos CoSimIO architecture, this package provides a
non-intrusive API for connecting independent FEniCSx solvers across
different processes for partitioned co-simulation.

Example
-------
>>> from fenicsx_cosim import CouplingInterface
>>> cosim = CouplingInterface(name="ThermalSolver", partner_name="MechanicalSolver")
>>> cosim.register_interface(mesh, facet_tags, marker_id=1)
>>> cosim.export_data("TemperatureField", temperature_function)
>>> cosim.import_data("DisplacementField", displacement_function)
"""

__version__ = "0.1.0"

from fenicsx_cosim.coupling_interface import CouplingInterface
from fenicsx_cosim.mesh_extractor import MeshExtractor
from fenicsx_cosim.communicator import Communicator
from fenicsx_cosim.data_mapper import DataMapper, NearestNeighborMapper

__all__ = [
    "CouplingInterface",
    "MeshExtractor",
    "Communicator",
    "DataMapper",
    "NearestNeighborMapper",
]
