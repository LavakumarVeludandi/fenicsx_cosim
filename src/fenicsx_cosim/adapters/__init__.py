"""
Solver adapters for fenicsx-cosim.

This subpackage provides adapter classes that bridge external solvers
(Kratos, Abaqus, etc.) to the fenicsx-cosim communication layer.

Each adapter implements the :class:`SolverAdapter` ABC, which standardizes
how boundary coordinates and field data are extracted/injected regardless
of the underlying solver.
"""

from fenicsx_cosim.adapters.base import SolverAdapter

__all__ = [
    "SolverAdapter",
]

# Lazy imports for optional adapters — avoids import errors when
# the corresponding solver is not installed.

def __getattr__(name):
    if name == "FEniCSxAdapter":
        from fenicsx_cosim.adapters.fenicsx_adapter import FEniCSxAdapter
        return FEniCSxAdapter
    if name == "KratosAdapter":
        from fenicsx_cosim.adapters.kratos_adapter import KratosAdapter
        return KratosAdapter
    if name == "AbaqusFileAdapter":
        from fenicsx_cosim.adapters.abaqus_adapter import AbaqusFileAdapter
        return AbaqusFileAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
