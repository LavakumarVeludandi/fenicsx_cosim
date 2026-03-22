"""
KratosAdapter — SolverAdapter for Kratos Multiphysics.

Bridges the Kratos Multiphysics Python API to fenicsx-cosim, enabling
bidirectional coupling between FEniCSx and Kratos solvers.

This adapter operates on a Kratos ``ModelPart`` and its sub-model-parts
(which define coupling interfaces). It translates between Kratos'
node-based data structures and the NumPy arrays expected by the
fenicsx-cosim communication layer.

Requirements
------------
- ``KratosMultiphysics`` Python package must be importable.
- The Kratos model must be set up with a sub-model-part defining the
  coupling boundary (e.g. ``"coupling_interface"``).

Typical usage
-------------
>>> import KratosMultiphysics
>>> from fenicsx_cosim.adapters import KratosAdapter
>>> from fenicsx_cosim import CouplingInterface
>>>
>>> # Set up Kratos model (simplified)
>>> model = KratosMultiphysics.Model()
>>> model_part = model.CreateModelPart("Structure")
>>> # ... populate mesh, define interface sub-model-part ...
>>>
>>> adapter = KratosAdapter(model_part, "coupling_interface")
>>> cosim = CouplingInterface.from_adapter(
...     adapter, name="KratosSolver", partner_name="FEniCSxSolver"
... )
>>> cosim.register_adapter_interface()
>>> cosim.export_via_adapter("TEMPERATURE")
>>> cosim.import_via_adapter("DISPLACEMENT_X")
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from fenicsx_cosim.adapters.base import SolverAdapter
from fenicsx_cosim.utils import get_logger

logger = get_logger(__name__)

# Lazy import of Kratos
try:
    import KratosMultiphysics
    _HAS_KRATOS = True
except ImportError:
    _HAS_KRATOS = False


def _require_kratos() -> None:
    """Raise an informative error if Kratos is not installed."""
    if not _HAS_KRATOS:
        raise ImportError(
            "KratosMultiphysics is required for KratosAdapter but could "
            "not be imported.  Install Kratos or ensure it is on your "
            "PYTHONPATH."
        )


class KratosAdapter(SolverAdapter):
    """Adapter for coupling Kratos Multiphysics solvers via fenicsx-cosim.

    Parameters
    ----------
    model_part : KratosMultiphysics.ModelPart
        The main Kratos model part containing the simulation data.
    interface_sub_model_part_name : str
        Name of the sub-model-part that defines the coupling boundary.
        All coordinate/field extraction happens on this sub-model-part's
        nodes.
    variable_map : dict[str, Any], optional
        Mapping from string field names to Kratos variable objects.
        If ``None`` (default), variables are looked up dynamically via
        ``KratosMultiphysics.KratosGlobals.GetVariable(field_name)``.

    Examples
    --------
    >>> adapter = KratosAdapter(model_part, "interface_boundary")
    >>> coords = adapter.get_boundary_coordinates()  # (N, 3) array
    >>> temp = adapter.extract_field("TEMPERATURE")   # (N,) array
    >>> adapter.inject_field("DISPLACEMENT_X", disp_x) # set values
    """

    def __init__(
        self,
        model_part: "KratosMultiphysics.ModelPart",
        interface_sub_model_part_name: str,
        variable_map: Optional[dict[str, Any]] = None,
    ) -> None:
        _require_kratos()

        self._model_part = model_part
        self._interface_name = interface_sub_model_part_name
        self._variable_map = variable_map or {}

        # Get the interface sub-model-part
        self._interface = model_part.GetSubModelPart(
            interface_sub_model_part_name
        )

        # Cache node IDs for consistent ordering
        self._node_ids = [node.Id for node in self._interface.Nodes]
        self._num_nodes = len(self._node_ids)

        logger.info(
            "KratosAdapter: interface '%s' has %d nodes",
            interface_sub_model_part_name, self._num_nodes,
        )

    # ------------------------------------------------------------------
    # SolverAdapter interface
    # ------------------------------------------------------------------

    def get_boundary_coordinates(self) -> np.ndarray:
        """Return coupling boundary node coordinates.

        Returns
        -------
        np.ndarray
            Shape ``(N, 3)`` — ``(x, y, z)`` for each interface node,
            in the same order as ``node_ids``.
        """
        coords = np.empty((self._num_nodes, 3), dtype=np.float64)
        for i, node in enumerate(self._interface.Nodes):
            coords[i, 0] = node.X
            coords[i, 1] = node.Y
            coords[i, 2] = node.Z
        return coords

    def extract_field(self, field_name: str) -> np.ndarray:
        """Extract a scalar field from interface nodes.

        Parameters
        ----------
        field_name : str
            Kratos variable name (e.g. ``"TEMPERATURE"``,
            ``"DISPLACEMENT_X"``).

        Returns
        -------
        np.ndarray
            Shape ``(N,)`` — one value per interface node.
        """
        var = self._resolve_variable(field_name)
        values = np.empty(self._num_nodes, dtype=np.float64)
        for i, node in enumerate(self._interface.Nodes):
            values[i] = node.GetSolutionStepValue(var, 0)
        return values

    def extract_vector_field(self, field_name: str) -> np.ndarray:
        """Extract a vector field (e.g. DISPLACEMENT, VELOCITY).

        Parameters
        ----------
        field_name : str
            Kratos vector variable name.

        Returns
        -------
        np.ndarray
            Shape ``(N, 3)`` — ``(x, y, z)`` components per node.
        """
        var = self._resolve_variable(field_name)
        values = np.empty((self._num_nodes, 3), dtype=np.float64)
        for i, node in enumerate(self._interface.Nodes):
            vec = node.GetSolutionStepValue(var, 0)
            values[i, 0] = vec[0]
            values[i, 1] = vec[1]
            values[i, 2] = vec[2]
        return values

    def inject_field(self, field_name: str, values: np.ndarray) -> None:
        """Set a scalar field at interface nodes.

        Parameters
        ----------
        field_name : str
            Kratos variable name.
        values : np.ndarray
            Shape ``(N,)`` — one value per interface node.

        Raises
        ------
        ValueError
            If *values* length doesn't match the number of interface nodes.
        """
        if len(values) != self._num_nodes:
            raise ValueError(
                f"Expected {self._num_nodes} values, got {len(values)}"
            )
        var = self._resolve_variable(field_name)
        for i, node in enumerate(self._interface.Nodes):
            node.SetSolutionStepValue(var, 0, float(values[i]))

    def inject_vector_field(
        self, field_name: str, values: np.ndarray
    ) -> None:
        """Set a vector field at interface nodes.

        Parameters
        ----------
        field_name : str
            Kratos vector variable name.
        values : np.ndarray
            Shape ``(N, 3)``.
        """
        if values.shape != (self._num_nodes, 3):
            raise ValueError(
                f"Expected shape ({self._num_nodes}, 3), got {values.shape}"
            )
        var = self._resolve_variable(field_name)
        
        # When mocked, KratosMultiphysics might not be in the global namespace of this file
        import KratosMultiphysics
        
        for i, node in enumerate(self._interface.Nodes):
            vec = KratosMultiphysics.Array3()
            vec[0] = float(values[i, 0])
            vec[1] = float(values[i, 1])
            vec[2] = float(values[i, 2])
            node.SetSolutionStepValue(var, 0, vec)

    def advance(self) -> None:
        """No-op — Kratos time-stepping is managed by the user script."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_variable(self, field_name: str) -> Any:
        """Resolve a field name to a Kratos variable.

        First checks the user-provided ``variable_map``, then falls back
        to ``KratosMultiphysics.KratosGlobals.GetVariable()``.
        """
        if field_name in self._variable_map:
            return self._variable_map[field_name]
        try:
            import KratosMultiphysics
            return KratosMultiphysics.KratosGlobals.GetVariable(field_name)
        except Exception as exc:
            raise KeyError(
                f"Cannot resolve Kratos variable '{field_name}'. "
                f"Pass it in variable_map or ensure it exists in "
                f"KratosGlobals. Original error: {exc}"
            ) from exc

    @property
    def node_ids(self) -> list[int]:
        """Interface node IDs in consistent order."""
        return self._node_ids

    @property
    def num_nodes(self) -> int:
        """Number of coupling boundary nodes."""
        return self._num_nodes

    def get_field_names(self) -> list[str]:
        """Return user-registered field names."""
        return list(self._variable_map.keys())

    def get_metadata(self) -> dict[str, Any]:
        """Return summary metadata about the Kratos interface."""
        return {
            "solver": "KratosMultiphysics",
            "interface": self._interface_name,
            "num_nodes": self._num_nodes,
        }
