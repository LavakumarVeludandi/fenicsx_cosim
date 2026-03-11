"""
CouplingInterface — The main user-facing API for fenicsx-cosim.

This class integrates the ``MeshExtractor``, ``Communicator``, and
``DataMapper`` into a single, clean entry point that hides all
networking and mapping details from the user.

It is the FEniCSx analogue of the top-level ``CoSimIO.Connect`` /
``ExportData`` / ``ImportData`` workflow in Kratos CoSimIO.

Typical usage (Thermal Solver)
------------------------------
>>> from fenicsx_cosim import CouplingInterface
>>> cosim = CouplingInterface(
...     name="ThermalSolver",
...     partner_name="MechanicalSolver",
... )
>>> cosim.register_interface(mesh, facet_tags, marker_id=1, function_space=V)
>>> cosim.export_data("TemperatureField", temperature)
>>> cosim.import_data("DisplacementField", displacement)
>>> cosim.advance_in_time()
"""

from __future__ import annotations

import json
from typing import Literal, Optional

import numpy as np

from fenicsx_cosim.communicator import Communicator
from fenicsx_cosim.data_mapper import DataMapper, NearestNeighborMapper
from fenicsx_cosim.mesh_extractor import MeshExtractor
from fenicsx_cosim.utils import get_logger

logger = get_logger(__name__)

# Lazy dolfinx import
try:
    import dolfinx
    import dolfinx.fem
    import dolfinx.mesh
except ImportError:
    pass  # Will be caught by MeshExtractor


class CouplingInterface:
    """High-level co-simulation coupling API for FEniCSx solvers.

    This is the **single entry point** that researchers interact with.
    It orchestrates boundary-data extraction, inter-process
    communication, and (optionally) non-conforming mesh mapping.

    Parameters
    ----------
    name : str
        A unique identifier for *this* solver instance (e.g.
        ``"ThermalSolver"``).
    partner_name : str
        The identifier of the partner solver.
    role : {"bind", "connect"}, optional
        ZeroMQ role.  The solver that starts first should ``"bind"``;
        the one that starts second should ``"connect"``.  If ``None``
        (default), the role is determined automatically by sorting the
        names lexicographically — the name that comes first binds.
    endpoint : str, optional
        ZeroMQ endpoint.  Default ``"tcp://localhost:5555"``.
    connection_type : {"tcp", "ipc"}, optional
        Shortcut to auto-generate an endpoint.  Ignored if *endpoint*
        is explicitly provided.
    timeout_ms : int, optional
        Receive timeout in ms (default 30 000).
    enable_mapping : bool, optional
        If ``True`` (default), a ``NearestNeighborMapper`` is built
        automatically when both sides exchange boundary coordinates.
    """

    def __init__(
        self,
        name: str,
        partner_name: str,
        role: Optional[Literal["bind", "connect"]] = None,
        endpoint: Optional[str] = None,
        connection_type: str = "tcp",
        timeout_ms: int = 30_000,
        enable_mapping: bool = True,
    ) -> None:
        self.name = name
        self.partner_name = partner_name
        self.enable_mapping = enable_mapping

        # --- Determine role automatically if not specified ---------------
        if role is None:
            role = "bind" if name < partner_name else "connect"
            logger.info(
                "[%s] Auto-assigned role='%s' (partner='%s')",
                name, role, partner_name,
            )
        self.role: Literal["bind", "connect"] = role

        # --- Determine endpoint -----------------------------------------
        if endpoint is None:
            # Create a deterministic connection name from sorted solver names
            conn_name = "_".join(sorted([name, partner_name]))
            if connection_type == "ipc":
                endpoint = f"ipc:///tmp/fenicsx_cosim_{conn_name}"
            else:
                endpoint = "tcp://localhost:5555"
        self.endpoint = endpoint

        # --- Internal components ----------------------------------------
        self._extractor = MeshExtractor()
        self._communicator = Communicator(
            name=name,
            partner_name=partner_name,
            role=self.role,
            endpoint=self.endpoint,
            timeout_ms=timeout_ms,
            handshake=True,
        )
        self._mapper: Optional[DataMapper] = None
        self._partner_coords: Optional[np.ndarray] = None
        self._function_space = None
        self._registered = False
        self._step_count = 0

        logger.info(
            "[%s] CouplingInterface ready — partner='%s', endpoint='%s'",
            name, partner_name, self.endpoint,
        )

    # ------------------------------------------------------------------
    # Interface registration
    # ------------------------------------------------------------------

    def register_interface(
        self,
        mesh: "dolfinx.mesh.Mesh",
        facet_tags: "dolfinx.mesh.MeshTags",
        marker_id: int,
        function_space: Optional["dolfinx.fem.FunctionSpace"] = None,
    ) -> None:
        """Register the coupling boundary and exchange coordinates with the
        partner solver.

        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
            The computational mesh.
        facet_tags : dolfinx.mesh.MeshTags
            Mesh tags labelling boundary facets.
        marker_id : int
            The integer marker for the coupling boundary.
        function_space : dolfinx.fem.FunctionSpace, optional
            The function space. If ``None``, a default scalar Lagrange-1
            space is created on *mesh*.
        """
        # Default function space
        if function_space is None:
            function_space = dolfinx.fem.functionspace(
                mesh, ("Lagrange", 1)
            )
        self._function_space = function_space

        # Extract boundary information
        self._extractor.register(mesh, facet_tags, marker_id, function_space)

        # Exchange boundary coordinates with partner for mapping
        my_coords = self._extractor.boundary_coordinates
        self._exchange_coordinates(my_coords)

        self._registered = True
        logger.info(
            "[%s] Interface registered — %d boundary DoFs",
            self.name, len(self._extractor.boundary_dof_indices),
        )

    def register_interface_from_locator(
        self,
        mesh: "dolfinx.mesh.Mesh",
        locator_fn,
        function_space: Optional["dolfinx.fem.FunctionSpace"] = None,
        marker_id: int = 0,
    ) -> None:
        """Register the coupling boundary using a geometric locator function.

        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
            The computational mesh.
        locator_fn : callable
            ``locator_fn(x: ndarray) -> ndarray[bool]``
        function_space : dolfinx.fem.FunctionSpace, optional
            Defaults to Lagrange-1.
        marker_id : int, optional
            Label for this boundary.
        """
        if function_space is None:
            function_space = dolfinx.fem.functionspace(
                mesh, ("Lagrange", 1)
            )
        self._function_space = function_space

        self._extractor.register_from_locator(
            mesh, locator_fn, function_space, marker_id
        )

        my_coords = self._extractor.boundary_coordinates
        self._exchange_coordinates(my_coords)

        self._registered = True
        logger.info(
            "[%s] Interface registered (locator) — %d boundary DoFs",
            self.name, len(self._extractor.boundary_dof_indices),
        )

    def _exchange_coordinates(self, my_coords: np.ndarray) -> None:
        """Exchange boundary coordinates with the partner and build the
        data mapper if needed.
        """
        if self.role == "bind":
            # Send our coordinates, then receive partner's
            self._communicator.send_array("boundary_coords", my_coords)
            _, partner_coords = self._communicator.receive_array()
        else:
            # Receive partner's coordinates, then send ours
            _, partner_coords = self._communicator.receive_array()
            self._communicator.send_array("boundary_coords", my_coords)

        self._partner_coords = partner_coords
        logger.info(
            "[%s] Exchanged coordinates — local: %d pts, partner: %d pts",
            self.name, len(my_coords), len(partner_coords),
        )

        # Build the mapper
        if self.enable_mapping and len(my_coords) > 0 and len(partner_coords) > 0:
            self._mapper = NearestNeighborMapper()
            # The mapper maps from partner (source) to us (target)
            self._mapper.build(
                source_coords=partner_coords,
                target_coords=my_coords,
            )

    # ------------------------------------------------------------------
    # Data exchange
    # ------------------------------------------------------------------

    def export_data(
        self,
        data_name: str,
        function: "dolfinx.fem.Function",
    ) -> None:
        """Extract boundary values from a FEniCSx Function and send them
        to the partner solver.

        Parameters
        ----------
        data_name : str
            Identifier for the data (e.g. ``"TemperatureField"``).
        function : dolfinx.fem.Function
            The FEniCSx function whose boundary values are to be sent.
        """
        self._check_registered()
        values = self._extractor.extract_boundary_values(function)
        self._communicator.send_array(data_name, values)
        logger.debug(
            "[%s] Exported '%s' — %d values", self.name, data_name, len(values)
        )

    def import_data(
        self,
        data_name: str,
        function: "dolfinx.fem.Function",
    ) -> None:
        """Receive boundary values from the partner solver and inject
        them into a FEniCSx Function.

        If the meshes are non-conforming (different DoF coordinates),
        the received values are first mapped using the ``DataMapper``
        before injection.

        Parameters
        ----------
        data_name : str
            Expected identifier for the incoming data.
        function : dolfinx.fem.Function
            The target FEniCSx function.
        """
        self._check_registered()
        received_name, received_values = self._communicator.receive_array()

        if received_name != data_name:
            logger.warning(
                "[%s] Expected data '%s' but received '%s'",
                self.name, data_name, received_name,
            )

        # Apply mapping if meshes differ
        if self._mapper is not None:
            mapped_values = self._mapper.map(received_values)
        else:
            mapped_values = received_values

        self._extractor.inject_boundary_values(function, mapped_values)
        logger.debug(
            "[%s] Imported '%s' — %d values injected",
            self.name, received_name, len(mapped_values),
        )

    # ------------------------------------------------------------------
    # Synchronization
    # ------------------------------------------------------------------

    def advance_in_time(self) -> None:
        """Synchronize both solvers at the end of a time step.

        This acts as a barrier — neither solver proceeds until both
        have called ``advance_in_time()``.
        """
        self._check_registered()
        self._communicator.synchronize()
        self._step_count += 1
        logger.debug(
            "[%s] Time step %d complete", self.name, self._step_count
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def boundary_coordinates(self) -> np.ndarray:
        """Boundary DoF coordinates of *this* solver — shape ``(N, 3)``."""
        self._check_registered()
        return self._extractor.boundary_coordinates

    @property
    def partner_coordinates(self) -> Optional[np.ndarray]:
        """Boundary DoF coordinates received from the partner."""
        return self._partner_coords

    @property
    def mapper(self) -> Optional[DataMapper]:
        """The data mapper (if mapping is enabled and built)."""
        return self._mapper

    @property
    def step_count(self) -> int:
        """Number of completed time steps."""
        return self._step_count

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def disconnect(self) -> None:
        """Gracefully close the connection to the partner solver."""
        if self._communicator is not None:
            self._communicator.close()
        logger.info("[%s] Disconnected", self.name)

    def _check_registered(self) -> None:
        if not self._registered:
            raise RuntimeError(
                f"[{self.name}] No interface registered. "
                "Call register_interface() first."
            )

    def __enter__(self) -> "CouplingInterface":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    def __del__(self) -> None:
        self.disconnect()
