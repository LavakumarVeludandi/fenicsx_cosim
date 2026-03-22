"""
CouplingInterface — The main user-facing API for fenicsx-cosim.

This class integrates the ``MeshExtractor``, ``Communicator``, and
``DataMapper`` into a single, clean entry point that hides all
networking and mapping details from the user.

It is the FEniCSx analogue of the top-level ``CoSimIO.Connect`` /
``ExportData`` / ``ImportData`` workflow in Kratos CoSimIO.

Additionally, it now supports:

* **Adaptive Mesh Refinement (AMR):** via ``update_interface_geometry()``
  which re-negotiates the mapping with the partner solver.
* **FE² Multiscale Homogenization:** via ``register_quadrature_space()``,
  ``scatter_data()``, and ``gather_data()`` for dispatching integration-
  point tensors to a pool of RVE solver workers.

Typical usage (Standard Boundary Coupling)
-------------------------------------------
>>> from fenicsx_cosim import CouplingInterface
>>> cosim = CouplingInterface(
...     name="ThermalSolver",
...     partner_name="MechanicalSolver",
... )
>>> cosim.register_interface(mesh, facet_tags, marker_id=1, function_space=V)
>>> cosim.export_data("TemperatureField", temperature)
>>> cosim.import_data("DisplacementField", displacement)
>>> cosim.advance_in_time()

Typical usage (FE² Scatter-Gather)
-----------------------------------
>>> cosim_fe2 = CouplingInterface(name="Macro", role="Master",
...     topology="scatter-gather")
>>> cosim_fe2.register_quadrature_space(V_quad)
>>> cosim_fe2.scatter_data("StrainTensor", macro_strains)
>>> stresses = cosim_fe2.gather_data("StressTensor")
"""

from __future__ import annotations

import json
from typing import Literal, Optional

import numpy as np

from fenicsx_cosim.adapters.base import SolverAdapter
from fenicsx_cosim.communicator import Communicator
from fenicsx_cosim.data_mapper import DataMapper, NearestNeighborMapper
from fenicsx_cosim.dynamic_mapper import DynamicMapper
from fenicsx_cosim.mesh_extractor import MeshExtractor
from fenicsx_cosim.quadrature_extractor import QuadratureExtractor
from fenicsx_cosim.scatter_gather_communicator import ScatterGatherCommunicator
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
        The identifier of the partner solver.  For ``scatter-gather``
        topology, this can be set to ``"Workers"``.
    role : {"bind", "connect", "Master", "Worker"}, optional
        ZeroMQ role.  The solver that starts first should ``"bind"``;
        the one that starts second should ``"connect"``.  If ``None``
        (default), the role is determined automatically by sorting the
        names lexicographically — the name that comes first binds.
        Use ``"Master"`` / ``"Worker"`` for scatter-gather topology.
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
    topology : {"pair", "scatter-gather"}, optional
        Communication topology.  ``"pair"`` (default) uses the standard
        ``zmq.PAIR`` socket for 1-to-1 coupling.  ``"scatter-gather"``
        uses ``PUSH/PULL`` for FE² fan-out / fan-in.
    push_endpoint : str, optional
        For ``scatter-gather`` only: PUSH socket endpoint.
    pull_endpoint : str, optional
        For ``scatter-gather`` only: PULL socket endpoint.
    """

    def __init__(
        self,
        name: str,
        partner_name: str = "Workers",
        role: Optional[str] = None,
        endpoint: Optional[str] = None,
        connection_type: str = "tcp",
        timeout_ms: int = 30_000,
        enable_mapping: bool = True,
        topology: str = "pair",
        push_endpoint: str = "tcp://*:5556",
        pull_endpoint: str = "tcp://*:5557",
    ) -> None:
        self.name = name
        self.partner_name = partner_name
        self.enable_mapping = enable_mapping
        self.topology = topology

        # --- Internal components (common) --------------------------------
        self._extractor: Optional[MeshExtractor] = None
        self._adapter: Optional[SolverAdapter] = None
        self._quad_extractor: Optional[QuadratureExtractor] = None
        self._mapper: Optional[DataMapper] = None
        self._dynamic_mapper: Optional[DynamicMapper] = None
        self._partner_coords: Optional[np.ndarray] = None
        self._function_space = None
        self._registered = False
        self._adapter_registered = False
        self._quad_registered = False
        self._step_count = 0
        self._disconnected = False

        # --- Topology-specific setup ------------------------------------
        if topology == "scatter-gather":
            # FE² scatter-gather mode
            sg_role = "master" if role in ("Master", "bind", None) else "worker"
            self._sg_communicator = ScatterGatherCommunicator(
                role=sg_role,
                push_endpoint=push_endpoint,
                pull_endpoint=pull_endpoint,
                timeout_ms=timeout_ms,
            )
            self._communicator = None
            self.role = sg_role
            logger.info(
                "[%s] CouplingInterface ready (scatter-gather, role=%s)",
                name, sg_role,
            )
        else:
            # Standard PAIR mode
            self._sg_communicator = None

            # Determine role automatically if not specified
            if role is None:
                role = "bind" if name < partner_name else "connect"
                logger.info(
                    "[%s] Auto-assigned role='%s' (partner='%s')",
                    name, role, partner_name,
                )
            elif role == "Master":
                role = "bind"
            elif role == "Worker":
                role = "connect"
            self.role = role

            # Determine endpoint
            if endpoint is None:
                conn_name = "_".join(sorted([name, partner_name]))
                if connection_type == "ipc":
                    endpoint = f"ipc:///tmp/fenicsx_cosim_{conn_name}"
                else:
                    endpoint = "tcp://localhost:5555"
            self.endpoint = endpoint

            self._communicator = Communicator(
                name=name,
                partner_name=partner_name,
                role=self.role,
                endpoint=self.endpoint,
                timeout_ms=timeout_ms,
                handshake=True,
            )

            logger.info(
                "[%s] CouplingInterface ready — partner='%s', endpoint='%s'",
                name, partner_name, self.endpoint,
            )

    # ==================================================================
    # Alternative constructor: from any SolverAdapter
    # ==================================================================

    @classmethod
    def from_adapter(
        cls,
        adapter: SolverAdapter,
        name: str,
        partner_name: str = "Partner",
        role: Optional[str] = None,
        endpoint: Optional[str] = None,
        connection_type: str = "tcp",
        timeout_ms: int = 30_000,
        enable_mapping: bool = True,
    ) -> "CouplingInterface":
        """Create a CouplingInterface using any :class:`SolverAdapter`.

        This allows non-FEniCSx solvers (Kratos, Abaqus, etc.) to
        participate in fenicsx-cosim coupling via their respective
        adapters.

        Parameters
        ----------
        adapter : SolverAdapter
            A concrete adapter for the solver.
        name : str
            Identifier for this solver.
        partner_name : str, optional
            Identifier of the partner solver.
        role : {"bind", "connect"}, optional
            ZeroMQ role.  Auto-determined if ``None``.
        endpoint : str, optional
            ZeroMQ endpoint.
        connection_type : {"tcp", "ipc"}, optional
            Transport type.
        timeout_ms : int, optional
            Receive timeout.
        enable_mapping : bool, optional
            Build nearest-neighbour mapper.

        Returns
        -------
        CouplingInterface
        """
        instance = cls(
            name=name,
            partner_name=partner_name,
            role=role,
            endpoint=endpoint,
            connection_type=connection_type,
            timeout_ms=timeout_ms,
            enable_mapping=enable_mapping,
            topology="pair",
        )
        instance._adapter = adapter
        return instance

    def register_adapter_interface(self) -> None:
        """Register the coupling interface using the attached adapter.

        Extracts boundary coordinates from the adapter and exchanges
        them with the partner solver to build the data mapping.

        Raises
        ------
        RuntimeError
            If no adapter has been set (use :meth:`from_adapter`).
        """
        if self._adapter is None:
            raise RuntimeError(
                f"[{self.name}] No adapter set. "
                "Create via CouplingInterface.from_adapter()."
            )
        my_coords = self._adapter.get_boundary_coordinates()
        self._exchange_coordinates(my_coords)
        self._adapter_registered = True
        logger.info(
            "[%s] Adapter interface registered — %d boundary nodes",
            self.name, len(my_coords),
        )

    def _ensure_extractor(self) -> None:
        """Lazily instantiate MeshExtractor if needed."""
        if self._extractor is None:
            from fenicsx_cosim.mesh_extractor import MeshExtractor
            self._extractor = MeshExtractor()

    def _ensure_quad_extractor(self) -> None:
        """Lazily instantiate QuadratureExtractor if needed."""
        if self._quad_extractor is None:
            from fenicsx_cosim.quadrature_extractor import QuadratureExtractor
            self._quad_extractor = QuadratureExtractor()

    def export_via_adapter(
        self,
        field_name: str,
    ) -> None:
        """Extract field from adapter and send to partner.

        Parameters
        ----------
        field_name : str
            Field name as understood by the adapter.
        """
        self._check_adapter_registered()
        values = self._adapter.extract_field(field_name)
        self._communicator.send_array(field_name, values)
        logger.debug(
            "[%s] Exported '%s' via adapter — %d values",
            self.name, field_name, len(values),
        )

    def import_via_adapter(
        self,
        field_name: str,
    ) -> None:
        """Receive field from partner and inject into adapter.

        Parameters
        ----------
        field_name : str
            Expected field name.
        """
        self._check_adapter_registered()
        received_name, received_values = self._communicator.receive_array()

        if received_name != field_name:
            logger.warning(
                "[%s] Expected '%s' but received '%s'",
                self.name, field_name, received_name,
            )

        # Apply mapping if meshes differ
        if self._mapper is not None:
            mapped_values = self._mapper.map(received_values)
        else:
            mapped_values = received_values

        self._adapter.inject_field(field_name, mapped_values)
        logger.debug(
            "[%s] Imported '%s' via adapter — %d values injected",
            self.name, received_name, len(mapped_values),
        )

    def advance_adapter(self) -> None:
        """Synchronize and advance the adapter's solver."""
        self._check_adapter_registered()
        self._communicator.synchronize()
        self._adapter.advance()
        self._step_count += 1
        logger.debug(
            "[%s] Adapter time step %d complete",
            self.name, self._step_count,
        )

    def _check_adapter_registered(self) -> None:
        if not self._adapter_registered:
            raise RuntimeError(
                f"[{self.name}] No adapter interface registered. "
                "Call register_adapter_interface() first."
            )

    # ==================================================================
    # Interface registration (standard boundary coupling)
    # ==================================================================

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
        self._ensure_extractor()
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

        self._ensure_extractor()
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
            # Use the DynamicMapper so AMR updates are seamless
            self._dynamic_mapper = DynamicMapper()
            self._dynamic_mapper.build(
                source_coords=partner_coords,
                target_coords=my_coords,
            )
            # Also keep a reference as the generic mapper
            self._mapper = NearestNeighborMapper()
            self._mapper.build(
                source_coords=partner_coords,
                target_coords=my_coords,
            )

    # ==================================================================
    # AMR: Update interface geometry
    # ==================================================================

    def update_interface_geometry(
        self,
        mesh: "dolfinx.mesh.Mesh",
        facet_tags: "dolfinx.mesh.MeshTags",
        marker_id: int,
        function_space: Optional["dolfinx.fem.FunctionSpace"] = None,
    ) -> None:
        """Notify the coupling system that the mesh has been refined (AMR).

        This method:

        1. Re-extracts boundary DoFs from the refined mesh.
        2. Invalidates the current data mapping.
        3. Negotiates a synchronized coordinate exchange with the
           partner solver.
        4. Rebuilds the ``DynamicMapper`` with the updated point clouds.

        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
            The **newly refined** mesh.
        facet_tags : dolfinx.mesh.MeshTags
            Updated facet tags for the refined mesh.
        marker_id : int
            The coupling boundary marker.
        function_space : dolfinx.fem.FunctionSpace, optional
            Updated function space (if ``None``, a new Lagrange-1 space
            is created on the refined mesh).

        See Also
        --------
        DynamicMapper.negotiate_update
        """
        self._check_registered()

        if function_space is None:
            function_space = dolfinx.fem.functionspace(
                mesh, ("Lagrange", 1)
            )
        self._function_space = function_space

        # Re-extract boundary data from the refined mesh
        self._extractor.register(mesh, facet_tags, marker_id, function_space)
        my_new_coords = self._extractor.boundary_coordinates

        # Invalidate and negotiate with partner
        if self._dynamic_mapper is not None:
            self._dynamic_mapper.invalidate()
            partner_coords = self._dynamic_mapper.negotiate_update(
                communicator=self._communicator,
                role=self.role,
                my_new_coords=my_new_coords,
            )
            if partner_coords is not None:
                self._partner_coords = partner_coords

                # Also rebuild the plain NearestNeighborMapper
                self._mapper = NearestNeighborMapper()
                self._mapper.build(
                    source_coords=partner_coords,
                    target_coords=my_new_coords,
                )
        else:
            # Fallback: full coordinate re-exchange
            self._exchange_coordinates(my_new_coords)

        logger.info(
            "[%s] Interface geometry updated (AMR) — %d boundary DoFs",
            self.name, len(self._extractor.boundary_dof_indices),
        )

    def check_mesh_update(self) -> bool:
        """Lightweight check to see if the partner solver updated its mesh.

        If the partner calls ``update_interface_geometry()``, *this* solver
        must call ``check_mesh_update()`` at the same point in the time
        loop to receive the new coordinates and re-build the mapping.

        If *neither* solver refined its mesh but AMR is active, both solvers
        should call ``check_mesh_update()`` to acknowledge no changes.

        Returns
        -------
        bool
            ``True`` if the partner updated its mesh and the mapping was
            rebuilt; ``False`` otherwise.
        """
        self._check_registered()
        if self._dynamic_mapper is None:
            return False

        partner_coords = self._dynamic_mapper.negotiate_update(
            communicator=self._communicator,
            role=self.role,
            my_new_coords=None,
        )

        if partner_coords is not None:
            self._partner_coords = partner_coords
            self._mapper = NearestNeighborMapper()
            self._mapper.build(
                source_coords=partner_coords,
                target_coords=self._extractor.boundary_coordinates,
            )
            logger.info("[%s] Responded to partner's mesh update.", self.name)
            return True
        return False

    # ==================================================================
    # FE²: Quadrature space registration
    # ==================================================================

    def register_quadrature_space(
        self,
        function_space_or_mesh=None,
        quadrature_degree: int = 2,
        tensor_shape=(),
    ) -> None:
        """Register a Quadrature function space for FE² homogenization.

        This replaces the standard boundary registration and instead
        sets up data exchange at integration points.

        Parameters
        ----------
        function_space_or_mesh : dolfinx.fem.FunctionSpace or dolfinx.mesh.Mesh
            Either a pre-existing Quadrature function space, or a mesh
            on which to create one.
        quadrature_degree : int, optional
            If a mesh is provided, the quadrature degree.  Default 2.
        tensor_shape : tuple[int, ...], optional
            Shape of the tensor at each integration point.
        """
        self._ensure_quad_extractor()

        # Determine if we got a function space or a mesh
        if hasattr(function_space_or_mesh, "dofmap"):
            # It's a function space
            self._quad_extractor.register_with_function_space(
                function_space_or_mesh, tensor_shape
            )
        else:
            # It's a mesh
            self._quad_extractor.register(
                function_space_or_mesh,
                quadrature_degree=quadrature_degree,
                tensor_shape=tensor_shape,
            )

        self._quad_registered = True
        qd = self._quad_extractor.quadrature_data
        logger.info(
            "[%s] Quadrature space registered — %d cells × %d pts/cell "
            "= %d total integration points",
            self.name, qd.num_cells, qd.points_per_cell, qd.total_points,
        )

    # ==================================================================
    # FE²: Scatter / Gather
    # ==================================================================

    def scatter_data(
        self,
        data_name: str,
        function: "dolfinx.fem.Function",
        metadata: Optional[list[dict]] = None,
    ) -> int:
        """Extract per-cell integration-point values and scatter them to
        the RVE worker pool.

        Parameters
        ----------
        data_name : str
            Name identifier (e.g. ``"StrainTensor"``).
        function : dolfinx.fem.Function
            Source function on the Quadrature space.
        metadata : list[dict], optional
            Optional per-cell metadata (material params, damage state).

        Returns
        -------
        int
            Number of work items dispatched.
        """
        self._check_scatter_gather("scatter_data")
        cell_values = self._quad_extractor.extract_for_dispatch(function)
        n = self._sg_communicator.scatter(cell_values, metadata)
        logger.info(
            "[%s] Scattered '%s' — %d cells", self.name, data_name, n
        )
        return n

    def gather_data(
        self,
        data_name: str,
        function: Optional["dolfinx.fem.Function"] = None,
        n_expected: Optional[int] = None,
    ) -> list[np.ndarray]:
        """Gather results from the RVE worker pool and optionally inject
        them into a Quadrature function.

        Parameters
        ----------
        data_name : str
            Name identifier (e.g. ``"StressTensor"``).
        function : dolfinx.fem.Function, optional
            If provided, the gathered values are injected into this
            Quadrature function.
        n_expected : int, optional
            Number of results to wait for.  Defaults to
            ``quadrature_data.num_cells``.

        Returns
        -------
        list[np.ndarray]
            The gathered per-cell result arrays.
        """
        self._check_scatter_gather("gather_data")
        if n_expected is None:
            n_expected = self._quad_extractor.quadrature_data.num_cells

        results = self._sg_communicator.gather(n_expected)
        logger.info(
            "[%s] Gathered '%s' — %d results", self.name, data_name, len(results)
        )

        if function is not None:
            self._quad_extractor.inject_from_gather(function, results)
            logger.debug(
                "[%s] Injected '%s' into Quadrature function",
                self.name, data_name,
            )

        return results

    def scatter_gather_data(
        self,
        scatter_name: str,
        scatter_function: "dolfinx.fem.Function",
        gather_name: str,
        gather_function: Optional["dolfinx.fem.Function"] = None,
        metadata: Optional[list[dict]] = None,
    ) -> list[np.ndarray]:
        """Convenience: scatter, then gather in one call.

        Parameters
        ----------
        scatter_name : str
            Name for the outgoing data.
        scatter_function : dolfinx.fem.Function
            Source Quadrature function.
        gather_name : str
            Name for the incoming data.
        gather_function : dolfinx.fem.Function, optional
            Target Quadrature function for injection.
        metadata : list[dict], optional
            Per-cell metadata.

        Returns
        -------
        list[np.ndarray]
            Gathered results.
        """
        n = self.scatter_data(scatter_name, scatter_function, metadata)
        return self.gather_data(gather_name, gather_function, n)

    # ==================================================================
    # Standard data exchange (boundary coupling)
    # ==================================================================

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

    def export_raw(self, data_name: str, array: np.ndarray | "sps.csr_matrix") -> None:
        """Send a raw NumPy array or SciPy sparse matrix to the partner solver.

        This is useful for sending non-field data like sparse stiffness matrices
        or scalar values (e.g. for Shakedown analysis or custom optimization loops).

        Parameters
        ----------
        data_name : str
            Identifier for the data.
        array : np.ndarray or scipy.sparse.spmatrix
            The raw data to transmit.
        """
        self._communicator.send_array(data_name, array)
        if hasattr(array, "shape"):
            size = array.shape
        else:
            size = len(array)
        logger.debug("[%s] Exported raw data '%s' — shape %s", self.name, data_name, size)

    def import_raw(self, data_name: str) -> np.ndarray | "sps.csr_matrix":
        """Receive a raw NumPy array or SciPy sparse matrix from the partner solver.

        Parameters
        ----------
        data_name : str
            Expected identifier for the incoming data.

        Returns
        -------
        np.ndarray or scipy.sparse.spmatrix
            The raw data received.
        """
        received_name, received_values = self._communicator.receive_array()

        if received_name != data_name:
            logger.warning(
                "[%s] Expected raw data '%s' but received '%s'",
                self.name, data_name, received_name,
            )

        if hasattr(received_values, "shape"):
            size = received_values.shape
        else:
            size = len(received_values)
        logger.debug("[%s] Imported raw data '%s' — shape %s", self.name, received_name, size)
        
        return received_values

    # ==================================================================
    # Synchronization
    # ==================================================================

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

    # ==================================================================
    # Accessors
    # ==================================================================

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
    def dynamic_mapper(self) -> Optional[DynamicMapper]:
        """The dynamic (AMR-aware) mapper, if active."""
        return self._dynamic_mapper

    @property
    def quadrature_extractor(self) -> Optional[QuadratureExtractor]:
        """The quadrature extractor, if registered."""
        return self._quad_extractor

    @property
    def step_count(self) -> int:
        """Number of completed time steps."""
        return self._step_count

    @property
    def extractor(self) -> MeshExtractor:
        """The underlying mesh extractor."""
        return self._extractor

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def disconnect(self) -> None:
        """Gracefully close the connection to the partner solver."""
        if getattr(self, "_disconnected", False):
            return
        if self._communicator is not None:
            self._communicator.close()
        if self._sg_communicator is not None:
            self._sg_communicator.close()
        logger.info("[%s] Disconnected", self.name)
        self._disconnected = True

    def _check_registered(self) -> None:
        if not self._registered:
            raise RuntimeError(
                f"[{self.name}] No interface registered. "
                "Call register_interface() first."
            )

    def _check_scatter_gather(self, method: str) -> None:
        if self._sg_communicator is None:
            raise RuntimeError(
                f"[{self.name}] {method}() requires topology='scatter-gather'. "
                "Initialize with: CouplingInterface(..., topology='scatter-gather')"
            )
        if not self._quad_registered:
            raise RuntimeError(
                f"[{self.name}] No quadrature space registered. "
                "Call register_quadrature_space() first."
            )

    def __enter__(self) -> "CouplingInterface":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self.disconnect()
        except Exception:
            pass

    def __del__(self) -> None:
        if not getattr(self, "_disconnected", True):
            try:
                self.disconnect()
            except Exception:
                pass
