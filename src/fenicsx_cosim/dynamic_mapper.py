"""
DynamicMapper — Adaptive mesh-aware data mapping for AMR & Shakedown.

Standard co-simulation assumes a *static* coupling interface: the boundary
coordinates are exchanged once during ``register_interface()`` and never
change.  When Adaptive Mesh Refinement (AMR) is used
(``dolfinx.mesh.refine``), the DoF coordinates change, rendering the
previous ``scipy.spatial.KDTree`` and its mapping indices invalid.

The ``DynamicMapper`` solves this by:

1. **Detecting** that the underlying mesh has changed (via a revision
   counter or explicit user call).
2. **Broadcasting** an ``UPDATE_MESH`` signal over ZeroMQ to the partner
   solver so *both* processes pause, exchange their newly-refined boundary
   point clouds, and rebuild the interpolation mapping.
3. **Rebuilding** the KDTree and mapping indices atomically, ensuring
   the next ``export_data`` / ``import_data`` call uses the updated geometry.

This corresponds to Section 1.1 of the Advanced Features Addendum in the
project specification.

Typical usage (inside a time loop)
-----------------------------------
>>> if error_too_high:
...     domain = dolfinx.mesh.refine(domain, markers)
...     cosim.update_interface_geometry(domain, facet_tags, marker_id=1)
>>> cosim.export_data("Temperature", temperature)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from fenicsx_cosim.data_mapper import DataMapper, NearestNeighborMapper
from fenicsx_cosim.utils import get_logger

logger = get_logger(__name__)

# Protocol signals for mesh-update negotiation
UPDATE_MESH_SIGNAL = b"COSIM_UPDATE_MESH"
UPDATE_MESH_ACK = b"COSIM_UPDATE_MESH_ACK"
NO_UPDATE_SIGNAL = b"COSIM_NO_UPDATE"


class DynamicMapper:
    """Mesh-change–aware wrapper around :class:`DataMapper`.

    The ``DynamicMapper`` keeps track of a *revision counter* which is
    bumped every time the mesh changes.  It provides helpers to negotiate
    a synchronized re-mapping with the partner solver.

    Parameters
    ----------
    base_mapper_cls : type[DataMapper], optional
        The concrete mapper to use.  Defaults to
        :class:`NearestNeighborMapper`.

    Attributes
    ----------
    revision : int
        The number of times the mapping has been (re-)built.
    max_distance : float or None
        Delegates to the underlying ``DataMapper.max_distance``.
    """

    def __init__(
        self,
        base_mapper_cls: type[DataMapper] = NearestNeighborMapper,
    ) -> None:
        self._mapper_cls = base_mapper_cls
        self._mapper: Optional[DataMapper] = None
        self._local_coords: Optional[np.ndarray] = None
        self._partner_coords: Optional[np.ndarray] = None
        self.revision: int = 0
        self._needs_update: bool = False

    # ------------------------------------------------------------------
    # Building / rebuilding
    # ------------------------------------------------------------------

    def build(
        self,
        source_coords: np.ndarray,
        target_coords: np.ndarray,
    ) -> None:
        """Build (or rebuild) the underlying mapping.

        Parameters
        ----------
        source_coords : np.ndarray
            Partner (source) boundary coordinates — shape ``(N_src, 3)``.
        target_coords : np.ndarray
            Local (target) boundary coordinates — shape ``(N_tgt, 3)``.
        """
        self._partner_coords = source_coords.copy()
        self._local_coords = target_coords.copy()

        self._mapper = self._mapper_cls()
        self._mapper.build(source_coords, target_coords)

        self.revision += 1
        self._needs_update = False
        logger.info(
            "DynamicMapper built (revision %d) — %d src, %d tgt pts",
            self.revision,
            len(source_coords),
            len(target_coords),
        )

    def invalidate(self) -> None:
        """Mark the current mapping as stale.

        This should be called whenever the *local* mesh has been refined
        or changed.  The mapping will not be usable until :meth:`build`
        is called again.
        """
        self._needs_update = True
        logger.info(
            "DynamicMapper invalidated (was revision %d)", self.revision
        )

    @property
    def needs_update(self) -> bool:
        """``True`` if the mapping has been invalidated and must be rebuilt."""
        return self._needs_update

    # ------------------------------------------------------------------
    # Negotiation helpers
    # ------------------------------------------------------------------

    def negotiate_update(
        self,
        communicator,
        role: str,
        my_new_coords: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Negotiate a mesh-update with the partner solver.

        Both solvers must call this method at the same point in the time
        loop.  The protocol:

        1. The ``bind`` side sends either ``UPDATE_MESH_SIGNAL`` or
           ``NO_UPDATE_SIGNAL``.
        2. The ``connect`` side receives the signal and then sends its
           own signal (likewise ``UPDATE_MESH_SIGNAL`` or
           ``NO_UPDATE_SIGNAL``).
        3. If *either* side signals an update, both exchange new
           boundary coordinates and return the partner's new coordinates.
           Otherwise ``None`` is returned and the mapping is untouched.

        Parameters
        ----------
        communicator : Communicator
            The ZeroMQ communicator already connected to the partner.
        role : str
            ``"bind"`` or ``"connect"``.
        my_new_coords : np.ndarray or None
            The new local boundary coordinates if the mesh changed, or
            ``None`` if no change occurred.

        Returns
        -------
        np.ndarray or None
            The partner's new boundary coordinates if an exchange took
            place, or ``None`` if no update was needed.
        """
        my_signal = UPDATE_MESH_SIGNAL if self._needs_update else NO_UPDATE_SIGNAL

        if role == "bind":
            communicator.send_raw(my_signal)
            partner_signal = communicator.receive_raw()
        else:
            partner_signal = communicator.receive_raw()
            communicator.send_raw(my_signal)

        either_needs_update = (
            my_signal == UPDATE_MESH_SIGNAL
            or partner_signal == UPDATE_MESH_SIGNAL
        )

        if not either_needs_update:
            logger.debug("DynamicMapper: no update needed on either side")
            return None

        logger.info(
            "DynamicMapper: mesh update detected (local=%s, partner=%s)",
            my_signal == UPDATE_MESH_SIGNAL,
            partner_signal == UPDATE_MESH_SIGNAL,
        )

        # Exchange new coordinates
        if my_new_coords is None:
            # Our mesh didn't change — re-send old coords
            my_new_coords = self._local_coords

        if role == "bind":
            communicator.send_array("updated_boundary_coords", my_new_coords)
            _, partner_coords = communicator.receive_array()
        else:
            _, partner_coords = communicator.receive_array()
            communicator.send_array("updated_boundary_coords", my_new_coords)

        # Acknowledge
        if role == "bind":
            communicator.send_raw(UPDATE_MESH_ACK)
            ack = communicator.receive_raw()
        else:
            ack = communicator.receive_raw()
            communicator.send_raw(UPDATE_MESH_ACK)

        if ack != UPDATE_MESH_ACK:
            logger.warning("Unexpected ACK during mesh update: %r", ack)

        # Rebuild the mapping
        self.build(
            source_coords=partner_coords,
            target_coords=my_new_coords,
        )

        return partner_coords

    # ------------------------------------------------------------------
    # Delegation to the underlying mapper
    # ------------------------------------------------------------------

    def map(self, source_values: np.ndarray) -> np.ndarray:
        """Map values from partner (source) to local (target) DoFs.

        Raises :class:`RuntimeError` if the mapping is stale.
        """
        self._check_valid()
        return self._mapper.map(source_values)

    def inverse_map(self, target_values: np.ndarray) -> np.ndarray:
        """Map values from local (target) back to partner (source) DoFs."""
        self._check_valid()
        return self._mapper.inverse_map(target_values)

    @property
    def max_distance(self) -> Optional[float]:
        """Maximum mapping distance from the underlying mapper."""
        if self._mapper is None:
            return None
        return getattr(self._mapper, "max_distance", None)

    @property
    def local_coordinates(self) -> Optional[np.ndarray]:
        """The local boundary coordinates used in the latest mapping."""
        return self._local_coords

    @property
    def partner_coordinates(self) -> Optional[np.ndarray]:
        """The partner boundary coordinates used in the latest mapping."""
        return self._partner_coords

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_valid(self) -> None:
        if self._mapper is None:
            raise RuntimeError("DynamicMapper not built. Call build() first.")
        if self._needs_update:
            raise RuntimeError(
                "DynamicMapper mapping is stale — the mesh has changed. "
                "Call negotiate_update() or build() to refresh."
            )
