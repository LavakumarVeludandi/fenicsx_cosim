"""
DataMapper — Interpolation engine for non-conforming mesh boundaries.

When two coupled solvers use different meshes on the coupling boundary,
the DoF coordinates will not coincide.  The ``DataMapper`` bridges this
gap by computing a mapping between the two point clouds and interpolating
field values accordingly.

This corresponds to **Phase 4** of the development roadmap and mirrors
the mapping capabilities provided by the Kratos ``MappingApplication``.

Currently implemented strategies:

* **NearestNeighborMapper** — for each target point, find the closest
  source point and copy its value.  Uses ``scipy.spatial.KDTree`` for
  O(N log N) performance.

Planned (future scope):

* **ProjectionMapper** — Galerkin projection between non-matching
  boundary meshes.

Typical usage
-------------
>>> mapper = NearestNeighborMapper()
>>> mapper.build(source_coords, target_coords)
>>> mapped_values = mapper.map(source_values)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from fenicsx_cosim.utils import get_logger

logger = get_logger(__name__)

try:
    from scipy.spatial import KDTree

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


class DataMapper(ABC):
    """Abstract base class for all data mapping strategies.

    Subclasses must implement :meth:`build` and :meth:`map`.
    """

    @abstractmethod
    def build(
        self,
        source_coords: np.ndarray,
        target_coords: np.ndarray,
    ) -> None:
        """Build the mapping from *source* to *target* point clouds.

        Parameters
        ----------
        source_coords : np.ndarray
            Coordinates of the source DoFs, shape ``(N_src, 3)``.
        target_coords : np.ndarray
            Coordinates of the target DoFs, shape ``(N_tgt, 3)``.
        """

    @abstractmethod
    def map(self, source_values: np.ndarray) -> np.ndarray:
        """Map values from source DoFs to target DoFs.

        Parameters
        ----------
        source_values : np.ndarray
            Field values at the source DoFs, shape ``(N_src,)`` or
            ``(N_src, D)`` for vector fields.

        Returns
        -------
        np.ndarray
            Interpolated values at the target DoFs.
        """

    @abstractmethod
    def inverse_map(self, target_values: np.ndarray) -> np.ndarray:
        """Map values from target DoFs back to source DoFs.

        Parameters
        ----------
        target_values : np.ndarray
            Field values at the target DoFs.

        Returns
        -------
        np.ndarray
            Interpolated values at the source DoFs.
        """


class NearestNeighborMapper(DataMapper):
    """Maps data by assigning each target DoF the value of its nearest
    source DoF.

    This is the simplest (and often most robust) mapping strategy.  It
    is suitable when the two meshes have similar resolution on the
    coupling boundary and the fields vary smoothly.

    Attributes
    ----------
    max_distance : float or None
        After building, the maximum distance between any target point
        and its nearest source point.  Useful for diagnosing mesh
        mismatch issues.
    """

    def __init__(self) -> None:
        if not _HAS_SCIPY:
            raise ImportError(
                "scipy is required for NearestNeighborMapper. "
                "Install it with: pip install scipy"
            )
        self._source_tree: Optional[KDTree] = None
        self._target_tree: Optional[KDTree] = None
        self._forward_indices: Optional[np.ndarray] = None  # target → source
        self._inverse_indices: Optional[np.ndarray] = None  # source → target
        self._forward_distances: Optional[np.ndarray] = None
        self._inverse_distances: Optional[np.ndarray] = None
        self.max_distance: Optional[float] = None

    def build(
        self,
        source_coords: np.ndarray,
        target_coords: np.ndarray,
    ) -> None:
        """Build the nearest-neighbor mapping.

        Parameters
        ----------
        source_coords : np.ndarray
            Shape ``(N_src, 3)``.
        target_coords : np.ndarray
            Shape ``(N_tgt, 3)``.
        """
        logger.info(
            "Building NearestNeighborMapper: %d source pts → %d target pts",
            len(source_coords),
            len(target_coords),
        )

        # Build KDTrees for both directions
        self._source_tree = KDTree(source_coords)
        self._target_tree = KDTree(target_coords)

        # Forward mapping: for each target point, find nearest source point
        self._forward_distances, self._forward_indices = (
            self._source_tree.query(target_coords)
        )

        # Inverse mapping: for each source point, find nearest target point
        self._inverse_distances, self._inverse_indices = (
            self._target_tree.query(source_coords)
        )

        self.max_distance = float(np.max(self._forward_distances))
        logger.info(
            "Mapping built — max forward distance: %.6e", self.max_distance
        )

        if self.max_distance > 1e-6:
            logger.warning(
                "Maximum mapping distance (%.6e) is non-negligible. "
                "The coupling boundaries may not align well.",
                self.max_distance,
            )

    def map(self, source_values: np.ndarray) -> np.ndarray:
        """Map values from source DoFs to target DoFs using nearest neighbor.

        Parameters
        ----------
        source_values : np.ndarray
            Shape ``(N_src,)`` for scalar or ``(N_src, D)`` for vector fields.

        Returns
        -------
        np.ndarray
            Shape ``(N_tgt,)`` or ``(N_tgt, D)``.
        """
        if self._forward_indices is None:
            raise RuntimeError("Mapper not built. Call build() first.")
        return source_values[self._forward_indices]

    def inverse_map(self, target_values: np.ndarray) -> np.ndarray:
        """Map values from target DoFs back to source DoFs.

        Parameters
        ----------
        target_values : np.ndarray
            Shape ``(N_tgt,)`` or ``(N_tgt, D)``.

        Returns
        -------
        np.ndarray
            Shape ``(N_src,)`` or ``(N_src, D)``.
        """
        if self._inverse_indices is None:
            raise RuntimeError("Mapper not built. Call build() first.")
        return target_values[self._inverse_indices]

    @property
    def forward_distances(self) -> np.ndarray:
        """Distances between each target point and its nearest source point."""
        if self._forward_distances is None:
            raise RuntimeError("Mapper not built.")
        return self._forward_distances


class InverseDistanceMapper(DataMapper):
    """Consistent k-nearest-neighbor inverse-distance (Shepard) mapping.

    Each target value is a weighted average of its ``k`` nearest source values
    with weights ``1 / d**power``. The weights form a partition of unity (sum to
    1), so the map is **consistent**: it reproduces a constant field exactly and
    converges faster than nearest-neighbor for smooth fields. Coincident points
    (``d -> 0``) collapse to the exact source value.

    This is the lightweight upgrade over :class:`NearestNeighborMapper`. It is
    *not* a conservative Galerkin/RBF projection (that needs FE mass matrices and
    is future scope) — use it for smooth interface fields where consistency and
    smoothness matter more than discrete flux conservation.

    Parameters
    ----------
    k : int
        Number of nearest source neighbors per target (clamped to N_src).
    power : float
        Inverse-distance exponent (2.0 is the usual Shepard choice).
    """

    def __init__(self, k: int = 4, power: float = 2.0) -> None:
        if not _HAS_SCIPY:
            raise ImportError("scipy is required for InverseDistanceMapper.")
        self.k = k
        self.power = power
        self._forward: Optional[tuple] = None   # (idx, weights) target<-source
        self._inverse: Optional[tuple] = None   # (idx, weights) source<-target
        self.max_distance: Optional[float] = None

    def _precompute(self, tree: "KDTree", query: np.ndarray, n_src: int) -> tuple:
        k = min(self.k, n_src)
        dist, idx = tree.query(query, k=k)
        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        w = 1.0 / np.maximum(dist, 1e-30) ** self.power
        w = w / w.sum(axis=1, keepdims=True)
        return idx, w, float(np.max(dist))

    @staticmethod
    def _apply(values: np.ndarray, packed: tuple) -> np.ndarray:
        idx, w, _ = packed
        nb = np.asarray(values)[idx]            # (N, k) or (N, k, D)
        if nb.ndim == 2:
            return np.einsum("nk,nk->n", w, nb)
        return np.einsum("nk,nkd->nd", w, nb)

    def build(self, source_coords: np.ndarray,
              target_coords: np.ndarray) -> None:
        src = np.asarray(source_coords, dtype=float)
        tgt = np.asarray(target_coords, dtype=float)
        src_tree, tgt_tree = KDTree(src), KDTree(tgt)
        self._forward = self._precompute(src_tree, tgt, len(src))
        self._inverse = self._precompute(tgt_tree, src, len(tgt))
        self.max_distance = self._forward[2]
        logger.info(
            "Built InverseDistanceMapper (k=%d, power=%.1f): %d -> %d, "
            "max distance %.3e", self.k, self.power, len(src), len(tgt),
            self.max_distance,
        )

    def map(self, source_values: np.ndarray) -> np.ndarray:
        if self._forward is None:
            raise RuntimeError("Mapper not built. Call build() first.")
        return self._apply(source_values, self._forward)

    def inverse_map(self, target_values: np.ndarray) -> np.ndarray:
        if self._inverse is None:
            raise RuntimeError("Mapper not built. Call build() first.")
        return self._apply(target_values, self._inverse)
