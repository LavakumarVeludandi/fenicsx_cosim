"""Code-neutral mesh container shared by the FEniCSx and Kratos sides.

Deliberately NumPy-only (no meshio / dolfinx / Kratos imports) so it can be
constructed and serialized in either environment. Node and element ids are
1-based to match both gmsh and Kratos conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Region:
    """A named group of same-type cells (a subdomain or a boundary).

    ``dim`` is the topological dimension of the cells (e.g. 2 for triangles in a
    2D model, 1 for the boundary lines). Domain regions carry a non-zero
    ``property_id`` (material); boundary regions typically use ``0``.
    """

    name: str
    dim: int
    property_id: int
    cell_type: str               # neutral type: 'triangle','quad','tetra','hexahedron','line'
    connectivity: np.ndarray     # (n_cells, nodes_per_cell), 1-based node ids
    cell_ids: np.ndarray         # (n_cells,), 1-based global element/condition ids
    material: "object | None" = None  # optional MaterialModel; drives Kratos element family

    def __post_init__(self) -> None:
        self.connectivity = np.asarray(self.connectivity, dtype=np.int64)
        self.cell_ids = np.asarray(self.cell_ids, dtype=np.int64)
        if self.connectivity.ndim != 2:
            raise ValueError(
                f"Region '{self.name}': connectivity must be 2D, got shape "
                f"{self.connectivity.shape}"
            )
        if len(self.cell_ids) != len(self.connectivity):
            raise ValueError(
                f"Region '{self.name}': {len(self.cell_ids)} cell ids but "
                f"{len(self.connectivity)} cells"
            )


@dataclass
class NeutralMesh:
    """Points + named regions. Ids are 1-based."""

    points: np.ndarray                       # (N, 3)
    node_ids: np.ndarray                     # (N,), 1-based
    regions: list[Region] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64)
        if self.points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        self.node_ids = np.asarray(self.node_ids, dtype=np.int64)
        if len(self.node_ids) != len(self.points):
            raise ValueError("node_ids length must match points")

    def domain_regions(self) -> list[Region]:
        d = self.max_dim()
        return [r for r in self.regions if r.dim == d]

    def boundary_regions(self) -> list[Region]:
        d = self.max_dim()
        return [r for r in self.regions if r.dim == d - 1]

    def max_dim(self) -> int:
        return max(r.dim for r in self.regions)
