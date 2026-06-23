"""DOLFINx <-> NeutralMesh bridge (imported only inside fenicsx-env).

``dolfinx`` is imported at call time, not module load, so the rest of
``fenicsx_cosim.interop`` stays usable in the Kratos conda-base env. First-order
(P1-geometry) simplicial/quad meshes are supported — the common RVE case.
"""

from __future__ import annotations

import numpy as np

from fenicsx_cosim.interop.neutral_mesh import NeutralMesh, Region

# dolfinx CellType name -> neutral cell type
_CELL_NAME = {
    "point": "vertex", "interval": "line",
    "triangle": "triangle", "quadrilateral": "quad",
    "tetrahedron": "tetra", "hexahedron": "hexahedron",
}


def from_dolfinx(domain, cell_tags=None, facet_tags=None,
                 region_names: dict | None = None) -> NeutralMesh:
    """Build a :class:`NeutralMesh` from a DOLFINx mesh + MeshTags.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
    cell_tags : dolfinx.mesh.MeshTags, optional
        Cell (subdomain) markers -> one domain Region per tag value.
    facet_tags : dolfinx.mesh.MeshTags, optional
        Facet (boundary) markers -> one boundary Region per tag value.
    region_names : dict[int, str], optional
        Maps a tag value to a region name (defaults to ``Region_<value>`` /
        ``Boundary_<value>``).
    """
    import dolfinx

    region_names = region_names or {}
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    x = domain.geometry.x
    points = x if x.shape[1] == 3 else np.hstack(
        [x, np.zeros((x.shape[0], 3 - x.shape[1]))]
    )
    n_nodes = points.shape[0]
    node_ids = np.arange(1, n_nodes + 1, dtype=np.int64)

    cell_neutral = _CELL_NAME[domain.topology.cell_name()]
    facet_neutral = _facet_neutral(cell_neutral)

    regions: list[Region] = []

    # --- domain regions (cells) ---
    if cell_tags is not None:
        for val in np.unique(cell_tags.values):
            cells = cell_tags.indices[cell_tags.values == val].astype(np.int32)
            conn = dolfinx.mesh.entities_to_geometry(domain, tdim, cells) + 1
            regions.append(Region(
                name=region_names.get(int(val), f"Region_{int(val)}"),
                dim=tdim, property_id=int(val), cell_type=cell_neutral,
                connectivity=conn, cell_ids=np.arange(1, len(cells) + 1),
            ))
    else:
        ncells = domain.topology.index_map(tdim).size_local
        cells = np.arange(ncells, dtype=np.int32)
        conn = dolfinx.mesh.entities_to_geometry(domain, tdim, cells) + 1
        regions.append(Region(
            name="Domain", dim=tdim, property_id=1, cell_type=cell_neutral,
            connectivity=conn, cell_ids=np.arange(1, ncells + 1),
        ))

    # --- boundary regions (facets) ---
    if facet_tags is not None:
        for val in np.unique(facet_tags.values):
            facets = facet_tags.indices[facet_tags.values == val].astype(np.int32)
            conn = dolfinx.mesh.entities_to_geometry(domain, fdim, facets) + 1
            regions.append(Region(
                name=region_names.get(int(val), f"Boundary_{int(val)}"),
                dim=fdim, property_id=0, cell_type=facet_neutral,
                connectivity=conn, cell_ids=np.arange(1, len(facets) + 1),
            ))

    return NeutralMesh(points=points, node_ids=node_ids, regions=regions)


def _facet_neutral(cell_neutral: str) -> str:
    return {
        "triangle": "line", "quad": "line",
        "tetra": "triangle", "hexahedron": "quad",
    }[cell_neutral]


def to_dolfinx(mesh: NeutralMesh, gdim: int = 2):
    """Build a DOLFINx mesh + (cell_tags, facet_tags) from a :class:`NeutralMesh`.

    Returns ``(domain, cell_tags, facet_tags)``. Cell/facet tag *values* are the
    region ``property_id`` (cells) and a 1-based boundary index (facets); the
    name->value maps are returned attributes ``cell_region_names`` /
    ``facet_region_names`` on the returned tags for traceability.
    """
    import basix.ufl
    import dolfinx
    import ufl
    from mpi4py import MPI

    dom = mesh.domain_regions()
    cell_neutral = dom[0].cell_type
    cells = np.vstack([r.connectivity for r in dom]) - 1  # 0-based
    points = mesh.points[:, :gdim]

    el = basix.ufl.element("Lagrange", cell_neutral, 1, shape=(gdim,))
    ufl_domain = ufl.Mesh(el)
    domain = dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD, cells.astype(np.int64), ufl_domain, points
    )

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, 0)
    domain.topology.create_connectivity(fdim, tdim)

    # Map original (geometry-node) cell connectivity -> new local cell index, so
    # tag values survive create_mesh's possible reordering. Match by sorted node
    # set of each cell.
    o2n = domain.topology.original_cell_index
    inv = np.empty(len(o2n), dtype=np.int64)
    inv[o2n] = np.arange(len(o2n))

    cell_idx, cell_val = [], []
    offset = 0
    for r in dom:
        nloc = len(r.connectivity)
        for k in range(nloc):
            cell_idx.append(inv[offset + k])
            cell_val.append(r.property_id)
        offset += nloc
    cell_idx = np.array(cell_idx, dtype=np.int32)
    order = np.argsort(cell_idx)
    cell_tags = dolfinx.mesh.meshtags(
        domain, tdim, cell_idx[order], np.array(cell_val, dtype=np.int32)[order]
    )
    cell_tags.cell_region_names = {r.property_id: r.name for r in dom}

    return domain, cell_tags, None
