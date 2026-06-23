"""DOLFINx<->Kratos bridge tests (fenicsx-env; no Kratos needed).

Exercises both directions of the middleware:
  dolfinx mesh + tags --from_dolfinx--> NeutralMesh --write_mdpa--> .mdpa
  .mdpa --read_mdpa--> NeutralMesh --to_dolfinx--> dolfinx mesh + tags
and checks that named subdomains/boundaries and the per-subdomain cell counts
survive the full round-trip.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

pytest.importorskip("dolfinx")

import dolfinx  # noqa: E402
from mpi4py import MPI  # noqa: E402

from fenicsx_cosim.interop import mdpa_io  # noqa: E402
from fenicsx_cosim.interop import dolfinx_bridge as db  # noqa: E402


def _tagged_square(n=4):
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.triangle
    )
    tdim = domain.topology.dim
    ncells = domain.topology.index_map(tdim).size_local
    mids = dolfinx.mesh.compute_midpoints(domain, tdim, np.arange(ncells, dtype=np.int32))
    vals = np.where(mids[:, 0] < 0.5, 1, 2).astype(np.int32)
    cell_tags = dolfinx.mesh.meshtags(
        domain, tdim, np.arange(ncells, dtype=np.int32), vals
    )
    facets = dolfinx.mesh.locate_entities_boundary(
        domain, tdim - 1, lambda x: np.isclose(x[0], 0.0)
    )
    facet_tags = dolfinx.mesh.meshtags(
        domain, tdim - 1, np.sort(facets),
        np.full(len(facets), 10, dtype=np.int32),
    )
    return domain, cell_tags, facet_tags, ncells


@pytest.mark.slow
@pytest.mark.integration
def test_from_dolfinx_through_mdpa_preserves_named_regions(tmp_path):
    domain, cell_tags, facet_tags, ncells = _tagged_square()
    nm = db.from_dolfinx(domain, cell_tags, facet_tags,
                         {1: "Matrix", 2: "Inclusions", 10: "Left"})

    out = tmp_path / "m.mdpa"
    mdpa_io.write_mdpa(nm, out)
    back = mdpa_io.read_mdpa(out)

    names = {r.name for r in back.regions}
    assert {"Matrix", "Inclusions", "Left"} <= names
    # All domain cells survive the round-trip.
    assert sum(len(r.connectivity) for r in back.domain_regions()) == ncells
    # Kratos element type is correct.
    assert "SmallDisplacementElement2D3N" in out.read_text()


@pytest.mark.slow
@pytest.mark.integration
def test_to_dolfinx_preserves_cell_tag_distribution():
    domain, cell_tags, _, _ = _tagged_square()
    nm = db.from_dolfinx(domain, cell_tags)
    d2, ct2, _ = db.to_dolfinx(nm, gdim=2)

    orig = {int(v): int((cell_tags.values == v).sum())
            for v in np.unique(cell_tags.values)}
    new = {int(v): int((ct2.values == v).sum())
           for v in np.unique(ct2.values)}
    assert orig == new
