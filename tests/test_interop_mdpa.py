"""Tests for the Kratos<->neutral mesh middleware (pure Python, no Kratos/dolfinx).

The middleware carries a code-neutral mesh (``NeutralMesh``) with named regions
(subdomains + boundaries) and round-trips it through Kratos ``.mdpa``. These
tests need neither Kratos nor DOLFINx — they verify the format layer that both
the Kratos worker (conda base) and the FEniCSx side (fenicsx-env) rely on.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fenicsx_cosim.interop.neutral_mesh import NeutralMesh, Region  # noqa: E402
from fenicsx_cosim.interop import mdpa_io  # noqa: E402


def _small_mesh():
    # Unit square split into 2 triangles; "inclusion" is the 2nd triangle.
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    node_ids = np.array([1, 2, 3, 4])
    regions = [
        Region(name="Matrix", dim=2, property_id=1, cell_type="triangle",
               connectivity=np.array([[1, 2, 3]]), cell_ids=np.array([1])),
        Region(name="Inclusions", dim=2, property_id=2, cell_type="triangle",
               connectivity=np.array([[1, 3, 4]]), cell_ids=np.array([2])),
        Region(name="Bottom", dim=1, property_id=0, cell_type="line",
               connectivity=np.array([[1, 2]]), cell_ids=np.array([1])),
    ]
    return NeutralMesh(points=points, node_ids=node_ids, regions=regions)


def test_write_mdpa_uses_real_kratos_element_names(tmp_path):
    out = tmp_path / "m.mdpa"
    mdpa_io.write_mdpa(_small_mesh(), out, family="SmallDisplacement")
    text = out.read_text()
    assert "Begin Elements SmallDisplacementElement2D3N" in text
    assert "Begin Conditions LineCondition2D2N" in text
    # Material grouping survives as Properties + named SubModelParts.
    assert "Begin Properties 1" in text and "Begin Properties 2" in text
    assert "Begin SubModelPart Matrix" in text
    assert "Begin SubModelPart Inclusions" in text


def test_mdpa_roundtrip_preserves_geometry_and_regions(tmp_path):
    out = tmp_path / "m.mdpa"
    original = _small_mesh()
    mdpa_io.write_mdpa(original, out, family="SmallDisplacement")
    back = mdpa_io.read_mdpa(out)

    np.testing.assert_allclose(back.points, original.points)
    assert {r.name for r in back.regions} == {"Matrix", "Inclusions", "Bottom"}

    by_name = {r.name: r for r in back.regions}
    assert by_name["Matrix"].property_id == 1
    assert by_name["Inclusions"].property_id == 2
    np.testing.assert_array_equal(
        by_name["Inclusions"].connectivity, np.array([[1, 3, 4]])
    )
    assert by_name["Bottom"].dim == 1
    np.testing.assert_array_equal(
        by_name["Bottom"].connectivity, np.array([[1, 2]])
    )
