"""Standalone Kratos-load validation (NO pytest import).

Run as a child process by ``test_kratos_load.py``. Importing pytest into the same
interpreter as Kratos segfaults on teardown, so this script stays pytest-free and
prints the sentinel ``KRATOS_LOAD_OK`` on success.

Proves the middleware output is real Kratos input: ``ModelPartIO`` reads the
written .mdpa, ``ReadMaterialsUtility`` reads the written materials.json, and the
SubModelPart, the solvable element, and the constitutive law all materialize —
the things meshio's mdpa output failed at despite passing substring checks.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication  # noqa: F401

OK = "KRATOS_LOAD_OK"

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from fenicsx_cosim.interop.neutral_mesh import NeutralMesh, Region  # noqa: E402
from fenicsx_cosim.interop import mdpa_io  # noqa: E402
from fenicsx_cosim.interop.material import (  # noqa: E402
    MaterialModel, write_materials_json,
)


# 2D unit square (nodes 1..4 CCW), two triangles.
_SQUARE = (
    np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]),
    np.array([1, 2, 3, 4]),
    "triangle",
    np.array([[1, 2, 3], [1, 3, 4]]),
)
# 3D unit tetra (nodes 1..4), one cell.
_TETRA = (
    np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
    np.array([1, 2, 3, 4]),
    "tetra",
    np.array([[1, 2, 3, 4]]),
)

# Cases load-verifiable in *this* Kratos build (StructuralMechanicsApplication
# only — no ConstitutiveLawsApplication, so no J2 / hyperelastic laws). Each
# exercises a distinct registry row: a different element family AND constitutive
# law name. ReadModelPart raises on an unregistered element and
# ReadMaterialsUtility raises on an unregistered law, so a clean load is the proof
# the names in interop.material._REGISTRY / element_map are real — the exact thing
# meshio's geometry-only "Triangle2D3" output failed at.
_CASES = [
    ("linear_elastic", "small_strain", "plane_strain", _SQUARE),
    ("linear_elastic", "small_strain", "plane_stress", _SQUARE),
    ("linear_elastic", "small_strain", None, _TETRA),
]


def _check_one(tmp: Path, idx: int, kind, kin, plane, geom) -> None:
    pts, nids, ctype, conn = geom
    mat = MaterialModel(kind=kind, kinematics=kin, plane=plane,
                        params={"E": 200e3, "nu": 0.3})
    mesh = NeutralMesh(
        points=pts, node_ids=nids,
        regions=[Region(name="Solid", dim=conn.shape[1] - 1 if ctype == "line"
                        else (2 if ctype in ("triangle", "quad") else 3),
                        property_id=1, cell_type=ctype, connectivity=conn,
                        cell_ids=np.arange(1, len(conn) + 1), material=mat)],
    )
    mdpa = tmp / f"model_{idx}.mdpa"
    mat_json = tmp / f"mat_{idx}.json"
    mdpa_io.write_mdpa(mesh, mdpa)  # element family derived from material
    write_materials_json({"Solid": mat}, mat_json, model_part="Structure",
                         property_ids={"Solid": 1})

    model = KM.Model()  # fresh model per case
    mp = model.CreateModelPart("Structure")
    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    # ModelPartIO takes the path WITHOUT the .mdpa suffix.
    KM.ModelPartIO(str(mdpa.with_suffix(""))).ReadModelPart(mp)

    assert mp.NumberOfNodes() == len(nids)
    assert mp.NumberOfElements() == len(conn)
    assert mp.HasSubModelPart("Solid")
    solid = mp.GetSubModelPart("Solid")
    assert solid.NumberOfElements() == len(conn)

    # materials.json carries the law name; a clean read proves the name is real.
    # (element.Info() / law.Info() segfault this Kratos pybind build — the
    # successful ModelPartIO + ReadMaterialsUtility are the assertions.)
    KM.ReadMaterialsUtility(
        KM.Parameters('{"Parameters":{"materials_filename":"%s"}}'
                      % mat_json.as_posix()),
        model,
    )
    props = next(iter(solid.Elements)).Properties
    assert props.Has(KM.CONSTITUTIVE_LAW)
    assert props.GetValue(KM.YOUNG_MODULUS) == 200e3


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    for idx, (kind, kin, plane, geom) in enumerate(_CASES):
        _check_one(tmp, idx, kind, kin, plane, geom)
    print(OK, flush=True)


if __name__ == "__main__":
    main()
