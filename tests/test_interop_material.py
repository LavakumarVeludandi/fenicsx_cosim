"""Material-model correspondence tests (pure Python; no Kratos/dolfinx/sdlab).

The middleware maps a neutral MaterialModel to the Kratos (element family +
constitutive_law + Variables) triple — because in Kratos the element name
encodes kinematics and is chosen *together* with the constitutive law. The
element family used when writing the .mdpa is therefore DERIVED from the
material, never hardcoded.
"""

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fenicsx_cosim.interop.material import (  # noqa: E402
    MaterialModel, kratos_spec, element_family_for, write_materials_json,
)


def test_linear_elastic_plane_strain_maps_to_small_displacement():
    m = MaterialModel(kind="linear_elastic", kinematics="small_strain",
                      plane="plane_strain", params={"E": 50e3, "nu": 0.2})
    spec = kratos_spec(m)
    assert spec.element_family == "SmallDisplacement"
    assert spec.constitutive_law == "LinearElasticPlaneStrain2DLaw"
    assert element_family_for(m) == "SmallDisplacement"


def test_finite_strain_maps_to_total_lagrangian():
    m = MaterialModel(kind="neo_hookean", kinematics="finite_strain",
                      plane=None, params={"E": 1e3, "nu": 0.45})
    assert element_family_for(m) == "TotalLagrangian"


def test_unmapped_model_raises_not_silently_wrong():
    m = MaterialModel(kind="cam_clay", kinematics="small_strain",
                      plane="plane_strain", params={})
    with pytest.raises(KeyError):
        kratos_spec(m)


def test_region_material_drives_mdpa_element_family(tmp_path):
    import numpy as np
    from fenicsx_cosim.interop.neutral_mesh import NeutralMesh, Region
    from fenicsx_cosim.interop import mdpa_io

    finite = MaterialModel(kind="st_venant_kirchhoff", kinematics="finite_strain",
                           plane=None, params={"E": 1e3, "nu": 0.3})
    mesh = NeutralMesh(
        points=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
        node_ids=np.array([1, 2, 3]),
        regions=[Region(name="Solid", dim=2, property_id=1, cell_type="triangle",
                        connectivity=np.array([[1, 2, 3]]),
                        cell_ids=np.array([1]), material=finite)],
    )
    out = tmp_path / "m.mdpa"
    mdpa_io.write_mdpa(mesh, out)  # no family arg -> derived from material
    assert "Begin Elements TotalLagrangianElement2D3N" in out.read_text()


def test_write_materials_json_uses_kratos_variable_names(tmp_path):
    import json
    models = {
        "Matrix": MaterialModel("linear_elastic", "small_strain",
                                params={"E": 50e3, "nu": 0.2, "density": 7850.0},
                                plane="plane_strain"),
        "Inclusions": MaterialModel("linear_elastic", "small_strain",
                                    params={"E": 210e3, "nu": 0.3},
                                    plane="plane_strain"),
    }
    out = tmp_path / "mat.json"
    write_materials_json(models, out, model_part="Structure",
                         property_ids={"Matrix": 1, "Inclusions": 2})
    data = json.loads(out.read_text())
    props = {p["properties_id"]: p for p in data["properties"]}
    assert props[1]["model_part_name"] == "Structure.Matrix"
    assert props[1]["Material"]["constitutive_law"]["name"] == "LinearElasticPlaneStrain2DLaw"
    v = props[1]["Material"]["Variables"]
    assert v["YOUNG_MODULUS"] == 50e3 and v["POISSON_RATIO"] == 0.2
