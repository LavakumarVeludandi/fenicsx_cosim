"""Neutral material-model descriptor and Kratos correspondence (physics-free).

In Kratos, the element name encodes kinematics (``SmallDisplacement`` vs
``TotalLagrangian`` ...) and is selected together with a ``constitutive_law``
(materials.json). FEniCSx has no such catalog — physics is the UFL σ(ε). This
module is the neutral pivot: a :class:`MaterialModel` maps to the Kratos
(element family, constitutive_law, Variables) triple. The FEniCSx side binding
lives in ``sdlab_fem`` (which depends on ``fenicsx_cosim``), keeping this
package standalone and physics-free.

The registry is **curated, not inferred**: only enumerated correspondences map;
anything else raises ``KeyError`` rather than emitting a silently-wrong element
or law.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MaterialModel:
    """Code-neutral description of a constitutive model for one region."""

    kind: str                       # 'linear_elastic','j2_plasticity','neo_hookean',...
    kinematics: str                 # 'small_strain' | 'finite_strain'
    params: dict = field(default_factory=dict)
    plane: str | None = None        # 'plane_strain' | 'plane_stress' | None (3D)


@dataclass
class KratosMaterialSpec:
    """How a MaterialModel realizes in Kratos."""

    element_family: str             # 'SmallDisplacement' | 'TotalLagrangian'
    constitutive_law: str           # e.g. 'LinearElasticPlaneStrain2DLaw'
    variable_map: dict              # neutral param name -> Kratos VARIABLE name
    requires_app: str | None = None  # Kratos app that must be installed


_E_NU = {"E": "YOUNG_MODULUS", "nu": "POISSON_RATIO", "density": "DENSITY"}
_E_NU_J2 = {**_E_NU, "sigma_y": "YIELD_STRESS",
            "H": "ISOTROPIC_HARDENING_MODULUS"}

# (kind, kinematics, plane) -> KratosMaterialSpec.  Curated; extend explicitly.
_REGISTRY: dict[tuple, KratosMaterialSpec] = {
    ("linear_elastic", "small_strain", "plane_strain"):
        KratosMaterialSpec("SmallDisplacement", "LinearElasticPlaneStrain2DLaw", _E_NU),
    ("linear_elastic", "small_strain", "plane_stress"):
        KratosMaterialSpec("SmallDisplacement", "LinearElasticPlaneStress2DLaw", _E_NU),
    ("linear_elastic", "small_strain", None):
        KratosMaterialSpec("SmallDisplacement", "LinearElastic3DLaw", _E_NU),
    # J2 lives in ConstitutiveLawsApplication — entry exists, but flagged.
    ("j2_plasticity", "small_strain", "plane_strain"):
        KratosMaterialSpec("SmallDisplacement",
                           "SmallStrainIsotropicPlasticityFactoryVonMisesPlaneStrain2D",
                           _E_NU_J2, requires_app="ConstitutiveLawsApplication"),
    # Hyperelastic laws live in ConstitutiveLawsApplication (NOT in
    # StructuralMechanicsApplication). With only StructuralMechanics installed
    # these law names are unverified — flagged, not load-tested. The
    # TotalLagrangian *element* family is in StructuralMechanics; only the law is
    # absent. Verify the exact law name against a build with the app before use.
    ("neo_hookean", "finite_strain", None):
        KratosMaterialSpec("TotalLagrangian",
                           "HyperElasticIsotropicNeoHookean3DLaw", _E_NU,
                           requires_app="ConstitutiveLawsApplication"),
    ("st_venant_kirchhoff", "finite_strain", None):
        KratosMaterialSpec("TotalLagrangian",
                           "HyperElasticIsotropicKirchhoff3DLaw", _E_NU,
                           requires_app="ConstitutiveLawsApplication"),
}


def kratos_spec(model: MaterialModel) -> KratosMaterialSpec:
    key = (model.kind, model.kinematics, model.plane)
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"No Kratos correspondence for {key}. Supported: "
            f"{sorted(_REGISTRY, key=str)}. Add an entry to interop.material._REGISTRY "
            f"(curated by design — no silent fallback)."
        )


def element_family_for(model: MaterialModel) -> str:
    return kratos_spec(model).element_family


def write_materials_json(models: dict, path, model_part: str = "Structure",
                         property_ids: dict | None = None) -> None:
    """Emit a Kratos ``StructuralMaterials.json`` for the given region models.

    Parameters
    ----------
    models : dict[str, MaterialModel]
        region name -> material model.
    property_ids : dict[str, int], optional
        region name -> Kratos properties id (defaults to enumeration order).
    """
    property_ids = property_ids or {n: i + 1 for i, n in enumerate(models)}
    props = []
    for name, model in models.items():
        spec = kratos_spec(model)
        variables = {
            spec.variable_map[k]: v
            for k, v in model.params.items()
            if k in spec.variable_map
        }
        props.append({
            "model_part_name": f"{model_part}.{name}",
            "properties_id": property_ids[name],
            "Material": {
                "constitutive_law": {"name": spec.constitutive_law},
                "Variables": variables,
                "Tables": {},
            },
        })
    Path(path).write_text(json.dumps({"properties": props}, indent=4))
