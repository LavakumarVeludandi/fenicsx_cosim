"""Mapping between neutral cell types and registered Kratos element/condition names.

Covers the standard StructuralMechanicsApplication solid elements. The
``family`` selects the formulation (e.g. ``SmallDisplacement`` vs
``TotalLagrangian``); domain cells become Elements, one-lower-dimension cells
become Conditions.

Kept as explicit tables (not clever inference) so the mapping is auditable and
extensible — add a row to support a new element.
"""

from __future__ import annotations

# neutral cell type -> topological dimension
DIM_OF = {
    "line": 1, "line3": 1,
    "triangle": 2, "triangle6": 2, "quad": 2, "quad8": 2,
    "tetra": 3, "tetra10": 3, "hexahedron": 3, "hexahedron20": 3,
}

# neutral cell type -> nodes per cell
NODES_PER_CELL = {
    "line": 2, "line3": 3,
    "triangle": 3, "triangle6": 6,
    "quad": 4, "quad8": 8,
    "tetra": 4, "tetra10": 10,
    "hexahedron": 8, "hexahedron20": 20,
}

# (family, dim, neutral_cell_type) -> Kratos Element name
_ELEMENTS = {
    ("SmallDisplacement", 2, "triangle"): "SmallDisplacementElement2D3N",
    ("SmallDisplacement", 2, "quad"): "SmallDisplacementElement2D4N",
    ("SmallDisplacement", 2, "triangle6"): "SmallDisplacementElement2D6N",
    ("SmallDisplacement", 3, "tetra"): "SmallDisplacementElement3D4N",
    ("SmallDisplacement", 3, "hexahedron"): "SmallDisplacementElement3D8N",
    ("TotalLagrangian", 2, "triangle"): "TotalLagrangianElement2D3N",
    ("TotalLagrangian", 2, "quad"): "TotalLagrangianElement2D4N",
    ("TotalLagrangian", 3, "tetra"): "TotalLagrangianElement3D4N",
    ("TotalLagrangian", 3, "hexahedron"): "TotalLagrangianElement3D8N",
}

# (dim_of_condition, neutral_cell_type) -> Kratos Condition name
_CONDITIONS = {
    (1, "line"): "LineCondition2D2N",
    (2, "triangle"): "SurfaceCondition3D3N",
    (2, "quad"): "SurfaceCondition3D4N",
}

# Reverse: Kratos element/condition name -> neutral cell type
_KRATOS_TO_NEUTRAL = {}
for (_fam, _d, _ct), _name in _ELEMENTS.items():
    _KRATOS_TO_NEUTRAL[_name] = _ct
for (_d, _ct), _name in _CONDITIONS.items():
    _KRATOS_TO_NEUTRAL[_name] = _ct


def element_name(family: str, dim: int, cell_type: str) -> str:
    try:
        return _ELEMENTS[(family, dim, cell_type)]
    except KeyError:
        raise KeyError(
            f"No Kratos element for (family={family!r}, dim={dim}, "
            f"cell_type={cell_type!r}). Known: {sorted(_ELEMENTS)}"
        )


def condition_name(dim: int, cell_type: str) -> str:
    try:
        return _CONDITIONS[(dim, cell_type)]
    except KeyError:
        raise KeyError(
            f"No Kratos condition for (dim={dim}, cell_type={cell_type!r}). "
            f"Known: {sorted(_CONDITIONS)}"
        )


def neutral_cell_type(kratos_name: str) -> str:
    try:
        return _KRATOS_TO_NEUTRAL[kratos_name]
    except KeyError:
        raise KeyError(
            f"Unknown Kratos element/condition '{kratos_name}'. "
            f"Known: {sorted(_KRATOS_TO_NEUTRAL)}"
        )
