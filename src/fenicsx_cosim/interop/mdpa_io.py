"""Read/write Kratos ``.mdpa`` from/to :class:`NeutralMesh` (pure NumPy).

Handles the parts meshio's mdpa writer drops: registered Kratos element/
condition type names, the per-element Properties id, and named **SubModelParts**
(which Kratos materials.json and BC assignment key on). Round-trips named
regions both ways.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

from fenicsx_cosim.interop import element_map as em
from fenicsx_cosim.interop.neutral_mesh import NeutralMesh, Region


# ----------------------------------------------------------------------
# Write
# ----------------------------------------------------------------------

def write_mdpa(mesh: NeutralMesh, path, family: str = "SmallDisplacement") -> None:
    dom = mesh.domain_regions()
    bnd = mesh.boundary_regions()

    # Assign globally-unique element / condition ids, remembering which belong
    # to each region (for the SubModelPart sections).
    elem_blocks: "OrderedDict[str, list]" = OrderedDict()
    region_elem_ids: dict[str, list[int]] = {}
    eid = 0
    for r in dom:
        # Element family is derived from the region's material model when
        # present (Kratos element name encodes kinematics); otherwise the
        # default `family` argument is used.
        fam = family
        if r.material is not None:
            from fenicsx_cosim.interop.material import element_family_for
            fam = element_family_for(r.material)
        kname = em.element_name(fam, r.dim, r.cell_type)
        region_elem_ids[r.name] = []
        for row in r.connectivity:
            eid += 1
            elem_blocks.setdefault(kname, []).append((eid, r.property_id, row))
            region_elem_ids[r.name].append(eid)

    cond_blocks: "OrderedDict[str, list]" = OrderedDict()
    region_cond_ids: dict[str, list[int]] = {}
    cid = 0
    for r in bnd:
        cname = em.condition_name(r.dim, r.cell_type)
        region_cond_ids[r.name] = []
        for row in r.connectivity:
            cid += 1
            cond_blocks.setdefault(cname, []).append((cid, r.property_id, row))
            region_cond_ids[r.name].append(cid)

    prop_ids = sorted({0} | {r.property_id for r in dom})

    out = []
    w = out.append
    w("Begin ModelPartData\nEnd ModelPartData\n")
    for pid in prop_ids:
        w(f"Begin Properties {pid}\nEnd Properties\n")

    w("Begin Nodes")
    for nid, p in zip(mesh.node_ids, mesh.points):
        w(f"{int(nid):d} {p[0]:.16e} {p[1]:.16e} {p[2]:.16e}")
    w("End Nodes\n")

    for kname, rows in elem_blocks.items():
        w(f"Begin Elements {kname}")
        for e_id, prop, nodes in rows:
            w(f"{e_id:d} {prop:d} " + " ".join(str(int(n)) for n in nodes))
        w("End Elements\n")

    for cname, rows in cond_blocks.items():
        w(f"Begin Conditions {cname}")
        for c_id, prop, nodes in rows:
            w(f"{c_id:d} {prop:d} " + " ".join(str(int(n)) for n in nodes))
        w("End Conditions\n")

    for r in dom:
        nodes = sorted(set(int(n) for n in r.connectivity.reshape(-1)))
        w(f"Begin SubModelPart {r.name}")
        w("  Begin SubModelPartNodes")
        for n in nodes:
            w(f"  {n}")
        w("  End SubModelPartNodes")
        w("  Begin SubModelPartElements")
        for e_id in region_elem_ids[r.name]:
            w(f"  {e_id}")
        w("  End SubModelPartElements")
        w("End SubModelPart\n")

    for r in bnd:
        nodes = sorted(set(int(n) for n in r.connectivity.reshape(-1)))
        w(f"Begin SubModelPart {r.name}")
        w("  Begin SubModelPartNodes")
        for n in nodes:
            w(f"  {n}")
        w("  End SubModelPartNodes")
        w("  Begin SubModelPartConditions")
        for c_id in region_cond_ids[r.name]:
            w(f"  {c_id}")
        w("  End SubModelPartConditions")
        w("End SubModelPart\n")

    Path(path).write_text("\n".join(out) + "\n")


# ----------------------------------------------------------------------
# Read
# ----------------------------------------------------------------------

def _strip(line: str) -> str:
    return line.split("//", 1)[0].strip()


def read_mdpa(path) -> NeutralMesh:
    lines = Path(path).read_text().splitlines()

    node_ids: list[int] = []
    coords: list[list[float]] = []
    elements: dict[int, tuple] = {}      # eid -> (cell_type, prop, nodes)
    conditions: dict[int, tuple] = {}    # cid -> (cell_type, prop, nodes)
    submodelparts: "OrderedDict[str, dict]" = OrderedDict()

    i, n = 0, len(lines)
    while i < n:
        s = _strip(lines[i])
        if not s:
            i += 1
            continue
        toks = s.split()

        if toks[0] == "Begin" and toks[1] == "Nodes":
            i += 1
            while _strip(lines[i]).split()[:1] != ["End"]:
                t = _strip(lines[i]).split()
                if t:
                    node_ids.append(int(t[0]))
                    coords.append([float(t[1]), float(t[2]), float(t[3])])
                i += 1

        elif toks[0] == "Begin" and toks[1] == "Elements":
            ctype = em.neutral_cell_type(toks[2])
            i += 1
            while _strip(lines[i]).split()[:1] != ["End"]:
                t = _strip(lines[i]).split()
                if t:
                    eid, prop = int(t[0]), int(t[1])
                    elements[eid] = (ctype, prop, [int(x) for x in t[2:]])
                i += 1

        elif toks[0] == "Begin" and toks[1] == "Conditions":
            ctype = em.neutral_cell_type(toks[2])
            i += 1
            while _strip(lines[i]).split()[:1] != ["End"]:
                t = _strip(lines[i]).split()
                if t:
                    cid, prop = int(t[0]), int(t[1])
                    conditions[cid] = (ctype, prop, [int(x) for x in t[2:]])
                i += 1

        elif toks[0] == "Begin" and toks[1] == "SubModelPart" and len(toks) >= 3:
            name = toks[2]
            smp = {"nodes": [], "elements": [], "conditions": []}
            i += 1
            while True:
                ss = _strip(lines[i])
                st = ss.split()
                if st[:2] == ["End", "SubModelPart"]:
                    break
                if st[:1] == ["Begin"] and st[1] == "SubModelPartNodes":
                    i += 1
                    while _strip(lines[i]).split()[:1] != ["End"]:
                        v = _strip(lines[i]).split()
                        if v:
                            smp["nodes"].append(int(v[0]))
                        i += 1
                elif st[:1] == ["Begin"] and st[1] == "SubModelPartElements":
                    i += 1
                    while _strip(lines[i]).split()[:1] != ["End"]:
                        v = _strip(lines[i]).split()
                        if v:
                            smp["elements"].append(int(v[0]))
                        i += 1
                elif st[:1] == ["Begin"] and st[1] == "SubModelPartConditions":
                    i += 1
                    while _strip(lines[i]).split()[:1] != ["End"]:
                        v = _strip(lines[i]).split()
                        if v:
                            smp["conditions"].append(int(v[0]))
                        i += 1
                i += 1
            submodelparts[name] = smp

        i += 1

    points = np.array(coords, dtype=np.float64) if coords else np.zeros((0, 3))
    node_ids_arr = np.array(node_ids, dtype=np.int64)

    regions: list[Region] = []
    for name, smp in submodelparts.items():
        if smp["elements"]:
            first = elements[smp["elements"][0]]
            ctype, prop = first[0], first[1]
            conn = np.array([elements[e][2] for e in smp["elements"]], dtype=np.int64)
            regions.append(Region(
                name=name, dim=em.DIM_OF[ctype], property_id=prop,
                cell_type=ctype, connectivity=conn,
                cell_ids=np.array(smp["elements"], dtype=np.int64),
            ))
        elif smp["conditions"]:
            first = conditions[smp["conditions"][0]]
            ctype, prop = first[0], first[1]
            conn = np.array([conditions[c][2] for c in smp["conditions"]], dtype=np.int64)
            regions.append(Region(
                name=name, dim=em.DIM_OF[ctype], property_id=prop,
                cell_type=ctype, connectivity=conn,
                cell_ids=np.array(smp["conditions"], dtype=np.int64),
            ))

    return NeutralMesh(points=points, node_ids=node_ids_arr, regions=regions)
