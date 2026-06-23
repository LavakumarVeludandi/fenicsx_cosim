"""Code-neutral mesh interoperability for fenicsx-cosim.

A small, dependency-light middleware that round-trips meshes between FEniCSx
(DOLFINx / gmsh) and Kratos (``.mdpa``) for solid/structural mechanics.

Layering (so it imports in both the Kratos conda-base env and fenicsx-env):

* ``neutral_mesh`` / ``element_map`` / ``mdpa_io`` — pure NumPy, no heavy deps.
* ``dolfinx_bridge`` — imported lazily; only needed inside fenicsx-env.

See ``docs/interop_design.md``.
"""

from fenicsx_cosim.interop.neutral_mesh import NeutralMesh, Region

__all__ = ["NeutralMesh", "Region"]
