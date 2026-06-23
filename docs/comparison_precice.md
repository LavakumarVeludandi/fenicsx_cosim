# fenicsx-cosim vs preCICE (honest, specific)

[preCICE](https://precice.org/) is the mature, C++-core, well-funded standard
for partitioned multiphysics. This document is deliberately specific so you can
choose correctly — including the many cases where **preCICE is the right call**.

## Where preCICE is stronger (use preCICE)

| Capability | preCICE | fenicsx-cosim |
|---|---|---|
| Implicit coupling acceleration | IQN-ILS **and** IQN-IMVJ (multi-vector), waveform iteration | Aitken + IQN-ILS (no IMVJ, no waveform) |
| Mapping | nearest-neighbor, nearest-projection, **RBF** (consistent & conservative) | nearest-neighbor, inverse-distance (consistent); **no RBF, no conservative FE projection yet** |
| Solver ecosystem | official adapters: OpenFOAM, SU2, CalculiX, code_aster, Abaqus, FEniCS, … | FEniCSx-native; Kratos/Abaqus via thin adapters |
| Multi-code, ≥3 participants | mature, configurable | `PAIR` is 1-to-1; ≥3 via broker is roadmap (P3) |
| Maturity / validation | years of published validation, large community | young; analytic benchmarks only |
| Runtime | optimized C++ with MPI ports | pure Python over ZeroMQ |

If you need production FSI/CHT across multiple codes, RBF conservative mapping,
or waveform/IMVJ acceleration: **use preCICE.** fenicsx-cosim does not try to win
that fight.

## Where fenicsx-cosim is stronger (use fenicsx-cosim)

| Capability | fenicsx-cosim | preCICE |
|---|---|---|
| Setup for a FEniCSx user | `pip install`, pure Python, operate on `dolfinx` objects directly | C++ runtime + per-solver adapter + XML config |
| FE² / RVE worker-pool dispatch | first-class (`PUSH/PULL`, `REQ/REP` broker, quadrature extraction) | not a target — couples a fixed participant set over a shared interface mesh |
| FEniCSx ↔ Kratos mesh + material translation | writes/reads loadable `.mdpa` + materials.json (element family, Properties, SubModelParts) | out of scope (runtime field exchange, not mesh/material formats) |
| Hackability | one small Python package; subclass a mapper or accelerator in minutes | larger C++ codebase |

## Decision rule

- **FEniCSx-native homogenization (FE²), RVE pools** → fenicsx-cosim.
- **Moving a mesh + constitutive model between FEniCSx and Kratos** → fenicsx-cosim.
- **A quick FEniCSx ↔ one-other-code coupling with minimal setup** → fenicsx-cosim.
- **Multi-code production FSI/CHT, conservative RBF, IMVJ/waveform** → preCICE.

Honest summary: for a team that lives in FEniCSx, fenicsx-cosim is the
lower-friction default and the only option for FE² farming and Kratos
mesh/material bridging. For everything preCICE was built for, preCICE wins.
