# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- **Strong (implicit) coupling** (`convergence.py`): `Aitken` dynamic relaxation
  and `IQNILS` quasi-Newton acceleration, plus `fixed_point_iterate`.
- **Consistent mapping**: `InverseDistanceMapper` (k-NN Shepard) — reproduces
  constants exactly; lower error than nearest-neighbor for smooth fields.
- **N > 2 coupling** (`broker_communicator.py`): `CouplingBroker` (ROUTER hub)
  + `BrokerClient` (DEALER), and `CouplingInterface(topology="broker")` with
  `register_broker` / `send_to` / `receive_from` / `barrier`.
- **FEniCSx ↔ Kratos interop** (`interop/`): neutral `NeutralMesh` / `MaterialModel`
  pivot, real `.mdpa` + `StructuralMaterials.json` read/write, dolfinx bridge.
- **Validated benchmarks** (`benchmarks/`) with analytic gates in CI: mapping
  accuracy (NN + IDW), implicit-coupling convergence, FE² homogenization vs Hooke.
- Real partitioned thermo-mechanical example; real J2/KUBC FE² RVE + worker pool.

### Changed

- Positioning rewritten (README, `docs/comparison_precice.md`): moat-first —
  FEniCSx-native ergonomics, FE² farming, Kratos translation; honest preCICE split.

### Fixed

- `test_kratos_load` now probes the actual check interpreter (`KRATOS_PYTHON`
  override) instead of mis-skipping when the Kratos package leaks without its core.
- Removed committed build artifacts (`docs/_build`, egg-info, `*.h5`/`*.xdmf`)
  and the false PyPI badge (package not yet published).

## [0.1.0] - 2026-04-13

### Added

- Initial `CouplingInterface` API for partitioned coupling workflows
- ZeroMQ-based communication layer for solver-to-solver exchange
- Boundary extraction and nearest-neighbor mapping utilities
- Dynamic mapping support for changing interfaces
- FE² helper utilities for quadrature-based data exchange
- Example scripts for basic coupling, AMR, FE², and adapter integrations
- Initial automated test suite covering core modules
