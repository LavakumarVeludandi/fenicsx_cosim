# fenicsx-cosim vs preCICE FEniCSx adapter

This document provides a practical comparison to help users choose the right tool.

## Installation complexity

- **fenicsx-cosim**: pure-Python package workflow with FEniCSx dependencies.
- **preCICE adapter**: typically involves preCICE runtime and adapter integration setup.

## Coupling schemes

- **fenicsx-cosim**: currently centered on partitioned explicit workflows, with strong-coupling extensions on the roadmap.
- **preCICE adapter**: mature support for several coupling configurations and schemes.

## Mapping methods

- **fenicsx-cosim**: nearest-neighbor mapping is available; higher-order methods are planned.
- **preCICE adapter**: broader established mapping options in existing preCICE workflows.

## Performance profile

- **fenicsx-cosim**: favors lightweight Python-level integration and rapid experimentation.
- **preCICE adapter**: optimized coupling infrastructure designed for larger multi-code deployments.

## When to choose each

Choose **fenicsx-cosim** when you want:
- fast prototyping in pure Python,
- direct integration with FEniCSx-native data structures,
- compact custom workflows (including FE²-style orchestration).

Choose **preCICE adapter** when you need:
- production-grade coupling features already available in preCICE,
- broad interoperability with multiple external solvers,
- mature coupling-ecosystem tooling out of the box.
