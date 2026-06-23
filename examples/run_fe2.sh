#!/bin/bash
# Real FE² multiscale coupling: 2D macro elasticity (FEniCSx) whose constitutive
# response at every macro quadrature point is a genuine two-phase J2 RVE solve
# on a worker, exchanged over fenicsx-cosim scatter-gather.
#
# IMPORTANT: milestone 1 uses EXACTLY ONE worker. The RVE state is path-dependent
# (J2 plasticity) and the scatter-gather socket round-robins with no worker
# affinity, so a single stateful worker must own all RVE states (keyed by macro
# quadrature-point index). Launching multiple workers would split each point's
# plastic history across processes and corrupt it. See docs/fe2_design.md for the
# milestone-2 plan (binary state frame) that lifts this to N parallel workers.
set -e
cd "$(dirname "$0")"

echo "Running FE2 Multiscale Coupling (1 macro + 1 stateful worker)..."

python -m fe2.micro_worker localhost &
PID_WORKER=$!

python -m fe2.macro_solver
wait $PID_WORKER

echo "Done with FE2 Coupling."
