#!/bin/bash
# Real partitioned thermo-mechanical coupling: two FEniCSx subdomains run an
# overlapping-Schwarz heat solve (exchanging interface temperature over
# fenicsx-cosim), then subdomain B runs a thermoelastic solve on the converged
# field. NOTE: this is a connector *demonstration* — two FEniCSx subdomains
# would normally be solved monolithically. See thermomech/thermal_a.py.
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"

echo "Running partitioned thermo-mechanical coupling..."
python -m thermomech.thermal_a &
PID_A=$!
python -m thermomech.thermal_b
wait $PID_A
echo "Done."
