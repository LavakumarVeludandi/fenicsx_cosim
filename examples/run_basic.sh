#!/bin/bash
echo "Running Basic Thermo-Mechanical Coupling..."

# Start both solvers in the background
python thermal_solver.py &
PID_THERMAL=$!

python mechanical_solver.py &
PID_MECH=$!

# Wait for both to finish
wait $PID_THERMAL
wait $PID_MECH

echo "Done with Basic Coupling."
