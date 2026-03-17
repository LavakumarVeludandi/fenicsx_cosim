#!/bin/bash
echo "Running AMR Thermo-Mechanical Coupling..."

# Start both solvers in the background
python amr_thermal_solver.py &
PID_THERMAL=$!

python amr_mechanical_solver.py &
PID_MECH=$!

# Wait for both to finish
wait $PID_THERMAL
wait $PID_MECH

echo "Done with AMR Coupling."
