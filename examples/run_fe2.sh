#!/bin/bash
echo "Running FE2 Multiscale Coupling..."

# Start the macro solver (Master) in the background
python fe2_macro_solver.py &
PID_MACRO=$!

# Start 2 micro workers in the background
python fe2_micro_worker.py &
PID_WORKER1=$!

python fe2_micro_worker.py &
PID_WORKER2=$!

# Wait for all processes to finish
wait $PID_MACRO
wait $PID_WORKER1
wait $PID_WORKER2

echo "Done with FE2 Coupling."
