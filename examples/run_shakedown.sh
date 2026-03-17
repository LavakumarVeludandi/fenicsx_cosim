#!/bin/bash
echo "Running Shakedown Optimizer Coupling..."

# Start optimizer worker first so the socket binds/connects properly (though ZeroMQ handles order)
python shakedown_optimizer_worker.py &
PID_WORKER=$!

python shakedown_fenicsx_master.py &
PID_MASTER=$!

wait $PID_WORKER
wait $PID_MASTER

echo "Done with Shakedown Coupling."
