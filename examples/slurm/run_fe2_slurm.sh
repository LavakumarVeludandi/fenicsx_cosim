#!/bin/bash
#SBATCH --job-name=fe2_cosim
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=fe2_cosim_%j.log

echo "========================================="
echo "Starting FE2 Co-Simulation on SLURM"
echo "Allocated nodes: $SLURM_JOB_NODELIST"
echo "========================================="

# Determine the master node (first node in the allocation)
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export COSIM_MASTER_IP=$MASTER_NODE
echo "Master Node IP/Hostname set to: $COSIM_MASTER_IP"

# Run the master on the first node in the background
srun --nodes=1 --ntasks=1 --nodelist=$MASTER_NODE --exclusive \
     python3 fe2_macro_solver_slurm.py &

# Wait briefly to let the Master initialize its ZeroMQ bind socket
sleep 3

# Determine the worker nodes (all remaining nodes)
WORKER_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | tail -n +2 | paste -sd, -)
echo "Starting Worker(s) on node(s): $WORKER_NODES"

# Run the worker on the remaining node(s)
srun --nodes=1 --ntasks=1 --nodelist=$WORKER_NODES --exclusive \
     python3 fe2_micro_worker_slurm.py &

# Wait for background processes to finish
wait
echo "FE2 Co-Simulation completed."
