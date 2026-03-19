#!/bin/bash
#SBATCH --job-name=shakedown_cosim
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=shakedown_%j.log

echo "========================================="
echo "Starting Shakedown Co-Simulation"
echo "Allocated nodes: $SLURM_JOB_NODELIST"
echo "========================================="

MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export COSIM_MASTER_IP=$MASTER_NODE
echo "Master (FEniCSx) Node set to: $COSIM_MASTER_IP"

srun --nodes=1 --ntasks=1 --nodelist=$MASTER_NODE --exclusive \
     python3 shakedown_fenicsx_master_slurm.py &

sleep 3

WORKER_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | tail -n +2 | head -n 1)
echo "Starting Optimizer Worker on node: $WORKER_NODES"

srun --nodes=1 --ntasks=1 --nodelist=$WORKER_NODES --exclusive \
     python3 shakedown_optimizer_worker_slurm.py &

wait
echo "Shakedown Co-Simulation completed."
