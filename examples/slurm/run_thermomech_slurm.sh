#!/bin/bash
#SBATCH --job-name=thermomech_cosim
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=thermomech_%j.log

echo "========================================="
echo "Starting Thermo-Mechanical Co-Simulation"
echo "Allocated nodes: $SLURM_JOB_NODELIST"
echo "========================================="

MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export COSIM_MASTER_IP=$MASTER_NODE
echo "Master (Thermal) Node set to: $COSIM_MASTER_IP"

srun --nodes=1 --ntasks=1 --nodelist=$MASTER_NODE --exclusive \
     python3 thermal_solver_slurm.py &

sleep 3

WORKER_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | tail -n +2 | head -n 1)
echo "Starting Mechanical Solver on node: $WORKER_NODES"

srun --nodes=1 --ntasks=1 --nodelist=$WORKER_NODES --exclusive \
     python3 mechanical_solver_slurm.py &

wait
echo "Thermo-Mechanical Co-Simulation completed."
