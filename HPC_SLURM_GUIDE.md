# FEniCSx Co-Simulation on HPC (SLURM)

Running `fenicsx-cosim` in a High-Performance Computing (HPC) environment managed by SLURM is straightforward because of the underlying **ZeroMQ TCP communication architecture**. ZeroMQ handles routing, queuing, and communication seamlessly whether the processes are on the same machine (using `localhost`) or distributed across multiple compute nodes in a cluster network.

However, moving from a local laptop to a multi-node SLURM environment requires exactly **two key adaptations**:

1. **Routing and Network Addresses**: You can no longer hardcode `tcp://localhost:5555`. The "Worker" or "Connect" processes must be explicitly provided the IP address or hostname of the compute node where the "Master" or "Bind" process is running.
2. **SLURM Job Orchestration**: You must write a SLURM batch script (`sbatch`) that launches the master process on one node, captures its address, and launches the worker processes on other nodes, feeding them the master's address.

---

## 1. Network Address Resolution

When running on an HPC cluster, the `bind` (master) process should listen on all network interfaces by binding to:
```python
# Master / Bind side
cosim = CouplingInterface(name="MacroSolver", role="master", endpoint="tcp://*:5556")
```

The `connect` (worker) process must connect using the exact hostname or IP address of the master node:
```python
# Worker / Connect side
import os

# Read the master IP from an environment variable passed by the SLURM script
master_ip = os.environ.get("MASTER_IP", "127.0.0.1")
endpoint = f"tcp://{master_ip}:5556"

cosim = CouplingInterface(name="MicroWorker", role="worker", endpoint=endpoint)
```

> **Important Setup Step**: You need to update the co-simulation scripts (like `fe2_micro_worker.py` and `amr_mechanical_solver.py`) to parse environment variables or command-line arguments for the endpoint URL, rather than hardcoding `localhost`.

---

## 2. Approach A: Single Monolithic SLURM Script (Recommended for FE²)

For a workload like **FE² Homogenization**, where a single macroscopic master dispatches work to a pool of microscopic workers, the easiest approach is to request multiple nodes in a single SLURM job and use `srun` to partition the tasks.

Here is an example `sbatch` script (`run_fe2.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=fe2_cosim
#SBATCH --nodes=3               # Request 3 total nodes
#SBATCH --ntasks=16             # Request 16 total tasks (cores)
#SBATCH --time=02:00:00
#SBATCH --output=fe2_%j.out

# Activate your conda/python environment
source activate fenicsx-env
export PYTHONPATH=src

# 1. Identify the Master Node
# SLURM_JOB_NODELIST contains a list of all nodes assigned to this job.
# We extract the very first node in the list to act as the Master.
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Get the IP address of the master node on the cluster network
export MASTER_IP=$(ssh $MASTER_NODE "hostname -i" | awk '{print $1}')

echo "Master Node identified as: $MASTER_NODE ($MASTER_IP)"

# 2. Launch the Master Process
# We use srun to constrain the master to exactly 1 task on the Master Node.
# The '&' sends it to the background so the script can continue.
srun --nodes=1 --ntasks=1 --nodelist=$MASTER_NODE --exclusive \
     python examples/fe2_macro_solver.py > macro.log 2>&1 &

# Sleep for a few seconds to ensure the Master has bound to the ZeroMQ sockets
sleep 5

# 3. Launch the Worker Processes
# We use the remaining 15 tasks for the workers. 
# They will inherit the $MASTER_IP environment variable, which the python
# script must use to build the tcp:// connection string.
srun --ntasks=15 --exclusive \
     python examples/fe2_micro_worker.py > micro_workers.log 2>&1 &

# 4. Wait for all background tasks to finish
wait
echo "FE² Co-Simulation Complete."
```

---

## 3. Approach B: Two Disconnected SLURM Jobs (Loose Coupling / AMR)

Sometimes you have two completely distinct solvers (like in the AMR Thermo-Mechanical example) and you want to submit them as two *separate* SLURM jobs, potentially with different resource requirements (e.g., Thermal needs 1 GPU node, Mechanical needs 4 CPU nodes).

To do this, the solvers must communicate their location through a shared filesystem file.

### Job 1: The Master (Bind) Script (`run_thermal.sh`)

```bash
#!/bin/bash
#SBATCH --job-name=amr_thermal
#SBATCH --nodes=1

source activate fenicsx-env

# Find our node IP
MY_IP=$(hostname -i | awk '{print $1}')

# Write the IP to a shared file so the connect job can find us
echo $MY_IP > /path/to/shared/storage/master_ip.txt

# Run the thermal solver (binds to tcp://*:5555)
python examples/amr_thermal_solver.py
```

### Job 2: The Worker (Connect) Script (`run_mechanical.sh`)

```bash
#!/bin/bash
#SBATCH --job-name=amr_mechanical
#SBATCH --nodes=1

source activate fenicsx-env

# Wait until the master job has written its IP address
SHARED_FILE="/path/to/shared/storage/master_ip.txt"
while [ ! -f $SHARED_FILE ]; do
    echo "Waiting for master IP..."
    sleep 2
done

export MASTER_IP=$(cat $SHARED_FILE)

# Run the mechanical solver (reads $MASTER_IP and connects)
python examples/amr_mechanical_solver.py
```

---

## Technical Considerations for SLURM

1. **Firewalls/Ports**: Ensure the ports used by ZeroMQ (e.g., `5555`, `5556`, `5557`) are open for inter-node communication on the cluster's internal network. If your HPC uses strict port locking, you may need to randomly allocate ports or request a specific port range from your administrator.
2. **MPI Integration**: The `CouplingInterface` uses ZeroMQ purely for inter-solver communication. This allows each individual FEniCSx solver to *also* use `mpi4py` internally across multiple cores without conflicting with the co-simulation networking.
3. **Storage**: Always use high-speed shared scratch environments (like Lustre or GPFS) if utilizing the "Shared File" (Approach B) method. 
