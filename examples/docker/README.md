# Docker-based FE2 Example

This folder contains scripts and configuration to run the FE2 example using Docker containers that communicate over a shared Docker network.

## How it works
- Two services are defined: `fe2_macro` and `fe2_micro`, each running in its own container.
- Both containers share the same workspace via a volume mount and communicate over a custom Docker network (`fe2net`).
- The macro and micro solvers are started using the same entrypoints as in the slurm setup, but now run as separate containers.

## Usage

From the project root, build the Docker image if you haven't already:

```
docker build -f docker/Dockerfile -t fenicsx-cosim .
```

Then, from this folder, launch the FE2 example:

```
cd examples/docker
bash run_fe2_docker.sh
```

This will start both containers and run the FE2 macro and micro solvers, allowing them to communicate as needed.

## Customization
- You can adapt the `docker-compose-fe2.yml` file to add more services or change resource limits.
- For other workflows, create similar compose files and scripts.
