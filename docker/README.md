# Build and run instructions for the Docker image

## Build the Docker image


**Important:** Run the following command from the ROOT of your project (not from the docker/ folder), so that Docker can access src, tests, and other files:

```
docker build --no-cache -f docker/Dockerfile -t wingsoflava/fenicsx-cosim:nightly .
```

## Run the Docker container

To start an interactive shell in the container:

```
docker run -it --rm wingsoflava/fenicsx-cosim:nightly
```

To mount your local project directory for development (optional):

```
docker run -it --rm -v $(pwd):/workspace wingsoflava/fenicsx-cosim:nightly
```

To push the docker container to the remote
```
docker push wingsoflava/fenicsx-cosim:nightly
```

## Notes
- The image is based on the official FEniCSx v10 (stable) Docker image.
- The package is installed using pip inside the container.
- You can modify the Dockerfile to add more dependencies as needed.
