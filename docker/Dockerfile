# This Dockerfile builds an image based on the official FEniCSx v10 image,
# copies the current package into the container, installs build dependencies,
# installs the package, and cleans up build dependencies.

FROM dolfinx/dolfinx:stable

# Set work directory
WORKDIR /workspace

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        python3-dev \
        git && \
    rm -rf /var/lib/apt/lists/*


# Copy only necessary files for package installation

# Copy only necessary files for package installation
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
COPY tests ./tests

# Install the Python package
RUN pip3 install --no-cache-dir .

# Default command
CMD ["python3"]
