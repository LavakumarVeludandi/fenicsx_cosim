#!/bin/bash
# Build context copy script for Docker
# Copies only the necessary files for package installation

set -e

# Copy pyproject.toml and requirements.txt
cp ../pyproject.toml ../requirements.txt .

# Copy src and tests directories
cp -r ../src .
cp -r ../tests .

# Optionally copy README and other metadata
cp ../README.md .
