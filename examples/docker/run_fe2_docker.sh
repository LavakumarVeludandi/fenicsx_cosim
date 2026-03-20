#!/bin/bash
# This script launches all required containers for the FE2 example using Docker Compose
# and ensures they can communicate via a shared network.

docker compose -f docker-compose-fe2.yml up --build
