#!/bin/bash
echo "Executing ALL fenicsx-cosim examples"
echo "====================================="

echo -e "\n---> Running Example 1: Basic Thermo-Mechanical Coupling"
./run_basic.sh

echo -e "\n---> Running Example 2: AMR Thermo-Mechanical Coupling"
./run_amr.sh

echo -e "\n---> Running Example 3: FE2 Multiscale Coupling"
./run_fe2.sh

echo -e "\n---> Running Example 4: Shakedown Optimizer Data Exchange"
./run_shakedown.sh

echo -e "\n====================================="
echo "All examples finished successfully!"
