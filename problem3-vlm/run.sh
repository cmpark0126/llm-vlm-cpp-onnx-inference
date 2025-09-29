#!/bin/bash
set -e

echo "Building Problem 3: VLM Text Generation..."

echo "Running baselines..."
cd ../baselines
python3 run_problem3_baseline.py
cd ../problem3-vlm

echo ""
echo "Running Problem 3 Solution..."

# Create build directory
mkdir -p build
cd build

# Configure and build
cmake ..
make

# Run
echo "Running Problem 3..."
./problem3_vlm
