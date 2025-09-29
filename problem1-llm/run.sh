#!/bin/bash
set -e

echo "Building Problem 1: LLM Text Generation..."

echo "Running baselines..."
cd ../baselines
python3 run_problem1_baseline.py
cd ../problem1-llm

echo ""
echo "Running Problem 1 Solution..."

# Create build directory
mkdir -p build
cd build

# Configure and build
cmake ..
make

# Run
echo "Running Problem 1..."
./problem1_llm
