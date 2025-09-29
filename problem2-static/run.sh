#!/bin/bash
set -e

echo "Problem 2: Static Graph Export & Text Generation"

echo "Running baselines..."
cd ../baselines
python3 run_problem2_baseline.py
cd ../problem2-static

echo "Running Problem 2 Solution..."

# Export ONNX Static Graph
python3 export_onnx_prefill.py
python3 export_onnx_decode.py

# Build C++ application
echo "Building C++ application..."
mkdir -p build
cd build

# Configure and build
cmake ..
make

# Run
echo "Running Problem 2..."
./problem2_static
