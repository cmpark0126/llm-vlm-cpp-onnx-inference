#!/bin/bash
set -e

echo "Building Problem 3: VLM Text Generation..."

# Create build directory
mkdir -p build
cd build

# Configure and build
cmake ..
make

# Run
echo "Running Problem 3..."
./problem3_vlm
