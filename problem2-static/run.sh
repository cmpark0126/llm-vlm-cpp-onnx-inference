#!/bin/bash
set -e

echo "Problem 2: Static Graph Export & Text Generation"

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
