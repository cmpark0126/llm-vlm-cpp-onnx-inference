#!/bin/bash
set -e

echo "Building Problem 1: LLM Text Generation..."

# Create build directory
mkdir -p build
cd build

# Configure and build
cmake ..
make

# Run
echo "Running Problem 1..."
./problem1_llm
