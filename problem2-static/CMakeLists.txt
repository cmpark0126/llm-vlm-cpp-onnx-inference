cmake_minimum_required(VERSION 3.16)
project(problem2_static)

set(CMAKE_CXX_STANDARD 17)

# Find ONNX Runtime
# find_package(onnxruntime REQUIRED)

add_executable(problem2_static main.cpp)

# target_link_libraries(problem2_static onnxruntime)
