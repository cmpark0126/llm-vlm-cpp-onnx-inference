cmake_minimum_required(VERSION 3.16)
project(problem1_llm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find ONNX Runtime
# find_package(onnxruntime REQUIRED)

add_executable(problem1_llm main.cpp)

# target_link_libraries(problem1_llm onnxruntime)
