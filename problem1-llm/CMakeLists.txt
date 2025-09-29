cmake_minimum_required(VERSION 3.16)
project(problem1_llm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set compiler optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

# ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS /usr/include/onnxruntime /usr/include)
find_library(ONNXRUNTIME_LIB onnxruntime
    PATHS /usr/lib)

# Add common directory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../common ${CMAKE_BINARY_DIR}/common)

add_executable(problem1_llm main.cpp)

target_include_directories(problem1_llm PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(problem1_llm ${ONNXRUNTIME_LIB} LlmTokenizer)
