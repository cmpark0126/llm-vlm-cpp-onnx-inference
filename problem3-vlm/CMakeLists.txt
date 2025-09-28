cmake_minimum_required(VERSION 3.16)
project(problem3_vlm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS /usr/include/onnxruntime /usr/include)
find_library(ONNXRUNTIME_LIB onnxruntime
    PATHS /usr/lib)

# OpenCV (system package)
find_package(OpenCV REQUIRED)

# Add common directory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../common ${CMAKE_BINARY_DIR}/common)

add_executable(problem3_vlm main.cpp)

target_include_directories(problem3_vlm PRIVATE ${ONNXRUNTIME_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(problem3_vlm ${ONNXRUNTIME_LIB} ${OpenCV_LIBS} VlmTokenizer)
