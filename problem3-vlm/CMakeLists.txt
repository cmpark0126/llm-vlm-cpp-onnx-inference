cmake_minimum_required(VERSION 3.16)
project(problem3_vlm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS /usr/include/onnxruntime /usr/local/include/onnxruntime ${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-linux-x64-1.19.0/include ${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-linux-aarch64-1.19.0/include)
find_library(ONNXRUNTIME_LIB onnxruntime
    PATHS /usr/lib /usr/local/lib ${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-linux-x64-1.19.0/lib ${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-linux-aarch64-1.19.0/lib)

# OpenCV (system package)
find_package(OpenCV REQUIRED)

# nlohmann/json for JSON parsing
include(FetchContent)
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3)
FetchContent_MakeAvailable(json)

add_executable(problem3_vlm main.cpp)

target_include_directories(problem3_vlm PRIVATE ${ONNXRUNTIME_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(problem3_vlm ${ONNXRUNTIME_LIB} ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
