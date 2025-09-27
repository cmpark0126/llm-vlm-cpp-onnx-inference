cmake_minimum_required(VERSION 3.16)
project(problem2_static)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS /usr/include/onnxruntime /usr/local/include/onnxruntime ${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-linux-x64-1.19.0/include ${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-linux-aarch64-1.19.0/include)
find_library(ONNXRUNTIME_LIB onnxruntime
    PATHS /usr/lib /usr/local/lib ${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-linux-x64-1.19.0/lib ${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-linux-aarch64-1.19.0/lib)

# nlohmann/json for JSON parsing
include(FetchContent)
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3)
FetchContent_MakeAvailable(json)

add_executable(problem2_static main.cpp)

target_include_directories(problem2_static PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(problem2_static ${ONNXRUNTIME_LIB} nlohmann_json::nlohmann_json)
