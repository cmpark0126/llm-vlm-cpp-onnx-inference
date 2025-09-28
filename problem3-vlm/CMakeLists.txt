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

# nlohmann/json for JSON parsing
include(FetchContent)
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3)
FetchContent_MakeAvailable(json)

add_executable(problem3_vlm main.cpp)

target_include_directories(problem3_vlm PRIVATE ${ONNXRUNTIME_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(problem3_vlm ${ONNXRUNTIME_LIB} ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
