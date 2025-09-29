cmake_minimum_required(VERSION 3.16)

# Set C++ standard and optimization flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")

# nlohmann/json for JSON parsing
include(FetchContent)
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3)
FetchContent_MakeAvailable(json)

# Create LlmTokenizer library
add_library(LlmTokenizer LlmTokenizer.cpp)
target_link_libraries(LlmTokenizer nlohmann_json::nlohmann_json)
target_include_directories(LlmTokenizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Create VlmTokenizer library
add_library(VlmTokenizer VlmTokenizer.cpp)
target_link_libraries(VlmTokenizer nlohmann_json::nlohmann_json)
target_include_directories(VlmTokenizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})