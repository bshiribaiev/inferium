cmake_minimum_required(VERSION 3.20)
project(inferium LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

add_library(inferium_core STATIC
    src/arena.cpp
    src/mapped_file.cpp
    src/model.cpp
    src/inference.cpp
    src/gguf_parser.cpp
    src/tokenizer.cpp
    src/forward_pass.cpp
    src/thread_pool.cpp
    src/engine.cpp)
target_include_directories(inferium_core PUBLIC src include)
target_link_libraries(inferium_core PUBLIC Threads::Threads)

add_executable(inferium src/main.cpp)
target_link_libraries(inferium PRIVATE inferium_core)

add_executable(inferium_bench src/bench.cpp)
target_link_libraries(inferium_bench PRIVATE inferium_core)
