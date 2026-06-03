cmake_minimum_required(VERSION 3.20)
project(inferium LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(inferium src/main.cpp src/gguf_parser.cpp src/tokenizer.cpp)
target_include_directories(inferium PRIVATE include)