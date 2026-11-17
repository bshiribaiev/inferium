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
    src/packed_matrix.cpp
    src/worker_team.cpp
    src/engine.cpp
    src/model_loader.cpp
    src/server_options.cpp
    src/http_api.cpp)
target_include_directories(inferium_core PUBLIC src include)
target_link_libraries(inferium_core PUBLIC Threads::Threads)

if(APPLE)
    enable_language(OBJCXX)
    target_sources(inferium_core PRIVATE src/metal_backend.mm)
    set_source_files_properties(src/metal_backend.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_link_libraries(inferium_core PUBLIC "-framework Metal" "-framework Foundation")
else()
    target_sources(inferium_core PRIVATE src/metal_unavailable.cpp)
endif()

add_executable(inferium src/main.cpp)
target_link_libraries(inferium PRIVATE inferium_core)

add_executable(inferium_bench src/bench.cpp)
target_link_libraries(inferium_bench PRIVATE inferium_core)

add_executable(inferium_server src/server_main.cpp)
target_link_libraries(inferium_server PRIVATE inferium_core)

find_package(Protobuf CONFIG)
find_package(gRPC CONFIG)

if(Protobuf_FOUND AND gRPC_FOUND)
    set(PROTO_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
    file(MAKE_DIRECTORY ${PROTO_DIR})
    set(PROTO_SRC ${PROTO_DIR}/inference.pb.cc)
    set(PROTO_GRPC_SRC ${PROTO_DIR}/inference.grpc.pb.cc)

    add_custom_command(
        OUTPUT ${PROTO_SRC} ${PROTO_GRPC_SRC}
        COMMAND protobuf::protoc
        ARGS --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
             --cpp_out=${PROTO_DIR}
             --grpc_out=${PROTO_DIR}
             --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
             ${CMAKE_CURRENT_SOURCE_DIR}/proto/inference.proto
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/proto/inference.proto
        COMMENT "Generating gRPC stubs from inference.proto")

    add_library(inferium_proto STATIC ${PROTO_SRC} ${PROTO_GRPC_SRC})
    target_include_directories(inferium_proto PUBLIC ${PROTO_DIR})
    target_link_libraries(inferium_proto PUBLIC gRPC::grpc++ protobuf::libprotobuf)

    add_executable(inferium_worker src/worker_main.cpp)
    target_link_libraries(inferium_worker PRIVATE inferium_core inferium_proto)

    add_executable(inferium_coordinator src/coordinator_main.cpp src/worker_client.cpp)
    target_link_libraries(inferium_coordinator PRIVATE inferium_core inferium_proto)
else()
    message(WARNING "gRPC/protobuf not found - skipping distributed targets")
endif()
