cmake_minimum_required(VERSION 3.5)

if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()

project(ObjectDetectionService CXX)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)

include(FetchContent)

set(FETCHCONTENT_QUIET OFF)
set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_ENABLE_INSTALL ON)
set(BUILD_SHARED_LIBS ON)
set(Protobuf_USE_STATIC_LIBS OFF)
set(gRPC_BUILD_SHARED_LIBS ON)

# Fetch Protobuf
FetchContent_Declare(
    protobuf
    GIT_REPOSITORY https://github.com/google/protobuf.git
    GIT_TAG        v3.12.4
    GIT_PROGRESS   TRUE
    GIT_SHALLOW    TRUE
    USES_TERMINAL_DOWNLOAD TRUE
    GIT_SUBMODULES_RECURSE FALSE
    GIT_SUBMODULES ""
)

set(protobuf_BUILD_TESTS OFF)
set(protobuf_BUILD_CONFORMANCE OFF)
set(protobuf_BUILD_EXAMPLES OFF)
set(protobuf_BUILD_PROTOC_BINARIES ON)
set(protobuf_DISABLE_RTTI ON)
set(protobuf_MSVC_STATIC_RUNTIME ON)
set(protobuf_WITH_ZLIB ON CACHE BOOL "" FORCE)

FetchContent_GetProperties(protobuf)
if(NOT protobuf_POPULATED)
    FetchContent_Populate(protobuf)
    set(PROTOBUF_ROOT_DIR "${protobuf_SOURCE_DIR}")
endif()

# Fetch gRPC
FetchContent_Declare(
    grpc
    GIT_REPOSITORY https://github.com/grpc/grpc.git
    GIT_TAG        v1.43.0
    GIT_PROGRESS   TRUE
    GIT_SHALLOW    TRUE
    USES_TERMINAL_DOWNLOAD TRUE
    GIT_SUBMODULES_RECURSE FALSE
    GIT_SUBMODULES
        "third_party/cares"
        "third_party/boringssl-with-bazel"
        "third_party/re2"
        "third_party/abseil-cpp"
)

set(gRPC_BUILD_TESTS OFF)
set(gRPC_BUILD_CODEGEN ON)
set(gRPC_BUILD_GRPC_CPP_PLUGIN ON)
set(gRPC_BUILD_CSHARP_EXT OFF)
set(gRPC_BUILD_GRPC_CSHARP_PLUGIN OFF)
set(gRPC_BUILD_GRPC_NODE_PLUGIN OFF)
set(gRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN OFF)
set(gRPC_BUILD_GRPC_PHP_PLUGIN OFF)
set(gRPC_BUILD_GRPC_PYTHON_PLUGIN OFF)
set(gRPC_BUILD_GRPC_RUBY_PLUGIN OFF)
set(gRPC_BENCHMARK_PROVIDER "none" CACHE STRING "" FORCE)
set(gRPC_PROTOBUF_PROVIDER "module" CACHE STRING "" FORCE)
set(gRPC_ZLIB_PROVIDER "package" CACHE STRING "" FORCE)
set(gRPC_USE_PROTO_LITE ON CACHE BOOL "" FORCE)

FetchContent_GetProperties(grpc)
if(NOT grpc_POPULATED)
    FetchContent_Populate(grpc)
    add_subdirectory(${grpc_SOURCE_DIR} ${grpc_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Fetch GoogleTest
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.12.1
    GIT_PROGRESS   TRUE
    GIT_SHALLOW    TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Set up protobuf compiler
if(NOT TARGET protobuf::protoc)
    message(FATAL_ERROR "Can not find target protoc")
endif()
set(_gRPC_PROTOBUF_PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)
set(_gRPC_PROTO_GENS_DIR ${CMAKE_BINARY_DIR}/gens)
file(MAKE_DIRECTORY ${_gRPC_PROTO_GENS_DIR})
set(_gRPC_PROTOBUF_WELLKNOWN_INCLUDE_DIR "${protobuf_SOURCE_DIR}/src")

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ZLIB_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/gens)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${PROTOBUF_ROOT_DIR}/src)

# Generate protobuf files
set(_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR} -I ${_gRPC_PROTOBUF_WELLKNOWN_INCLUDE_DIR})

get_filename_component(ABS_FIL ${CMAKE_CURRENT_SOURCE_DIR}/proto/exchange_protocol.proto ABSOLUTE)
get_filename_component(FIL_WE proto/exchange_protocol.proto NAME_WE)
file(RELATIVE_PATH REL_FIL ${CMAKE_CURRENT_SOURCE_DIR} ${ABS_FIL})
get_filename_component(REL_DIR ${REL_FIL} DIRECTORY)
if(NOT REL_DIR)
    set(RELFIL_WE "${FIL_WE}")
else()
    set(RELFIL_WE "${REL_DIR}/${FIL_WE}")
endif()

if(NOT TARGET grpc_cpp_plugin)
    message(FATAL_ERROR "Can not find target grpc_cpp_plugin")
endif()
set(_gRPC_CPP_PLUGIN $<TARGET_FILE:grpc_cpp_plugin>)

add_custom_command(
    OUTPUT  "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.grpc.pb.cc"
            "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.grpc.pb.h"
            "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}_mock.grpc.pb.h"
            "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.pb.cc"
            "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.pb.h"
    COMMAND ${_gRPC_PROTOBUF_PROTOC_EXECUTABLE}
    ARGS --grpc_out=generate_mock_code=true:${_gRPC_PROTO_GENS_DIR}
        --cpp_out=${_gRPC_PROTO_GENS_DIR}
        --plugin=protoc-gen-grpc=${_gRPC_CPP_PLUGIN}
        ${_protobuf_include_path}
        ${REL_FIL}
    DEPENDS ${ABS_FIL} protobuf::protoc grpc_cpp_plugin
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running gRPC C++ protocol buffer compiler on ${REL_FIL}"
    VERBATIM
)

# Create a library for generated protobuf files
add_library(proto_lib STATIC
    "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.grpc.pb.cc"
    "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.pb.cc"
)
target_include_directories(proto_lib PUBLIC
    ${_gRPC_PROTO_GENS_DIR}
    ${_gRPC_PROTOBUF_WELLKNOWN_INCLUDE_DIR}
    ${grpc_SOURCE_DIR}/include
    ${grpc_SOURCE_DIR}/third_party/abseil-cpp
)
target_link_libraries(proto_lib PUBLIC
    protobuf::libprotobuf
    grpc
    grpc++
)

# Server executable
add_executable(server src/server.cpp src/yolo_detector.cpp)
target_link_libraries(server PRIVATE
    proto_lib
    ${OpenCV_LIBS}
)

# Client executable
add_executable(client src/client.cpp)
target_link_libraries(client PRIVATE
    proto_lib
    ${OpenCV_LIBS}
)

# Enable testing
enable_testing()
include(GoogleTest)

# Test: PolygonProcessor
add_executable(test_polygon_processor tests/test_polygon_processor.cpp)
target_include_directories(test_polygon_processor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(test_polygon_processor PRIVATE
    proto_lib
    gtest
    gtest_main
    ${OpenCV_LIBS}
)
gtest_discover_tests(test_polygon_processor)

# Test: YoloDetector
add_executable(test_yolo_detector tests/test_yolo_detector.cpp src/yolo_detector.cpp)
target_include_directories(test_yolo_detector PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(test_yolo_detector PRIVATE
    proto_lib
    gtest
    gtest_main
    ${OpenCV_LIBS}
)
gtest_discover_tests(test_yolo_detector)

# Custom test target
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS test_polygon_processor test_yolo_detector
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all unit tests..."
)

# Doxygen documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs)
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_LATEX NO)
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
    doxygen_add_docs(docs 
        ${CMAKE_CURRENT_SOURCE_DIR}/src 
        ${CMAKE_CURRENT_SOURCE_DIR}/README.md
    )
endif()