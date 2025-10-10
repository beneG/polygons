cmake_minimum_required(VERSION 3.5)

# Fetch Google Test
#include(FetchContent)

#FetchContent_Declare(
#    googletest
#    GIT_REPOSITORY https://github.com/google/googletest.git
#    GIT_TAG        release-1.12.1
#    GIT_PROGRESS   TRUE
#    GIT_SHALLOW    TRUE
#)

# For Windows: Prevent overriding the parent project's compiler/linker settings
#set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

#FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()
include(GoogleTest)

#{#####################################################################################
set(_protobuf_include_path --proto_path /app/proto -I /app/proto -I ${_gRPC_PROTOBUF_WELLKNOWN_INCLUDE_DIR})


get_filename_component(ABS_FIL /app/proto/exchange_protocol.proto ABSOLUTE)
get_filename_component(FIL_WE /app/proto/exchange_protocol.proto NAME_WE)
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
    DEPENDS ${ABS_FIL} ${_gRPC_PROTOBUF_PROTOC} ${_gRPC_CPP_PLUGIN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running gRPC C++ protocol buffer compiler on proto/exchange_protocol.proto"
    VERBATIM
)

# target_sources(${target} PRIVATE
#     "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.grpc.pb.cc"
#     "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.grpc.pb.h"
#     "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}_mock.grpc.pb.h"
#     "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.pb.cc"
#     "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.pb.h"
# )
# target_include_directories(${target} PRIVATE
#     $<BUILD_INTERFACE:${_gRPC_PROTO_GENS_DIR}>
#     $<BUILD_INTERFACE:${_gRPC_PROTOBUF_WELLKNOWN_INCLUDE_DIR}>
#     $<BUILD_INTERFACE:${grpc_SOURCE_DIR}/include>
#     $<BUILD_INTERFACE:${grpc_SOURCE_DIR}/third_party/abseil-cpp>
# )






#}#####################################################################################




# Test executable for PolygonProcessor
add_executable(test_polygon_processor test_polygon_processor.cpp)

# Generate protobuf files for test_polygon_processor
#target_add_protobuf(test_polygon_processor ${CMAKE_SOURCE_DIR}/../proto/exchange_protocol.proto)
#target_add_protobuf(test_polygon_processor /app/proto/exchange_protocol.proto)

#target_include_directories(test_polygon_processor PRIVATE
#    $<BUILD_INTERFACE:${_gRPC_PROTO_GENS_DIR}>
#    $<BUILD_INTERFACE:${_gRPC_PROTOBUF_WELLKNOWN_INCLUDE_DIR}>
#    $<BUILD_INTERFACE:${grpc_SOURCE_DIR}/include>
#    $<BUILD_INTERFACE:${grpc_SOURCE_DIR}/third_party/abseil-cpp>
#)

#target_sources(test_polygon_processor PRIVATE
#    "${_gRPC_PROTO_GENS_DIR}/proto/exchange_protocol.grpc.pb.cc"
#    "${_gRPC_PROTO_GENS_DIR}/proto/exchange_protocol.grpc.pb.h"
#    "${_gRPC_PROTO_GENS_DIR}/proto/exchange_protocol_mock.grpc.pb.h"
#    "${_gRPC_PROTO_GENS_DIR}/proto/exchange_protocol.pb.cc"
#    "${_gRPC_PROTO_GENS_DIR}/proto/exchange_protocol.pb.h"
#)

target_sources(test_polygon_processor PRIVATE
    "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.grpc.pb.cc"
    "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.grpc.pb.h"
    "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}_mock.grpc.pb.h"
    "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.pb.cc"
    "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.pb.h"
)
target_include_directories(test_polygon_processor PRIVATE
    $<BUILD_INTERFACE:${_gRPC_PROTO_GENS_DIR}>
    $<BUILD_INTERFACE:${_gRPC_PROTOBUF_WELLKNOWN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${grpc_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${grpc_SOURCE_DIR}/third_party/abseil-cpp>
)



target_include_directories(test_polygon_processor PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_polygon_processor PRIVATE
    gtest
    gtest_main
    ${OpenCV_LIBS}
    libprotobuf
)

# Test executable for YoloDetector
#add_executable(test_yolo_detector 
#    test_yolo_detector.cpp 
#    ${CMAKE_SOURCE_DIR}/src/yolo_detector.cpp
#)

# Generate protobuf files for test_yolo_detector
#target_add_protobuf(test_yolo_detector ${CMAKE_SOURCE_DIR}/proto/exchange_protocol.proto)

#target_include_directories(test_yolo_detector PRIVATE
#    ${CMAKE_SOURCE_DIR}/src
#    ${OpenCV_INCLUDE_DIRS}
#)

#target_link_libraries(test_yolo_detector PRIVATE
#    gtest
#    gtest_main
#    ${OpenCV_LIBS}
#    libprotobuf
#)

# Register tests with CTest
gtest_discover_tests(test_polygon_processor)
#gtest_discover_tests(test_yolo_detector)

# Add custom target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    #DEPENDS test_polygon_processor test_yolo_detector
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all unit tests..."
)