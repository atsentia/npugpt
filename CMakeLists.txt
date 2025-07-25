cmake_minimum_required(VERSION 3.20)
project(npugpt VERSION 1.0.0 DESCRIPTION "NPU Optimized GPT-2 for Qualcomm Snapdragon X Elite")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set platform to ARM64 for Snapdragon X Elite
if(NOT CMAKE_GENERATOR_PLATFORM)
    set(CMAKE_GENERATOR_PLATFORM ARM64)
endif()

# Build configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags for ARM64 optimization
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /arch:ARM64")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi")

# Find QNN SDK
set(QNN_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/qualcomm" CACHE PATH "QNN SDK Root Directory")

if(NOT EXISTS "${QNN_SDK_ROOT}/lib/QnnHtp.dll")
    message(WARNING "QNN SDK not found at ${QNN_SDK_ROOT}")
    message(WARNING "Please run setup from INSTALL.md before building")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/picogpt/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann
    ${QNN_SDK_ROOT}/include
)

# Link directories
link_directories(
    ${QNN_SDK_ROOT}/lib
)

# Common libraries
set(QNN_LIBRARIES
    QnnHtp
    QnnSystem
    QnnCpu
)

# Core GPT-2 source files
set(CORE_SOURCES
    src/fused_npu_gpt2_engine.cpp
    src/flashattention2_npu_gpt2_engine.cpp
    src/flashattention2_fused_npu_gpt2_engine.cpp
    src/flashattention2_kvcache_npu_gpt2_engine.cpp
    src/flashattention2_fused_kvcache_npu_gpt2_engine.cpp
    src/npu_callback_profiling.cpp
    picogpt/src/gpt2_loader.cpp
    picogpt/src/bpe_tokenizer.cpp
    picogpt/src/simple_tokenizer.cpp
)

# Demo executable
add_executable(npu_gpt2_demo
    examples/npu_gpt2_demo.cpp
    ${CORE_SOURCES}
)

target_link_libraries(npu_gpt2_demo ${QNN_LIBRARIES})

# Comprehensive benchmark
add_executable(comprehensive_gpt2_benchmark
    benchmarks/comprehensive_gpt2_benchmark.cpp
    ${CORE_SOURCES}
)

target_link_libraries(comprehensive_gpt2_benchmark ${QNN_LIBRARIES})

# Energy efficiency benchmark
add_executable(energy_efficiency_benchmark
    benchmarks/energy_efficiency_benchmark.cpp
    ${CORE_SOURCES}
)

target_link_libraries(energy_efficiency_benchmark ${QNN_LIBRARIES})

# Scalability benchmark suite
add_executable(scalability_benchmark_suite
    benchmarks/scalability_benchmark_suite.cpp
    ${CORE_SOURCES}
)

target_link_libraries(scalability_benchmark_suite ${QNN_LIBRARIES})

# Comprehensive benchmark with KV cache (6-way comparison)
add_executable(comprehensive_gpt2_benchmark_with_kvcache
    benchmarks/comprehensive_gpt2_benchmark_with_kvcache.cpp
    ${CORE_SOURCES}
)

target_link_libraries(comprehensive_gpt2_benchmark_with_kvcache ${QNN_LIBRARIES})

# Standalone KV cache demo
add_executable(standalone_kvcache_demo
    benchmarks/standalone_kvcache_demo.cpp
)

target_link_libraries(standalone_kvcache_demo)

# Attention layer KV cache benchmark
add_executable(attention_kvcache_benchmark
    benchmarks/attention_kvcache_benchmark.cpp
)

target_link_libraries(attention_kvcache_benchmark)

# Simple 6-way variant comparison
add_executable(simple_6way_benchmark
    simple_6way_benchmark.cpp
)

target_link_libraries(simple_6way_benchmark)

# Real GPT-2 KV Cache benchmark with actual text generation
add_executable(real_gpt2_kvcache_benchmark
    real_gpt2_kvcache_benchmark.cpp
    src/real_kvcache_gpt2_engine.cpp
    ${CORE_SOURCES}
)

target_link_libraries(real_gpt2_kvcache_benchmark ${QNN_LIBRARIES})

# Real KV Cache output demonstration
add_executable(demo_real_kvcache_outputs
    demo_real_kvcache_outputs.cpp
)

target_link_libraries(demo_real_kvcache_outputs)

# Consolidated 6-way benchmark
add_executable(consolidated_6way_benchmark
    consolidated_6way_benchmark.cpp
)

target_link_libraries(consolidated_6way_benchmark)

# Realistic KV cache benchmark
add_executable(realistic_kvcache_benchmark
    realistic_kvcache_benchmark.cpp
)

target_link_libraries(realistic_kvcache_benchmark)

# Simple KV cache attention demo (minimal working example)
add_executable(simple_kvcache_attention_demo
    simple_kvcache_attention_demo.cpp
)

target_link_libraries(simple_kvcache_attention_demo)

# Working NPU KV cache benchmark (real implementation)
add_executable(working_npu_kvcache_benchmark
    working_npu_kvcache_benchmark.cpp
)

target_link_libraries(working_npu_kvcache_benchmark)

# Set output directory
set_target_properties(
    npu_gpt2_demo
    comprehensive_gpt2_benchmark
    energy_efficiency_benchmark
    scalability_benchmark_suite
    comprehensive_gpt2_benchmark_with_kvcache
    standalone_kvcache_demo
    attention_kvcache_benchmark
    simple_6way_benchmark
    real_gpt2_kvcache_benchmark
    demo_real_kvcache_outputs
    consolidated_6way_benchmark
    realistic_kvcache_benchmark
    simple_kvcache_attention_demo
    working_npu_kvcache_benchmark
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
)

# Copy QNN DLLs to output directory
add_custom_command(TARGET npu_gpt2_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${QNN_SDK_ROOT}/lib/QnnHtp.dll"
    "${QNN_SDK_ROOT}/lib/QnnHtpV73Stub.dll"
    "${QNN_SDK_ROOT}/lib/QnnSystem.dll"
    "${QNN_SDK_ROOT}/lib/QnnCpu.dll"
    $<TARGET_FILE_DIR:npu_gpt2_demo>
    COMMENT "Copying QNN DLLs to output directory"
)

# Installation rules
install(TARGETS npu_gpt2_demo DESTINATION bin)
install(FILES 
    ${QNN_SDK_ROOT}/lib/QnnHtp.dll
    ${QNN_SDK_ROOT}/lib/QnnHtpV73Stub.dll
    ${QNN_SDK_ROOT}/lib/QnnSystem.dll
    ${QNN_SDK_ROOT}/lib/QnnCpu.dll
    DESTINATION bin
)

# Print configuration summary
message(STATUS "NPU GPT-2 Configuration Summary:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Platform: ${CMAKE_GENERATOR_PLATFORM}")
message(STATUS "  QNN SDK: ${QNN_SDK_ROOT}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Targets: npu_gpt2_demo, comprehensive_gpt2_benchmark, energy_efficiency_benchmark, scalability_benchmark_suite")