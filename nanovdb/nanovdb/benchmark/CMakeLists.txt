# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Builds the ReadAccessor benchmark twice -- once per accessor mode (OLD vs NEW)
# -- so you can run them back-to-back and compare without a full reconfigure.
# When NANOVDB_USE_CUDA is ON, the GPU counterpart is built the same way.

cmake_minimum_required(VERSION 3.24)
project(NanoVDBBenchmark LANGUAGES CXX)

# --- CPU benchmark -----------------------------------------------------------

# NEW accessor (bug-fixed, default). NanoVDB.h enables NANOVDB_USE_OLD_ACCESSOR
# unless suppressed, so the NEW build defines the negative macro.
add_executable(bench_accessor_new BenchAccessor.cc)
target_link_libraries(bench_accessor_new PRIVATE nanovdb)
target_compile_definitions(bench_accessor_new PRIVATE NANOVDB_NO_OLD_ACCESSOR)

# OLD accessor (legacy, pre-fix behaviour).
add_executable(bench_accessor_old BenchAccessor.cc)
target_link_libraries(bench_accessor_old PRIVATE nanovdb)
target_compile_definitions(bench_accessor_old PRIVATE NANOVDB_USE_OLD_ACCESSOR)

# --- GPU benchmark -----------------------------------------------------------

if(NANOVDB_USE_CUDA)
  add_executable(bench_accessor_cuda_new BenchAccessorCuda.cu)
  target_link_libraries(bench_accessor_cuda_new PRIVATE nanovdb)
  target_compile_definitions(bench_accessor_cuda_new PRIVATE NANOVDB_NO_OLD_ACCESSOR)
  set_target_properties(bench_accessor_cuda_new PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  add_executable(bench_accessor_cuda_old BenchAccessorCuda.cu)
  target_link_libraries(bench_accessor_cuda_old PRIVATE nanovdb)
  target_compile_definitions(bench_accessor_cuda_old PRIVATE NANOVDB_USE_OLD_ACCESSOR)
  set_target_properties(bench_accessor_cuda_old PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
