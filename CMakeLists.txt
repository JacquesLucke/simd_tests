cmake_minimum_required(VERSION 3.0.0)
project(simd_test VERSION 0.1.0)

add_executable(simd_test main.cpp simd_core.hpp noise_common.hpp perlin_noise.hpp timeit.hpp)
