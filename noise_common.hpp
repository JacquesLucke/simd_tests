#pragma once

#include "simd_core.hpp"

template <unsigned int N>
static float_v<N> hash_position(int32_v<N> x, int32_v<N> y,
                                int32_v<N> z) {
#define xor_rot(a, b, k)                                             \
  a = a ^ b;                                                         \
  a = a - b.rotate<k>();

  int32_v<N> magic = int32_v<N>(0xdeadbeef);
  int32_v<N> a = x * magic;
  int32_v<N> b = y * magic;
  int32_v<N> c = z * magic;

  xor_rot(c, b, 14);
  xor_rot(a, c, 11);
  xor_rot(b, a, 25);
  xor_rot(c, b, 16);
  xor_rot(a, c, 4);
  xor_rot(b, a, 14);
  xor_rot(c, b, 24);

  float_v<N> result = c.as_float();
  return result * (1.0f / (1 << 31));

#undef xor_rot
}

template <unsigned int N> static float_v<N> fade(float_v<N> t) {
  return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

template <unsigned int N>
static float_v<N> interpolate_linear(float_v<N> t, float_v<N> v_0,
                                     float_v<N> v_1) {
  return (1 - t) * v_0 + t * v_1;
}

template <unsigned int N>
static float_v<N>
interpolate_bilinear(float_v<N> t1, float_v<N> t2, float_v<N> v_0_0,
                     float_v<N> v_0_1, float_v<N> v_1_0,
                     float_v<N> v_1_1) {
  float_v<N> v_t1_0 = interpolate_linear(t1, v_0_0, v_1_0);
  float_v<N> v_t1_1 = interpolate_linear(t1, v_0_1, v_1_1);
  float_v<N> v_t1_t2 = interpolate_linear(t2, v_t1_0, v_t1_1);
  return v_t1_t2;
}

template <unsigned int N>
static float_v<N>
interpolate_trilinear(float_v<N> t1, float_v<N> t2, float_v<N> t3,
                      float_v<N> v_0_0_0, float_v<N> v_0_0_1,
                      float_v<N> v_0_1_0, float_v<N> v_0_1_1,
                      float_v<N> v_1_0_0, float_v<N> v_1_0_1,
                      float_v<N> v_1_1_0, float_v<N> v_1_1_1) {
  float_v<N> v_t1_t2_0 = interpolate_bilinear(
      t1, t2, v_0_0_0, v_0_1_0, v_1_0_0, v_1_1_0);
  float_v<N> v_t1_t2_1 = interpolate_bilinear(
      t1, t2, v_0_0_1, v_0_1_1, v_1_0_1, v_1_1_1);
  float_v<N> v_t1_t2_t3 =
      interpolate_linear(t3, v_t1_t2_0, v_t1_t2_1);
  return v_t1_t2_t3;
}
