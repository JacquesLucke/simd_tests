#include <smmintrin.h>
#include <stdio.h>
#include <xmmintrin.h>

#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "simd_core.hpp"

#define PRINT_EXPR(expression)                                       \
  std::cout << #expression << "\t " << (expression) << "\n"

template <unsigned int N>
static float_v<N> hash_position(int32_v<N> x, int32_v<N> y,
                                int32_v<N> z) {
  int32_v<N> h1 = x * (int32_t)4528654243247 +
                  y * (int32_t)54299802411 +
                  z * (int32_t)8736254234431;
  float_v<N> h2 = h1.as_float() * (1.0f / (float)(1 << 31));
  return h2;
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

/* Evaluate the noise function at 4 separate positions. */
template <unsigned int N>
static float_v<N> eval_noise(float_v<N> x, float_v<N> y,
                             float_v<N> z) {

  /* Compute grid cell boundaries for every point. */
  float_v<N> x_low = x.floor();
  float_v<N> y_low = y.floor();
  float_v<N> z_low = z.floor();
  float_v<N> x_high = x.ceil();
  float_v<N> y_high = y.ceil();
  float_v<N> z_high = z.ceil();

  /* Compute fractional offset into a cell. */
  float_v<N> x_frac = x - x_low;
  float_v<N> y_frac = y - y_low;
  float_v<N> z_frac = z - z_low;

  /* Compute interpolation factors in cell. */
  float_v<N> x_fac = fade(x_frac);
  float_v<N> y_fac = fade(y_frac);
  float_v<N> z_fac = fade(z_frac);

  /* Reinterpret boundary coordinates as ints.
   * They can also be converted, but this is actually not necessary.
   */
  int32_v<N> x_low_id = x_low.cast_to_int32();
  int32_v<N> y_low_id = y_low.cast_to_int32();
  int32_v<N> z_low_id = z_low.cast_to_int32();
  int32_v<N> x_high_id = x_high.cast_to_int32();
  int32_v<N> y_high_id = y_high.cast_to_int32();
  int32_v<N> z_high_id = z_high.cast_to_int32();

  /* Compute cell corner values.
   * There are 4 * 8 = 32 corners to compute.
   */
  float_v<N> corner_lll = hash_position(x_low_id, y_low_id, z_low_id);
  float_v<N> corner_llh =
      hash_position(x_low_id, y_low_id, z_high_id);
  float_v<N> corner_lhl =
      hash_position(x_low_id, y_high_id, z_low_id);
  float_v<N> corner_lhh =
      hash_position(x_low_id, y_high_id, z_high_id);
  float_v<N> corner_hll =
      hash_position(x_high_id, y_low_id, z_low_id);
  float_v<N> corner_hlh =
      hash_position(x_high_id, y_low_id, z_high_id);
  float_v<N> corner_hhl =
      hash_position(x_high_id, y_high_id, z_low_id);
  float_v<N> corner_hhh =
      hash_position(x_high_id, y_high_id, z_high_id);

  /* Interpolate corner values for position in cell. */
  float_v<N> result = interpolate_trilinear(
      x_fac, y_fac, z_fac, corner_lll, corner_llh, corner_lhl,
      corner_lhh, corner_hll, corner_hlh, corner_hhl, corner_hhh);

  return result;
}

int main(int argc, char const *argv[]) {
  float_v<4> a{3, 4, 5, 2};
  float_v<4> b{7.4f, 8.0f, 2.23f, 3.5f};
  float_v<8> c{a, b};
  PRINT_EXPR(a);
  PRINT_EXPR(a);
  PRINT_EXPR(b);
  PRINT_EXPR(c);

  float step = 0.1f;
  for (float y = 0.0f; y <= 3.0f; y += step) {
    for (float x = 0.0f; x <= 1.0f; x += step) {
      float result = eval_noise<4>(x, y, 0.0f).get<0>();
      std::cout << std::fixed << std::setw(8) << std::setprecision(3)
                << result << " ";
    }
    std::cout << "\n";
  }

  std::getchar();
  return 0;
}
