#include <smmintrin.h>
#include <stdio.h>
#include <xmmintrin.h>

#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "simd_core.hpp"

#define PRINT_EXPR(expression)                                       \
  std::cout << #expression << "\t " << (expression) << "\n"

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

  /* Get IDs of cell corners.
   * TODO: Compare cast vs convert to int.
   */
  int32_v<N> x_low_id = x_low.as_int32();
  int32_v<N> y_low_id = y_low.as_int32();
  int32_v<N> z_low_id = z_low.as_int32();
  int32_v<N> x_high_id = x_high.as_int32();
  int32_v<N> y_high_id = y_high.as_int32();
  int32_v<N> z_high_id = z_high.as_int32();

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

float eval_perlin_1(float x, float y, float z) {
  float v1 = eval_noise<1>(x, y, z).get<0>();
  float v2 = eval_noise<1>(x * 2, y * 2, z * 2).get<0>();
  float v3 = eval_noise<1>(x * 4, y * 4, z * 4).get<0>();
  float v4 = eval_noise<1>(x * 8, y * 8, z * 8).get<0>();
  return v1 + v2 * 0.5f + v3 * 0.25f + v4 * 0.125f;
}
float eval_perlin_2(float x, float y, float z) {
  float_v<4> x_positions{x, x * 2, x * 4, x * 8};
  float_v<4> y_positions{y, y * 2, y * 4, y * 8};
  float_v<4> z_positions{z, z * 2, z * 4, z * 8};
  float_v<4> values =
      eval_noise(x_positions, y_positions, z_positions);
  float v1 = values.get<0>();
  float v2 = values.get<1>();
  float v3 = values.get<2>();
  float v4 = values.get<3>();
  return v1 + v2 * 0.5f + v3 * 0.25f + v4 * 0.125f;
}

std::vector<float> noise_texture(unsigned int width,
                                 unsigned int height, float scale) {
  std::vector<float> pixels;
  pixels.reserve(width * height);
  for (unsigned int y = 0; y < height; y++) {
    for (unsigned int x = 0; x < width; x++) {
      float value = eval_perlin_1(x * scale, y * scale, 0.0f);
      pixels.push_back(value);
    }
  }
  return pixels;
}

std::string texture_as_json(std::vector<float> pixels,
                            unsigned int width, unsigned int height) {
  assert(pixels.size() == width * height);
  std::stringstream ss;
  ss << "{\"width\":" << width << ", \"height\":" << height
     << ", \"pixels\":[";
  for (unsigned int i = 0; i < pixels.size(); i++) {
    ss << pixels[i];
    if (i < pixels.size() - 1) {
      ss << ",";
    }
  }
  ss << "]}\n";
  return ss.str();
}

int main(int argc, char const *argv[]) {
  float_v<4> a{3, 4, 5, 2};
  float_v<4> b{7.4f, 8.0f, 2.23f, 3.5f};
  float_v<8> c{a, b};
  int32_v<4> d{5, 6, 7, 8};

  unsigned int width = 1000;
  unsigned int height = 1000;
  auto pixels = noise_texture(width, height, 0.01f);
  auto texture_json = texture_as_json(pixels, width, height);

  std::ofstream myfile{"test.json"};
  myfile << texture_json;
  myfile.close();

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
