#pragma once

#include <algorithm>

#include "noise_common.hpp"

/* Evaluate the noise function at N separate positions. */
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

template <unsigned int N>
float perlin_noise__multi_level(float x, float y, float z,
                                float_v<N> frequency_factors,
                                float_v<N> amplitude_factors) {
  float_v<N> xs = x * frequency_factors;
  float_v<N> ys = y * frequency_factors;
  float_v<N> zs = z * frequency_factors;

  float_v<N> raw_values = eval_noise(xs, ys, zs);
  float_v<N> values = raw_values * amplitude_factors;

  float sum = values.get<0>() + values.get<1>() + values.get<2>() +
              values.get<3>();
  return sum;
}

float perlin_noise(float x, float y, float z, float octaves) {
  float frequency = 1.0f;
  float amplitude = 1.0f;
  float_v<8> frequency_factors_8{1, 2, 4, 8, 16, 32, 64, 128};
  float_v<8> amplitude_factors_8{1.0f,      1.0f / 2,  1.0f / 4,
                                 1.0f / 8,  1.0f / 16, 1.0f / 32,
                                 1.0f / 64, 1.0f / 128};
  float_v<4> frequency_factors_4 = frequency_factors_8.low();
  float_v<4> amplitude_factors_4 = amplitude_factors_8.low();

  float result = 0.0f;

  while (octaves > 0.0f) {
    float amplitudes[4];
    memset(amplitudes, 0, sizeof(amplitudes));
    if (octaves <= 1.0f) {
      amplitudes[0] = octaves;
    } else {
      amplitudes[0] = 1.0f;
      if (octaves <= 2.0f) {
        amplitudes[1] = octaves - 1.0f;
      } else {
        amplitudes[1] = 1.0f;
        if (octaves <= 3.0f) {
          amplitudes[2] = octaves - 2.0f;
        } else {
          amplitudes[2] = 1.0f;
          if (octaves <= 4.0f) {
            amplitudes[3] = octaves - 3.0f;
          } else {
            amplitudes[3] = 1.0f;
          }
        }
      }
    }

    float_v<4> amplitude_factors =
        amplitude_factors_4 * float_v<4>(amplitudes);

    result += perlin_noise__multi_level(x, y, z, frequency_factors_4,
                                        amplitude_factors);

    frequency *= (1 << 4);
    amplitude *= 1.0f / (1 << 4);
    octaves -= 4.0f;
  }

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
