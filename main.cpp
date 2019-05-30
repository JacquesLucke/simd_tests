#include <smmintrin.h>
#include <stdio.h>
#include <xmmintrin.h>

#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

class float_4;
class int32_4;

class float_4 {
  __m128 m_value;

 public:
  float_4(__m128 v) : m_value(v) {}
  float_4(float v) { m_value = _mm_set_ps1(v); }
  float_4(float a, float b, float c, float d) {
    m_value = _mm_set_ps(a, b, c, d);
  }

  __m128 m128() const { return m_value; }

  friend float_4 operator+(const float_4 &a, const float_4 &b) {
    return _mm_add_ps(a.m128(), b.m128());
  }

  friend float_4 operator*(const float_4 &a, const float_4 &b) {
    return _mm_mul_ps(a.m128(), b.m128());
  }

  friend float_4 operator-(const float_4 &a, const float_4 &b) {
    return _mm_sub_ps(a.m128(), b.m128());
  }

  float_4 floor() const { return _mm_floor_ps(m_value); }
  float_4 ceil() const { return _mm_ceil_ps(m_value); }

  int32_4 cast_to_int32() const;

  friend std::ostream &operator<<(std::ostream &stream,
                                  const float_4 &v) {
    char data[128];
    snprintf(data, sizeof(data), "(%.5f, %.5f, %.5f, %.5f)",
             v.get<0>(), v.get<1>(), v.get<2>(), v.get<3>());
    stream << data;
    return stream;
  }

  template <unsigned char Index> float get() const {
    static_assert(Index < 4, "invalid index");
    __m128 shuffled = _mm_shuffle_ps(m_value, m_value, Index);
    return _mm_cvtss_f32(shuffled);
  }
};

class int32_4 {
  __m128i m_value;

 public:
  int32_4() = default;
  int32_4(__m128i v) : m_value(v) {}
  int32_4(int32_t v) { m_value = _mm_set1_epi32(v); }
  int32_4(int32_t a, int32_t b, int32_t c, int32_t d) {
    m_value = _mm_set_epi32(a, b, c, d);
  }

  __m128i m128i() const { return m_value; }

  friend int32_4 operator+(const int32_4 &a, const int32_4 &b) {
    return _mm_add_epi32(a.m128i(), b.m128i());
  }

  friend int32_4 operator*(const int32_4 &a, const int32_4 &b) {
    return _mm_mul_epi32(a.m128i(), b.m128i());
  }

  float_4 as_float() const { return _mm_cvtepi32_ps(m_value); }

  friend std::ostream &operator<<(std::ostream &stream,
                                  const int32_4 &v) {
    char data[128];
    snprintf(data, sizeof(data), "(%d, %d, %d, %d)", v.get<0>(),
             v.get<1>(), v.get<2>(), v.get<3>());
    stream << data;
    return stream;
  }

  template <unsigned char Index> int32_t get() const {
    static_assert(Index < 4, "invalid index");
    return _mm_extract_epi32(m_value, Index);
  }
};

int32_4 float_4::cast_to_int32() const {
  return _mm_castps_si128(m_value);
}

#define PRINT_EXPR(expression)                                       \
  std::cout << #expression << "\t " << (expression) << "\n"

static float_4 hash_position(int32_4 x, int32_4 y, int32_4 z) {
  int32_4 h1 = x * (int32_t)4528654243247 + y * (int32_t)54299802411 +
               z * (int32_t)8736254234431;
  float_4 h2 = h1.as_float() * (1.0f / (float)(1 << 31));
  return h2;
}

static float_4 fade(float_4 t) {
  return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

static float_4 linear_interpolation(float_4 t, float_4 a, float_4 b) {
  return (1 - t) * a + t * b;
}

static float_4 bilinear_interpolation(float_4 t1, float_4 t2,
                                      float_4 v_ll, float_4 v_lh,
                                      float_4 v_hl, float_4 v_hh) {
  float_4 low = linear_interpolation(t2, v_ll, v_lh);
  float_4 high = linear_interpolation(t2, v_hl, v_hh);
  float_4 v = linear_interpolation(t1, low, high);
  return v;
}

static float_4 trilinear_interpolation(float_4 t1, float_4 t2,
                                       float_4 t3, float_4 v_lll,
                                       float_4 v_llh, float_4 v_lhl,
                                       float_4 v_lhh, float_4 v_hll,
                                       float_4 v_hlh, float_4 v_hhl,
                                       float_4 v_hhh) {
  float_4 low =
      bilinear_interpolation(t2, t3, v_lll, v_llh, v_lhl, v_lhh);
  float_4 high =
      bilinear_interpolation(t2, t3, v_hll, v_hlh, v_hhl, v_hhh);
  return linear_interpolation(t1, low, high);
}

/* Evaluate the noise function at 4 separate positions. */
static float_4 eval_noise(float_4 x, float_4 y, float_4 z) {

  /* Compute grid cell boundaries for every point. */
  float_4 x_low = x.floor();
  float_4 y_low = y.floor();
  float_4 z_low = z.floor();
  float_4 x_high = x.ceil();
  float_4 y_high = y.ceil();
  float_4 z_high = z.ceil();

  /* Compute fractional offset into a cell. */
  float_4 x_frac = x - x_low;
  float_4 y_frac = y - y_low;
  float_4 z_frac = z - z_low;

  /* Compute interpolation factors in cell. */
  float_4 x_fac = fade(x_frac);
  float_4 y_fac = fade(y_frac);
  float_4 z_fac = fade(z_frac);

  /* Reinterpret boundary coordinates as ints.
   * They can also be converted, but this is actually not necessary.
   */
  int32_4 x_low_id = x_low.cast_to_int32();
  int32_4 y_low_id = y_low.cast_to_int32();
  int32_4 z_low_id = z_low.cast_to_int32();
  int32_4 x_high_id = x_high.cast_to_int32();
  int32_4 y_high_id = y_high.cast_to_int32();
  int32_4 z_high_id = z_high.cast_to_int32();

  /* Compute cell corner values.
   * There are 4 * 8 = 32 corners to compute.
   */
  float_4 corner_lll = hash_position(x_low_id, y_low_id, z_low_id);
  float_4 corner_llh = hash_position(x_low_id, y_low_id, z_high_id);
  float_4 corner_lhl = hash_position(x_low_id, y_high_id, z_low_id);
  float_4 corner_lhh = hash_position(x_low_id, y_high_id, z_high_id);
  float_4 corner_hll = hash_position(x_high_id, y_low_id, z_low_id);
  float_4 corner_hlh = hash_position(x_high_id, y_low_id, z_high_id);
  float_4 corner_hhl = hash_position(x_high_id, y_high_id, z_low_id);
  float_4 corner_hhh = hash_position(x_high_id, y_high_id, z_high_id);

  /* Interpolate corner values for position in cell. */
  float_4 result = trilinear_interpolation(
      x_fac, y_fac, z_fac, corner_lll, corner_llh, corner_lhl,
      corner_lhh, corner_hll, corner_hlh, corner_hhl, corner_hhh);

  return result;
}

int main(int argc, char const *argv[]) {
  float_4 a{3, 4, 5, 2};
  float_4 b{7.4f, 8.0f, 2.23f, 3.5f};
  int32_4 c{6, 7, 8, 9};
  PRINT_EXPR(a);
  PRINT_EXPR(b);
  PRINT_EXPR(c);
  PRINT_EXPR(b.floor());
  PRINT_EXPR(b.ceil());
  PRINT_EXPR(b.floor().cast_to_int32());
  PRINT_EXPR(b.ceil().cast_to_int32());
  PRINT_EXPR(a * b);

  float step = 0.1f;
  for (float y = 0.0f; y <= 3.0f; y += step) {
    for (float x = 0.0f; x <= 1.0f; x += step) {
      float_4 result = eval_noise(x, y, 0).get<0>();
      std::cout << std::fixed << std::setw(8) << std::setprecision(3)
                << result.get<0>() << " ";
    }
    std::cout << "\n";
  }

  std::getchar();
  return 0;
}
