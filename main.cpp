#include <smmintrin.h>
#include <stdio.h>
#include <xmmintrin.h>

#include <cassert>
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

  template <unsigned char Index> int32_t extract() const {
    static_assert(Index < 4, "invalid index");
    return _mm_extract_ps(m_value, Index);
  }

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
  int32_4(__m128i v) : m_value(v) {}
  int32_4(int32_t v) { m_value = _mm_set1_epi32(v); }
  int32_4(int32_t a, int32_t b, int32_t c, int32_t d) {
    m_value = _mm_set_epi32(a, b, c, d);
  }

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
  std::getchar();
  return 0;
}
