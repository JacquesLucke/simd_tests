#pragma once

#include <immintrin.h>
#include <smmintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <xmmintrin.h>

#include <iostream>

template <unsigned int N> class float_v;
template <unsigned int N> class int32_v;

template <> class float_v<1> {
 private:
  float m_value;

 public:
  float_v() = default;
  float_v(float value) : m_value(value) {}

  float value() const { return m_value; }

  friend float_v operator+(float_v a, float_v b) {
    return a.value() + b.value();
  }
  friend float_v operator*(float_v a, float_v b) {
    return a.value() * b.value();
  }
  friend float_v operator-(float_v a, float_v b) {
    return a.value() - b.value();
  }

  float_v floor() const { return std::floorf(m_value); }
  float_v ceil() const { return std::ceilf(m_value); }

  int32_v<1> cast_to_int32() const;

  friend std::ostream &operator<<(std::ostream &stream,
                                  float_v value) {
    stream << value.m_value;
    return stream;
  }
};

template <> class float_v<4> {
 private:
  __m128 m_value;

 public:
  float_v() = default;
  float_v(__m128 v) : m_value(v) {}
  float_v(float v) : m_value(_mm_set_ps1(v)) {}
  float_v(float a, float b, float c, float d)
      : m_value(_mm_set_ps(a, b, c, d)) {}

  __m128 m128() const { return m_value; }

  friend float_v operator+(float_v a, float_v b) {
    return _mm_add_ps(a.m128(), b.m128());
  }

  friend float_v operator*(float_v a, float_v b) {
    return _mm_mul_ps(a.m128(), b.m128());
  }

  friend float_v operator-(float_v a, float_v b) {
    return _mm_sub_ps(a.m128(), b.m128());
  }

  float_v floor() const { return _mm_floor_ps(m_value); }
  float_v ceil() const { return _mm_ceil_ps(m_value); }

  int32_v<4> cast_to_int32() const;

  friend std::ostream &operator<<(std::ostream &stream, float_v v) {
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

template <> class float_v<8> {
 private:
  __m256 m_value;

 public:
  float_v() = default;
  float_v(__m256 v) : m_value(v) {}
  float_v(float_v<4> low, float_v<4> high)
      : m_value(_mm256_setr_m128(low.m128(), high.m128())) {}
  float_v(float v) : m_value(_mm256_set1_ps(v)) {}
  float_v(float v7, float v6, float v5, float v4, float v3, float v2,
          float v1, float v0)
      : m_value(_mm256_set_ps(v7, v6, v5, v4, v3, v2, v1, v0)) {}

  __m256 m256() const { return m_value; }

  friend float_v operator+(float_v a, float_v b) {
    return _mm256_add_ps(a.m256(), b.m256());
  }

  friend float_v operator*(float_v a, float_v b) {
    return _mm256_mul_ps(a.m256(), b.m256());
  }

  friend float_v operator-(float_v a, float_v b) {
    return _mm256_sub_ps(a.m256(), b.m256());
  }

  float_v<4> low() const { return _mm256_extractf128_ps(m_value, 0); }

  float_v<4> high() const {
    return _mm256_extractf128_ps(m_value, 1);
  }

  float_v floor() const { return _mm256_floor_ps(m_value); }
  float_v ceil() const { return _mm256_ceil_ps(m_value); }

  int32_v<8> cast_to_int32() const;

  friend std::ostream &operator<<(std::ostream &stream, float_v v) {
    char data[128];
    snprintf(data, sizeof(data),
             "(%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f)",
             v.get<0>(), v.get<1>(), v.get<2>(), v.get<3>(),
             v.get<4>(), v.get<5>(), v.get<6>(), v.get<7>());
    stream << data;
    return stream;
  }

  template <unsigned char Index> float get() const {
    static_assert(Index < 8, "invalid index");
    float array[8];
    _mm256_store_ps(array, m_value);
    return array[Index];
  }
};

template <> class int32_v<1> {
 private:
  int32_t m_value;

 public:
  int32_v() = default;
  int32_v(int32_t v) : m_value(v) {}

  int32_t value() const { return m_value; }

  friend int32_v operator+(int32_v a, int32_v b) {
    return a.value() + b.value();
  }
  friend int32_v operator*(int32_v a, int32_v b) {
    return a.value() * b.value();
  }

  float_v<1> as_float() const { return (float)m_value; }

  friend std::ostream &operator<<(std::ostream &stream, int32_v v) {
    stream << v.m_value;
    return stream;
  }
};

template <> class int32_v<4> {
 private:
  __m128i m_value;

 public:
  int32_v() = default;
  int32_v(__m128i v) : m_value(v) {}
  int32_v(int32_t v) : m_value(_mm_set1_epi32(v)) {}
  int32_v(int32_t a, int32_t b, int32_t c, int32_t d)
      : m_value(_mm_set_epi32(a, b, c, d)) {}

  __m128i m128i() const { return m_value; }

  friend int32_v operator+(int32_v a, int32_v b) {
    return _mm_add_epi32(a.m128i(), b.m128i());
  }

  friend int32_v operator*(int32_v a, int32_v b) {
    return _mm_mul_epi32(a.m128i(), b.m128i());
  }

  float_v<4> as_float() const { return _mm_cvtepi32_ps(m_value); }

  friend std::ostream &operator<<(std::ostream &stream, int32_v v) {
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

template <> class int32_v<8> {
 private:
  __m256i m_value;

 public:
  int32_v() = default;
  int32_v(__m256i v) : m_value(v) {}
  int32_v(int32_t v) : m_value(_mm256_set1_epi32(v)) {}
  int32_v(int32_t v7, int32_t v6, int32_t v5, int32_t v4, int32_t v3,
          int32_t v2, int32_t v1, int32_t v0)
      : m_value(_mm256_set_epi32(v7, v6, v5, v4, v3, v2, v1, v0)) {}

  __m256i m256() const { return m_value; }

  friend int32_v operator+(int32_v a, int32_v b) {
    return _mm256_add_epi32(a.m256(), b.m256());
  }

  friend int32_v operator*(int32_v a, int32_v b) {
    return _mm256_mul_epi32(a.m256(), b.m256());
  }

  float_v<8> as_float() const { return _mm256_cvtepi32_ps(m_value); }

  friend std::ostream &operator<<(std::ostream &stream, int32_v v) {
    char data[128];
    snprintf(data, sizeof(data), "(%d, %d, %d, %d, %d, %d, %d, %d)",
             v.get<0>(), v.get<1>(), v.get<2>(), v.get<3>(),
             v.get<4>(), v.get<5>(), v.get<6>(), v.get<7>());
    stream << data;
    return stream;
  }

  template <unsigned char Index> int32_t get() const {
    static_assert(Index < 8, "invalid index");
    return _mm256_extract_epi32(m_value, Index);
  }
};

int32_v<1> float_v<1>::cast_to_int32() const {
  union {
    float f;
    int i;
  } value;
  value.f = m_value;
  return value.i;
}

int32_v<4> float_v<4>::cast_to_int32() const {
  return _mm_castps_si128(m_value);
}

int32_v<8> float_v<8>::cast_to_int32() const {
  return _mm256_castps_si256(m_value);
}
