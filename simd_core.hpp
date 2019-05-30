#pragma once

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
  float_v(float v) { m_value = _mm_set_ps1(v); }
  float_v(float a, float b, float c, float d) {
    m_value = _mm_set_ps(a, b, c, d);
  }

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
  int32_v(int32_t v) { m_value = _mm_set1_epi32(v); }
  int32_v(int32_t a, int32_t b, int32_t c, int32_t d) {
    m_value = _mm_set_epi32(a, b, c, d);
  }

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
