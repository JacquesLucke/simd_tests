#pragma once

#include <immintrin.h>
#include <smmintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <xmmintrin.h>

#include <iostream>

template <unsigned int N> class float_v;
template <unsigned int N> class int32_v;

template <unsigned int N> class float_v {
  static const int N_Half = N / 2;
  static_assert(N_Half * 2 == N, "N is not a power of two");

 private:
  float_v<N_Half> m_low;
  float_v<N_Half> m_high;

 public:
  float_v() = default;
  float_v(float value) : m_low(value), m_high(value) {}
  float_v(float_v<N_Half> low, float_v<N_Half> high)
      : m_low(low), m_high(high) {}

  float_v<N_Half> low() const { return m_low; }
  float_v<N_Half> high() const { return m_high; }

  friend float_v operator+(float_v a, float_v b) {
    return float_v(a.low() + b.low(), a.high() + b.high());
  }

  friend float_v operator-(float_v a, float_v b) {
    return float_v(a.low() - b.low(), a.high() - b.high());
  }

  friend float_v operator*(float_v a, float_v b) {
    return float_v(a.low() * b.low(), a.high() * b.high());
  }

  float_v floor() const {
    return float_v(m_low.floor(), m_high.floor());
  }

  float_v ceil() const {
    return float_v(m_low.ceil(), m_high.ceil());
  }

  int32_v<N> as_int32() const;
  int32_v<N> cast_to_int32() const;

  friend std::ostream &operator<<(std::ostream &stream,
                                  float_v value) {
    stream << "(" << value.low() << ", " << value.high() << ")";
    return stream;
  }

  template <int Index> float get() const {
    static_assert(Index < N, "invalid index");
    if (Index < N_Half) {
      return m_low.get<Index>();
    } else {
      return m_high.get < Index - N_Half < 0 ? 0
                                             : Index - N_Half > ();
    }
  }
};

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

  int32_v<1> as_int32() const;
  int32_v<1> cast_to_int32() const;

  friend std::ostream &operator<<(std::ostream &stream,
                                  float_v value) {
    stream << value.m_value;
    return stream;
  }

  template <int Index> float get() {
    static_assert(Index == 0, "invalid index");
    return m_value;
  }
};

template <> class float_v<4> {
 private:
  __m128 m_value;

 public:
  float_v() = default;
  float_v(__m128 v) : m_value(v) {}
  float_v(float v) : m_value(_mm_set_ps1(v)) {}
  float_v(float v0, float v1, float v2, float v3)
      : m_value(_mm_set_ps(v3, v2, v1, v0)) {}

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

  int32_v<4> as_int32() const;
  int32_v<4> cast_to_int32() const;

  friend std::ostream &operator<<(std::ostream &stream, float_v v) {
    char data[128];
    snprintf(data, sizeof(data), "(%.5f, %.5f, %.5f, %.5f)",
             v.get<0>(), v.get<1>(), v.get<2>(), v.get<3>());
    stream << data;
    return stream;
  }

  template <int Index> float get() const {
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
  float_v(float v0, float v1, float v2, float v3, float v4, float v5,
          float v6, float v7)
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

  int32_v<8> as_int32() const;
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

  template <int Index> float get() const {
    static_assert(Index < 8, "invalid index");
    float array[8];
    _mm256_store_ps(array, m_value);
    return array[Index];
  }
};

template <unsigned int N> class int32_v {
 private:
  static const int N_Half = N / 2;
  static_assert(N_Half * 2 == N, "N is not a power of two");

  int32_v<N_Half> m_low;
  int32_v<N_Half> m_high;

 public:
  int32_v() = default;
  int32_v(int32_t v) : m_low(v), m_high(v) {}
  int32_v(int32_v<N_Half> low, int32_v<N_Half> high)
      : m_low(low), m_high(high) {}

  int32_v<N_Half> low() const { return m_low; }
  int32_v<N_Half> high() const { return m_high; }

  friend int32_v operator+(int32_v a, int32_v b) {
    return int32_v(a.low() + b.low(), a.high() + b.high());
  }

  friend int32_v operator-(int32_v a, int32_v b) {
    return int32_v(a.low() - b.low(), a.high() - b.high());
  }

  friend int32_v operator*(int32_v a, int32_v b) {
    return int32_v(a.low() * b.low(), a.high() * b.high());
  }

  friend int32_v operator^(int32_v a, int32_v b) {
    return int32_v(a.low() ^ b.low(), a.high() ^ b.high());
  }

  template <unsigned int Count> int32_v rotate() const {
    return int32_v(m_low.rotate<Count>(), m_high.rotate<Count>());
  }

  float_v<N> as_float() const {
    return float_v<N>(m_low.as_float(), m_high.as_float());
  }

  friend std::ostream &operator<<(std::ostream &stream, int32_v v) {
    stream << "(" << v.low() << ", " << v.high() << ")";
    return stream;
  }

  template <int Index> int32_t get() const {
    static_assert(Index < N, "invalid index");
    if (Index < N_Half) {
      return m_low.get<Index>();
    } else {
      return m_high.get < Index - N_Half < 0 ? 0
                                             : Index - N_Half > ();
    }
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

  friend int32_v operator-(int32_v a, int32_v b) {
    return a.value() - b.value();
  }

  friend int32_v operator*(int32_v a, int32_v b) {
    return a.value() * b.value();
  }

  friend int32_v operator^(int32_v a, int32_v b) {
    return a.value() ^ b.value();
  }

  template <unsigned int Count> int32_v rotate() const {
    return ((uint32_t)m_value << Count) |
           ((uint32_t)m_value >> (32 - Count));
  }

  float_v<1> as_float() const { return (float)m_value; }

  friend std::ostream &operator<<(std::ostream &stream, int32_v v) {
    stream << v.m_value;
    return stream;
  }

  template <int Index> int32_t get() const {
    static_assert(Index == 0, "invalid index");
    return m_value;
  }
};

template <> class int32_v<4> {
 private:
  __m128i m_value;

 public:
  int32_v() = default;
  int32_v(__m128i v) : m_value(v) {}
  int32_v(int32_t v) : m_value(_mm_set1_epi32(v)) {}
  int32_v(int32_t v0, int32_t v1, int32_t v2, int32_t v3)
      : m_value(_mm_set_epi32(v3, v2, v1, v0)) {}

  __m128i m128i() const { return m_value; }

  friend int32_v operator+(int32_v a, int32_v b) {
    return _mm_add_epi32(a.m128i(), b.m128i());
  }

  friend int32_v operator-(int32_v a, int32_v b) {
    return _mm_sub_epi32(a.m128i(), b.m128i());
  }

  friend int32_v operator*(int32_v a, int32_v b) {
    return _mm_mullo_epi32(a.m128i(), b.m128i());
  }

  friend int32_v operator^(int32_v a, int32_v b) {
    return _mm_xor_si128(a.m128i(), b.m128i());
  }

  template <unsigned int Count> int32_v rotate() const {
    __m128i left = _mm_slli_epi32(m_value, Count);
    __m128i right = _mm_srli_epi32(m_value, 32 - Count);
    return _mm_or_si128(left, right);
  }

  float_v<4> as_float() const { return _mm_cvtepi32_ps(m_value); }

  friend std::ostream &operator<<(std::ostream &stream, int32_v v) {
    char data[128];
    snprintf(data, sizeof(data), "(%d, %d, %d, %d)", v.get<0>(),
             v.get<1>(), v.get<2>(), v.get<3>());
    stream << data;
    return stream;
  }

  template <int Index> int32_t get() const {
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
  int32_v(int32_t v0, int32_t v1, int32_t v2, int32_t v3, int32_t v4,
          int32_t v5, int32_t v6, int32_t v7)
      : m_value(_mm256_set_epi32(v7, v6, v5, v4, v3, v2, v1, v0)) {}

  __m256i m256i() const { return m_value; }

  friend int32_v operator+(int32_v a, int32_v b) {
    return _mm256_add_epi32(a.m256i(), b.m256i());
  }

  friend int32_v operator-(int32_v a, int32_v b) {
    return _mm256_sub_epi32(a.m256i(), b.m256i());
  }

  friend int32_v operator*(int32_v a, int32_v b) {
    return _mm256_mullo_epi32(a.m256i(), b.m256i());
  }

  friend int32_v operator^(int32_v a, int32_v b) {
    return _mm256_xor_si256(a.m256i(), b.m256i());
  }

  template <unsigned int Count> int32_v rotate() const {
    __m256i left = _mm256_slli_epi32(m_value, Count);
    __m256i right = _mm256_srli_epi32(m_value, 32 - Count);
    return _mm256_or_si256(left, right);
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

  template <int Index> int32_t get() const {
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

template <unsigned int N>
int32_v<N> float_v<N>::cast_to_int32() const {
  return int32_v<N>(m_low.cast_to_int32(), m_high.cast_to_int32());
}

int32_v<4> float_v<4>::cast_to_int32() const {
  return _mm_castps_si128(m_value);
}

int32_v<8> float_v<8>::cast_to_int32() const {
  return _mm256_castps_si256(m_value);
}

int32_v<1> float_v<1>::as_int32() const {
  return int32_v<1>((int32_t)m_value);
}

int32_v<4> float_v<4>::as_int32() const {
  return _mm_cvtps_epi32(m_value);
}

int32_v<8> float_v<8>::as_int32() const {
  return _mm256_cvtps_epi32(m_value);
}
