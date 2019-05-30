#include <smmintrin.h>
#include <stdio.h>
#include <xmmintrin.h>

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

class fvec4 {
  __m128 m_value;

 public:
  fvec4(__m128 v) : m_value(v) {}
  fvec4(float v) { m_value = _mm_set_ps1(v); }
  fvec4(float a, float b, float c, float d) {
    m_value = _mm_set_ps(a, b, c, d);
  }

  __m128 m128() const { return m_value; }

  friend fvec4 operator+(const fvec4 &a, const fvec4 &b) {
    return _mm_add_ps(a.m128(), b.m128());
  }

  friend fvec4 operator*(const fvec4 &a, const fvec4 &b) {
    return _mm_mul_ps(a.m128(), b.m128());
  }

  friend fvec4 operator-(const fvec4 &a, const fvec4 &b) {
    return _mm_sub_ps(a.m128(), b.m128());
  }

  friend std::ostream &operator<<(std::ostream &stream, const fvec4 &v) {
    char data[128];
    snprintf(data, sizeof(data), "(%.5f, %.5f, %.5f, %.5f)", v.get<0>(),
             v.get<1>(), v.get<2>(), v.get<3>());
    stream << data;
    return stream;
  }

  template <unsigned char Index> float get() const {
    static_assert(Index < 4, "invalid index");
    __m128 shuffled = _mm_shuffle_ps(m_value, m_value, Index);
    return _mm_cvtss_f32(shuffled);
  }
};

int main(int argc, char const *argv[]) {
  fvec4 a{3, 4, 5, 2};
  fvec4 b{7, 8, 2, 3};
  std::cout << "a " << a << "\n";
  std::cout << "b " << b << "\n";
  std::cout << "R " << (a * b) << "\n";
  std::getchar();
  return 0;
}
