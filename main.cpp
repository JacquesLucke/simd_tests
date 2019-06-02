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

#include "perlin_noise.hpp"

#define PRINT_EXPR(expression)                                       \
  std::cout << #expression << "\t " << (expression) << "\n"

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
