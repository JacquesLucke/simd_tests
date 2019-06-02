#pragma once

#include <chrono>
#include <iostream>

class ScopedTimer {
 private:
  using Clock = std::chrono::high_resolution_clock;
  using Duration = std::chrono::duration<float>;

  const char *m_name;
  Clock::time_point m_start;

 public:
  ScopedTimer(const char *name = "") {
    m_name = name;
    m_start = Clock::now();
  }

  ~ScopedTimer() {
    Clock::time_point end = Clock::now();
    Duration duration = end - m_start;
    double ms = duration.count() * 1000.0f;
    std::cout << "Timer '" << m_name << "' took " << ms << " ms\n";
  }
};

#define SCOPED_TIMER(name) ScopedTimer t(name);
