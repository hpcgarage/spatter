/*!
  \file Ume/Timer.hh
*/

#ifndef UME_TIMER_HH
#define UME_TIMER_HH 1

#include <cassert>
#include <chrono>
#include <ostream>

namespace Spatter {

//! A simple elapsed-time class
class Timer {
  typedef std::chrono::system_clock CLOCK_T;

public:
  void start() {
    assert(!running);
    running = true;
    start_tp = CLOCK_T::now();
  }
  void stop() {
    std::chrono::time_point<CLOCK_T> stop_tp = CLOCK_T::now();
    assert(running);
    running = false;
    accum += stop_tp - start_tp;
  }
  double seconds() const { return accum.count(); }
  void clear() {
    running = false;
    accum = std::chrono::duration<double>(0.0);
  }

private:
  bool running = false;
  std::chrono::duration<double> accum = std::chrono::duration<double>(0.0);
  std::chrono::time_point<CLOCK_T> start_tp;
};

inline std::ostream &operator<<(std::ostream &os, const Timer &t) {
  return os << t.seconds() << 's';
}

} // namespace Spatter
#endif
