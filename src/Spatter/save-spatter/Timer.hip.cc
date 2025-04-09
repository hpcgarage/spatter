/*!
  \file Ume/Timer.cc
*/

#include "Timer.hh"

namespace Spatter {

//! A simple elapsed-time class
void Timer::start() {
  assert(!running);
  running = true;
  start_tp = CLOCK_T::now();
}

void Timer::stop() {
  std::chrono::time_point<CLOCK_T> stop_tp = CLOCK_T::now();
  assert(running);
  running = false;
  accum += stop_tp - start_tp;
}

double Timer::seconds() const { return accum.count(); }

void Timer::clear() {
  running = false;
  accum = std::chrono::duration<double>(0.0);
}

inline std::ostream &operator<<(std::ostream &os, const Timer &t) {
  return os << t.seconds() << 's';
}

} // namespace Spatter
