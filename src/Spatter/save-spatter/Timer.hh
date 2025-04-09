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
  void start();
  void stop();
  double seconds() const;
  void clear();

private:
  bool running = false;
  std::chrono::duration<double> accum = std::chrono::duration<double>(0.0);
  std::chrono::time_point<CLOCK_T> start_tp;
};

inline std::ostream &operator<<(std::ostream &os, const Timer &t);

} // namespace Spatter
#endif
