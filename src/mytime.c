#include "mytime.h"

struct timespec t;

void zero_time(void){
  //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
  clock_gettime(CLOCK_MONOTONIC, &t);
}

//Returns ms since zero time
double get_time(void){
  struct timespec s;
  //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &s);
  clock_gettime(CLOCK_MONOTONIC, &s);

  return (s.tv_sec - t.tv_sec)   * 1e3    +
         (s.tv_nsec - t.tv_nsec) / 1e6;

}
