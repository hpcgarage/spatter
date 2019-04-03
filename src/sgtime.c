#include "sgtime.h"
#include <stdio.h>

struct timespec starttime;
struct timespec endtime;

void sg_zero_time(void){
  //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
  clock_gettime(CLOCK_MONOTONIC, &starttime);
}

double diff_ms(void)
{
 unsigned long long diff_ns, diff_s; 
 double diff_ms;
 diff_ns = endtime.tv_nsec - starttime.tv_nsec;
 diff_s = endtime.tv_sec - starttime.tv_sec;
 diff_ms = ((double)diff_s * 1000.0) + ((double)diff_ns / 1000000.0);
 
 //Print out for debugging
 //printf("The difference in ms is %f\n",diff_ms);
 
 return diff_ms;

}


//Returns ms since zero time
double sg_get_time_ms(void){
  //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &s);
  clock_gettime(CLOCK_MONOTONIC, &endtime);

  //Print out for debugging
  //printf("Time is now %llu s and %llu ns for endtime and %llu s and %llu ns for t\n", endtime.tv_sec, endtime.tv_nsec, starttime.tv_sec, starttime.tv_nsec);
  return diff_ms();
}
