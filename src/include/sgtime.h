#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#ifndef SGTIME_H
#define SGTIME_H

#include <time.h>

void   sg_zero_time(void);
double sg_get_time_ms(void);

#endif
