#ifndef _VRAND_H
#define _VRAND_H


// For generating random number i with probability ->p[i].
typedef struct st_dist
{
  double *p;
  int    *a;
  int     n;
  double  m1;
  double  m2;
} dist_t;


// Interface
#ifndef _VRAND_C
// Basic interface.
extern void         vrand_init  (unsigned int j);
extern unsigned int vrand_uint  ();
extern double       vrand_double();
// Interface for sampling from arbitraty distributions:
// select elements from {0..d->n-1} according to d.
extern dist_t*      vrand_dist_alloc(unsigned int n);
extern dist_t*      vrand_dist_init (dist_t *d, double s);
extern unsigned int vrand_dist      (dist_t *d);
extern void         vrand_dist_free (dist_t *d);
#endif


#endif
