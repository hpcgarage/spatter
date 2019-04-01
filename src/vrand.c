/*//////////////////////////////////////////////////////////////////////////////
//
// Implementation from:
// "A linear algorithm for generating random numbers with a given distribution"
// Software Engineering, IEEE Transactions (Volume:17, Issue: 9, pp. 972-975)
// Vose, M.D. ; Dept. of Comput. Sci., Tennessee Univ., Knoxville, TN, USA 
//
///////////////////////////////////////////////////////////////////////////// */


#include <stdio.h>
#include <stdlib.h>
#define _VRAND_C
#include "vrand.h"


////////////////////////////////////////////////////////////


static __thread unsigned int rndx, rtab[55];


////////////////////////////////////////////////////////////


static int nrndm()
{
  unsigned int k;
  
  for (k = 0; k < 24; k++) rtab[k] -= rtab[k+31];
  for (k = 24; k < 55; k++) rtab[k] -= rtab[k-24];
  return 0;
}

static void error_rnd(char *s)
{
  printf("%s\n",s);
  exit(1);
}


////////////////////////////////////////////////////////////


void vrand_init(unsigned int j)
{
  unsigned int h,i,k;
  
  for (rtab[54] = j |= (k = i = 1); i < 55; i++)
    h = (21*i)%55, rtab[--h] = k, k = j - k, j = rtab[h];
  while (i--){
    for (k = 0; k < 24; k++) rtab[k] -= rtab[k+31];
    for (k = 24; k < 55; k++) rtab[k] -= rtab[k-24];
  }
  rndx = 0;
}

dist_t *vrand_dist_alloc(unsigned int n)
{
  dist_t *d;
  
  if (!(d = (dist_t *) malloc(sizeof(dist_t))))
    error_rnd("malloc (allocdist: d)");
  d->n = n;
  if (!(d->a = (int *)malloc(d->n * sizeof(int))))
    error_rnd("malloc (allocdist: d->a)");
  if (!(d->p = (double *)malloc(d->n * sizeof(double))))
    error_rnd("malloc (allocdist: d->p)");
  return d;
}

void vrand_dist_free(dist_t *d)
{
  free(d->a);
  free(d->p);
  free(d);
}

#define TWO_32   (4294967296.0)
#define getsmall { while (p[j] >= q) if ((++j) == stop) goto end; t = j++; }
#define getlarge while (p[k] < q) if ((++k) == stop) goto cleanup;

/* Initialize the distribution d */
dist_t *vrand_dist_init(dist_t *d, double s)
{
  /*
    d->p must have d->n elements which sum to s on entry to initdist.
    d->p and d->a are overwritten by the initialization process.
  */
  int j,k,t,stop,*a; double q,*p;
  
  stop = d->n, q = s/stop, j = k = 0;
  
  d->m1 = stop/TWO_32;
  d->m2 = s/(stop * TWO_32);
  
  a = d->a;
  p = d->p;
  
  getsmall; getlarge;
  
 loop:
  
  a[t] = k;
  p[k] += p[t] - q;
  
  if (p[k] >= q) {
    if (j == stop) goto end;
    getsmall;
    goto loop;
  }
  t = k++;
  if (k == stop) goto cleanup;
  if (j < k) getsmall;
  getlarge;
  goto loop;
  
 cleanup:
  
  a[t] = t;
  while (j < stop) { a[j] = j; j++; }
  
 end:
  return d;
}

#undef getsmall
#undef getlarge

#define rndm() ((++rndx>54)?rtab[rndx=nrndm()]:rtab[rndx])

unsigned int vrand_uint()
{
  return rndm();
}

double vrand_double()
{
  return vrand_uint() / TWO_32;
}

/* Returns element from {0..d->n-1} according to d */
unsigned int vrand_dist(dist_t *d)
{
  unsigned int p0 = vrand_uint();
  unsigned int p1 = vrand_uint();
  unsigned int j  = p0 * d->m1;

  if ( (p1 * d->m2) < d->p[j] ) return j;
  return d->a[j];
}


////////////////////////////////////////////////////////////


#undef rndm
#undef TWO_32


////////////////////////////////////////////////////////////
