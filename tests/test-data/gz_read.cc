#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

// compile with -lz

#define NBUFS (1 << 18) // gzfile read buffer

int gz_buf_read(gzFile fp, uint64_t *buf, uint64_t **pbuf, int *edx) {

  int idx;

  idx = (*edx) / sizeof(uint64_t);
  // first read
  if (*pbuf == NULL) {
    *edx = gzread(fp, buf, sizeof(uint64_t) * NBUFS);
    *pbuf = buf;

  } else if (*pbuf == &buf[idx]) {
    *edx = gzread(fp, buf, sizeof(uint64_t) * NBUFS);
    *pbuf = buf;
  }

  if (*edx == 0)
    return 0;

  return 1;
}

///* EXAMPLE USE
int main(int argc, char **argv) {
  if (argc <= 1) {
    printf("Usage: ./gz_read <file>\n");
    exit(1);
  }

  char fname[256];

  strncpy(fname, argv[1], 256);

  int izret = 0;
  uint64_t *pzbuff = NULL;
  uint64_t zbuff[NBUFS];
  gzFile zfp = NULL;

  zfp = gzopen(fname, "hrb");
  if (zfp == NULL) {
    printf("ERROR: Could not open %s!\n", fname);
    exit(-1);
  }

  while (gz_buf_read(zfp, zbuff, &pzbuff, &izret)) {

    printf("%d\n", (*pzbuff));

    pzbuff++;
  }

  gzclose(zfp);

  return 0;
}
