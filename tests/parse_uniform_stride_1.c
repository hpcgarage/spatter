#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "parse-args.h"

#define STRLEN (1024)

int main ()
{
    int nrc = 0;
    struct run_config *rc;

    int argc_ = 2;
    char **argv_ = (char**)malloc(sizeof(char*) * argc_);
    for (int i = 0; i < argc_; i++) {
        argv_[i] = (char*)malloc(sizeof(char)*STRLEN);
    }

    strcpy(argv_[0], "./spatter");
    strcpy(argv_[1], "-pUNIFORM:8:1");

    parse_args(argc_, argv_, &nrc, &rc);

    
    if (nrc != 1)
        printf("Test failure on Uniform Stride: Expected number of runs to be 1, actually was %d.\n", nrc);

    if (rc == NULL)
        printf("Test failure on Uniform Stride: failed to create or allocate run_config struct.\n");

    spIdx_t gold[8] = {0,1,2,3,4,5,6,7};

    for (int i = 0; i < 8; i++) {
        if (gold[i] != rc[0].pattern[i]) {
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;

}
