#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "parse-args.h"

#define STRLEN (1024)

int main (int argc, char **argv)
{
    int nrc = 0;
    struct run_config *rc;

    int argc_ = 4;
    char **argv_ = (char**)malloc(sizeof(char*) * argc_);
    for (int i = 0; i < argc_; i++) {
        argv_[i] = (char*)malloc(sizeof(char)*STRLEN);
    }

    strcpy(argv_[0], "./spatter");
    strcpy(argv_[1], "-hUNIFORM:8:1");
    strcpy(argv_[2], "-gUNIFORM:8:1");
    strcpy(argv_[3], "-kGS");

    parse_args(argc_, argv_, &nrc, &rc);

    if (nrc != 1) {
        printf("Test failure on Uniform Stride: Expected number of runs to be 1, actually was %d.\n", nrc);
        return EXIT_FAILURE;
    }

    if (rc == NULL) {
        printf("Test failure on Uniform Stride: failed to create or allocate run_config struct.\n");
        return EXIT_FAILURE;
    }

    int gold[8] = {0,1,2,3,4,5,6,7};

    for (int i = 0; i < 8; i++) {
        if (gold[i] != rc[0].pattern_scatter[i]) {
            printf("Failure");
            return EXIT_FAILURE;
        }
        if (gold[i] != rc[0].pattern_gather[i]) {
            printf("Failure");
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;

}
