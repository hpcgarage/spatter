#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "parse-args.h"

#define STRLEN (1024)

int multilevel_test(int multigather) {
    int nrc = 0;
    struct run_config *rc;

    int argc_ = 4;
    char **argv_ = (char**)malloc(sizeof(char*) * argc_);
    for (int i = 0; i < argc_; i++) {
        argv_[i] = (char*)malloc(sizeof(char)*STRLEN);
    }

    strcpy(argv_[0], "./spatter");
    strcpy(argv_[1], "-pUNIFORM:8:1");

    if (multigather) {
        strcpy(argv_[2], "-gUNIFORM:8:1");
        strcpy(argv_[3], "-kMultiGather");
    }
    else {
        strcpy(argv_[2], "-hUNIFORM:8:1");
        strcpy(argv_[3], "-kMultiScatter");
    }

    parse_args(argc_, argv_, &nrc, &rc);

    if (nrc != 1)
        printf("Test failure on Uniform Stride: Expected number of runs to be 1, actually was %d.\n", nrc);

    if (rc == NULL)
        printf("Test failure on Uniform Stride: failed to create or allocate run_config struct.\n");

    int gold[8] = {0,1,2,3,4,5,6,7};

    for (int i = 0; i < 8; i++) {
        if (gold[i] != rc[0].pattern[i]) {
            return -1;
        }
        if (multigather) {
            if (gold[i] != rc[0].pattern_gather[i]) {
                return -1;
            }
        }
        else {
            if (gold[i] != rc[0].pattern_scatter[i]) {
                return -1;
            }
        }
    }

    for (int i = 0; i < argc_; i++) {
        free(argv_[i]);
    }
    free(argv_);

    return 0;
}

int main (int argc, char **argv)
{
    // MultiGather Test
    if (multilevel_test(1) < 0)
        return EXIT_FAILURE;

    // MultScatter Test
    if (multilevel_test(0) < 0)
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
