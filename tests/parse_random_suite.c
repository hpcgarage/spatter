#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "parse-args.h"
#include <time.h>

#define STRLEN (1024)

int random_test(int seed, int argc, char** argv)
{
    int nrc = 0; 
    struct run_config *rc = NULL;

    parse_args(argc, argv, &nrc, &rc);

    if (nrc != 1)
    {
        printf("Test failure on random argument: Expected number of runs to be 1, actually was %d.\n", nrc);
        return EXIT_FAILURE;
    }

    if (rc == NULL)
    {
        printf("Test failure on random argument: failed to create or allocate run_config struct.\n");
        return EXIT_FAILURE;
    }

    // Be generous and give the test a relatively large margin of error all things considered
    if (seed < 0 && time(NULL) - rc[0].random_seed > 5)
    {
        printf("Test failure on random argument: expected a random seed approximately near %ld but instead got %zu.\n", time(NULL), rc[0].random_seed);
        return EXIT_FAILURE;
    }
    else if (seed >= 0 && rc[0].random_seed != seed)
    {
        printf("Test failure on random argument: expected a random seed of %d but got a seed of %zu.\n", seed, rc[0].random_seed);
        return EXIT_FAILURE;
    }

    free(rc);
    return EXIT_SUCCESS;
}


int main (int argc, char **argv)
{
    int argc_ = 3;
    char **argv_ = (char**)malloc(sizeof(char*) * (argc_ - 1));
    for (int i = 0; i < argc_; i++) {
        argv_[i] = (char*)malloc(sizeof(char)*STRLEN);
    }
    strcpy(argv_[0], "./spatter");
    sprintf(argv_[1], "-p1,2,3,4");

    sprintf(argv_[2], "-s");
    if (random_test(-1, argc_, argv_) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    sprintf(argv_[2], "-s123");
    if (random_test(123, argc_, argv_) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    sprintf(argv_[2], "-s 456");
    if (random_test(456, argc_, argv_) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    sprintf(argv_[2], "--random");
    if (random_test(-1, argc_, argv_) != EXIT_SUCCESS)
        return EXIT_FAILURE;

        sprintf(argv_[2], "--random=789");
    if (random_test(789, argc_, argv_) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
