#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "parse-args.h"

#define STRLEN (1024)

int uniform_stride_test(int indexLength, int stride, int argc, char** argv)
{
    int nrc = 0; 
    struct run_config *rc = NULL;

    parse_args(argc, argv, &nrc, &rc);

    if (nrc != 1)
    {
        printf("Test failure on Uniform Stride: Expected number of runs to be 1, actually was %d.\n", nrc);
        return EXIT_FAILURE;
    }

    if (rc == NULL)
    {
        printf("Test failure on Uniform Stride: failed to create or allocate run_config struct.\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < indexLength; i++)
    {
        if (rc[0].pattern[i] != i * stride)
        {
            printf("Test failure on Uniform Stride with parameters:\nIndex Length: %d\nStride: %d\n", indexLength, stride);
            printf("Element at index %d expected %d, actually was %ld.\n", i, i * stride, rc[0].pattern[i]);
            return EXIT_FAILURE;
        }
    }

    free(rc);
    return EXIT_SUCCESS;
}


int main (int argc, char **argv)
{
    int argc_ = 2;
    char **argv_ = (char**)malloc(sizeof(char*) * (argc_ - 1));
    for (int i = 0; i < argc_; i++) {
        argv_[i] = (char*)malloc(sizeof(char)*STRLEN);
    }
    strcpy(argv_[0], "./spatter");
    
    for (int i = 1; i <= 16; i *= 2)
    {
        for (int j = 8; j <= 64; j *= 2)
        {
            sprintf(argv_[1], "-pUNIFORM:%d:%d", j, i);
            if (uniform_stride_test(j, i, argc_, argv_) != EXIT_SUCCESS)
                return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}
