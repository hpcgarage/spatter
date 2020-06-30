#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "parse-args.h"

#define STRLEN (1024)

int json_test(int runCount, char* kernel, int* strides, int* patternLengths, int** patterns, int* counts, int argc, char** argv)
{
    int nrc = 0; 
    struct run_config *rc = NULL;

    parse_args(argc, argv, &nrc, &rc);

    if (nrc != runCount)
    {
        printf("Test failure on JSON Parse: Expected number of runs to be %d, actually was %d.\n", runCount, nrc);
        return EXIT_FAILURE;
    }

    if (rc == NULL)
    {
        printf("Test failure on JSON Parse: failed to create or allocate run_config struct.\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < runCount; i++)
    {
        if (rc[i].pattern_len != patternLengths[i])
        {
            printf("Test failure on JSON Parse: pattern length for run_config %d was %ld, expected %d.\n", i, rc[i].pattern_len, patternLengths[i]);
            return EXIT_FAILURE;
        }

        if (rc[i].generic_len != counts[i])
        {
            printf("Test failure on JSON Parse: counts for run_config %d was %ld, expected %d.\n", i, rc[i].generic_len, counts[i]);
            return EXIT_FAILURE;
        }

        if (!strcmp(kernel, "GATHER") && rc[i].kernel != 2)
        {
            printf("Test failure on JSON Parse: user requested kernel GATHER but instead got other kernel %d.\n", rc[i].kernel);
            return EXIT_FAILURE;
        }

        if (!strcmp(kernel, "SCATTER") && rc[i].kernel != 1)
        {
            printf("Test failure on JSON Parse: user requested kernel SCATTER but instead got other kernel %d.\n", rc[i].kernel);
            return EXIT_FAILURE;
        }

        if (!strcmp(kernel, "SG") && rc[i].kernel != 3)
        {
            printf("Test failure on JSON Parse: user requested kernel SG but instead got other kernel %d.\n", rc[i].kernel);
            return EXIT_FAILURE;
        }

        for (int j = 0; j < patternLengths[i]; j++)
        {
            if (patterns[i][j] != rc[i].pattern[j])
            {
                printf("Test failure on JSON Parse: pattern mismatch at index %d, got value %ld but expected %d.\n", j, rc[i].pattern[j], patterns[i][j]);
                return EXIT_FAILURE;
            }
        }
    }

    free(rc);
    return EXIT_SUCCESS;
}


int main (int argc, char **argv)
{
    #ifndef JSON_SRC
    printf("JSON SRC Directory not defined!\n");
    return EXIT_FAILURE;
    #else
    int argc_ = 2;
    char **argv_ = (char**)malloc(sizeof(char*) * (argc_ - 1));
    for (int i = 0; i < argc_; i++) {
        argv_[i] = (char*)malloc(sizeof(char)*STRLEN);
    }
    strcpy(argv_[0], "./spatter");
    sprintf(argv_[1], "-pFILE=%s", JSON_SRC);

    int strides[2] = {1, 1};
    int patternLengths[2] = {16, 16};
    int pattern1[16] = {1333, 0, 1, 2, 36, 37, 38, 72, 73, 74, 1296, 1297, 1298, 1332, 1334, 1368};
    int pattern2[16] = {1333, 0, 1, 36, 37, 72, 73, 1296, 1297, 1332, 1368, 1369, 2592, 2593, 2628, 2629};
    int* patterns[2] = {pattern1, pattern2};
    int counts[2] = {1454647, 1454647};
    if (json_test(2, "GATHER", strides, patternLengths, patterns, counts, argc_, argv_) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
    #endif
}
