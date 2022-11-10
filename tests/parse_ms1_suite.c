#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "parse-args.h"

#define STRLEN (1024)

void print_long_unsigned_int_array(int length, long unsigned int* array)
{
    char output[STRLEN];
    sprintf(output, "[ ");

    int index = 2;

    for (int i = 0; i < length; i++)
    {
        sprintf(&output[index], "%ld ", array[i]);
        index += 2;
    }

    output[index] = ']';
    output[index + 1] = '\0';
    printf("%s\n", output);
}

int ms1_test(int indexLength, int gapLocationCount, int* gapLocations, int gapSizeCount, int* gapSizes, int argc, char** argv)
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

    if (gapLocationCount > 1 && gapSizeCount != gapLocationCount && gapSizeCount != 1)
    {
        printf("Test failure on MS1: specified number of gap locations is inconsistent with number of gap sizes.\n");
        return EXIT_FAILURE;
    }

    int currentValue = -1;
    int gapIndex = 0;
    for (int i = 0; i < indexLength; i++)
    {
        if (gapIndex < gapLocationCount && gapLocations[gapIndex] == i)
        {
            if (gapSizeCount > 1) 
            {
                currentValue += gapSizes[gapIndex];
            }
            else
            {
                currentValue += gapSizes[0];
            }
            gapIndex++;
        }
        else
        {
            currentValue++;
        }
        if ((unsigned long)currentValue != rc[0].pattern[i])
        {
            printf("Test failure on MS1, patterns do not match!\n Got pattern: ");
            print_long_unsigned_int_array(indexLength, rc[0].pattern);
            printf("Got value %ld, expected value %d.\nYou can find the expect patterns in comments in the parse_ms1_suite.c file.\n", rc[0].pattern[i], currentValue);
            return EXIT_FAILURE;
        }
    }

    free(rc);
    return EXIT_SUCCESS;
}

int main ()
{
    int argc_ = 2;
    char **argv_ = (char**)malloc(sizeof(char*) * argc_);
    for (int i = 0; i < argc_; i++) {
        argv_[i] = (char*)malloc(sizeof(char)*STRLEN);
    }
    strcpy(argv_[0], "./spatter");
    
    for (int i = 8; i <= 64; i *= 2)
    {
        // Uniform gap sizes
        // [ 9, 10, 11, 12, 13, 14, 15, 16 ]
        sprintf(argv_[1], "-pMS1:%d:%s:%d", i, "0", 10);
        int gapLocations[1] = {0};
        int gapSizes[1] = {10};
        if (ms1_test(i, 1, gapLocations, 1, gapSizes, argc_, argv_) != EXIT_SUCCESS)
            return EXIT_FAILURE;
        
        // [ 0, 5, 6, 7, 8, 13, 14, 19]
        sprintf(argv_[1], "-pMS1:%d:%s:%d", i, "1, 5, 7", 5);
        int gapLocations1[3] = {1, 5, 7};
        int gapSizes1[1] = {5};
        if (ms1_test(i, 3, gapLocations1, 1, gapSizes1, argc_, argv_) != EXIT_SUCCESS)
            return EXIT_FAILURE;

        // Variable gap sizes
        // [ 4, 5, 14, 15, 16, 17, 18, 19 ]
        sprintf(argv_[1], "-pMS1:%d:%s:%s", i, "0, 2", "5, 9");
        int gapLocations2[2] = {0, 2};
        int gapSizes2[2] = {5, 9};
        if (ms1_test(i, 2, gapLocations2, 2, gapSizes2, argc_, argv_) != EXIT_SUCCESS)
            return EXIT_FAILURE;
        
        // [ 0, 4, 5, 6, 7, 15, 16, 36 ]
        sprintf(argv_[1], "-pMS1:%d:%s:%s", i, "1, 5, 7", "3, 8, 20");
        int gapLocations3[3] = {1, 5, 7};
        int gapSizes3[3] = {3, 8, 20};
        if (ms1_test(i, 3, gapLocations3, 3, gapSizes3, argc_, argv_) != EXIT_SUCCESS)
            return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
