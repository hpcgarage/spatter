#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "parse-args.h"
#include "unused.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

#define STRLEN (1024)

int omp_thread_test(int thread_count, int argc, char** argv)
{
#ifndef USE_OPENMP
    UNUSED_VAR(thread_count);
    UNUSED_VAR(argc);
    UNUSED_VAR(argv);
    return EXIT_SUCCESS;
#else

    int nrc = 0;
    struct run_config *rc = NULL;

    parse_args(argc, argv, &nrc, &rc);

    if (nrc != 1)
    {
        printf("Test failure on OMP Threads: Expected number of runs to be 1, actually was %d.\n", nrc);
        return EXIT_FAILURE;
    }

    if (rc == NULL)
    {
        printf("Test failure on OMP Threads: failed to create or allocate run_config struct.\n");
        return EXIT_FAILURE;
    }

    if (thread_count < 0 && rc[0].omp_threads < 0)
    {
        printf("Test failure on OMP Threads: user requested negative thread count and request was granted.\n");
        return EXIT_FAILURE;
    }

    if (thread_count == 0 && rc[0].omp_threads != (unsigned int)omp_get_max_threads())
    {
        printf("Test failure on OMP Threads: requested 0 threads and should have defaulted to OMP Max Thread Count, but instead allocated %ld threads.\n", rc[0].omp_threads);
        return EXIT_FAILURE;
    }

    if (thread_count > 0 && thread_count != 0 && rc[0].omp_threads != (unsigned int)thread_count && thread_count < omp_get_max_threads())
    {
        printf("Test failure on OMP Threads: requested %d threads but got %ld threads instead.\n)", thread_count, rc[0].omp_threads);
        return EXIT_FAILURE;
    }

    free(rc);
    return EXIT_SUCCESS;
    #endif
}


int main ()
{
    int argc_ = 3;
    char **argv_ = (char**)malloc(sizeof(char*) * argc_);
    for (int i = 0; i < argc_; i++) {
        argv_[i] = (char*)malloc(sizeof(char)*STRLEN);
    }
    strcpy(argv_[0], "./spatter");
    strcpy(argv_[1], "-p1,2,3,4");

    sprintf(argv_[2], "-t4");
    if (omp_thread_test(4, argc_, argv_) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    sprintf(argv_[2], "-t 0");
    if (omp_thread_test(0, argc_, argv_) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    sprintf(argv_[2], "-t10000000");
    if (omp_thread_test(10000000, argc_, argv_) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    sprintf(argv_[2], "-t-1");
    if (omp_thread_test(-1, argc_, argv_) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    sprintf(argv_[2], "--omp-threads=0");
    if (omp_thread_test(0, argc_, argv_) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    argc_ = 2;
    if (omp_thread_test(0, argc_, argv_) != EXIT_SUCCESS)
        return EXIT_FAILURE;


    return EXIT_SUCCESS;
}
