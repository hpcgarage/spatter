#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "parse-args.h"

#define STRLEN (1024)

int k_tests(int argc_, char** argv_, int* nrc, struct run_config* rc)
{
    asprintf(&argv_[2], "-kGATHER");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].kernel != 2)
    {
        printf("Test failure on run_config suite: POSIX-style k with argument GATHER resulted in kernel %d.\n", rc[0].kernel);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "-k GATHER");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].kernel != 2)
    {
        printf("Test failure on run_config suite: k with argument GATHER resulted in kernel %d.\n", rc[0].kernel);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "-kSCATTER");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].kernel != 1)
    {
        printf("Test failure on run_config suite: POSIX-style k with argument SCATTER resulted in kernel %d.\n", rc[0].kernel);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "-k SCATTER");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].kernel != 1)
    {
        printf("Test failure on run_config suite: k with argument SCATTER resulted in kernel %d.\n", rc[0].kernel);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "-kSG");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].kernel != 3)
    {
        printf("Test failure on run_config suite: POSIX-style k with argument SG resulted in kernel %d.\n", rc[0].kernel);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "-k SG");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].kernel != 3)
    {
        printf("Test failure on run_config suite: k with argument SG resulted in kernel %d.\n", rc[0].kernel);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "--kernel-name=GATHER");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].kernel != 2)
    {
        printf("Test failure on run_config suite: --kernel-name with argument GATHER resulted in kernel %d.\n", rc[0].kernel);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "--kernel-name=SCATTER");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].kernel != 1)
    {
        printf("Test failure on run_config suite: --kernel-name with argument SCATTER resulted in kernel %d.\n", rc[0].kernel);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "--kernel-name=SG");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].kernel != 3)
    {
        printf("Test failure on run_config suite: --kernel-name with argument SG resulted in kernel %d.\n", rc[0].kernel);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int d_tests(int argc_, char** argv_, int* nrc, struct run_config* rc)
{
    asprintf(&argv_[2], "-d9");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].deltas_len != 1)
    {
        printf("Test failure on run_config suite: -d with argument 9 had incorrect length of %zu.\n", rc[0].deltas_len);
        return EXIT_FAILURE;
    }
    if (rc[0].deltas[0] != 9)
    {
        printf("Test failure on run_config suite: -d with argument 9 had incorrect element of %zu.\n", rc[0].deltas[0]);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "--delta=9,7");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].deltas_len != 2)
    {
        printf("Test failure on run_config suite: -d with argument 9,7 had incorrect length of %zu.\n", rc[0].deltas_len);
        return EXIT_FAILURE;
    }
    if (rc[0].deltas[0] != 9)
    {
        printf("Test failure on run_config suite: -d with argument 9,7 had incorrect element of %zu.\n", rc[0].deltas[0]);
        return EXIT_FAILURE;
    }
    if (rc[0].deltas[1] != 7)
    {
        printf("Test failure on run_config suite: -d with argument 9,7 had incorrect element of %zu.\n", rc[0].deltas[1]);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int l_tests(int argc_, char** argv_, int* nrc, struct run_config* rc)
{
    asprintf(&argv_[2], "-l100");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].generic_len != 100)
    {
        printf("Test failure on run_config suite: -l with argument 100 had incorrect value of %lu.\n", rc[0].generic_len);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "-l 500");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].generic_len != 500)
    {
        printf("Test failure on run_config suite: -l with argument 500 had incorrect value of %lu.\n", rc[0].generic_len);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "--count=1000");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].generic_len != 1000)
    {
        printf("Test failure on run_config suite: -l with argument 1000 had incorrect value of %lu.\n", rc[0].generic_len);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int w_tests(int argc_, char** argv_, int* nrc, struct run_config* rc)
{
    asprintf(&argv_[2], "-w100");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].wrap != 100)
    {
        printf("Test failure on run_config suite: -w with argument 100 had incorrect value of %zu.\n", rc[0].wrap);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "-w 500");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].wrap != 500)
    {
        printf("Test failure on run_config suite: -w with argument 500 had incorrect value of %zu.\n", rc[0].wrap);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "--wrap=1000");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].wrap != 1000)
    {
        printf("Test failure on run_config suite: -w with argument 1000 had incorrect value of %zu.\n", rc[0].wrap);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int v_tests(int argc_, char** argv_, int* nrc, struct run_config* rc)
{
    asprintf(&argv_[2], "-v100");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].vector_len != 100)
    {
        printf("Test failure on run_config suite: -v with argument 100 had incorrect value of %zu.\n", rc[0].vector_len);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "-v 500");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].vector_len != 500)
    {
        printf("Test failure on run_config suite: -v with argument 500 had incorrect value of %zu.\n", rc[0].vector_len);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "--vector-len=1000");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].vector_len != 1000)
    {
        printf("Test failure on run_config suite: -v with argument 1000 had incorrect value of %zu.\n", rc[0].vector_len);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int R_tests(int argc_, char** argv_, int* nrc, struct run_config* rc)
{
    asprintf(&argv_[2], "-R100");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].nruns != 100)
    {
        printf("Test failure on run_config suite: -R with argument 100 had incorrect value of %d.\n", rc[0].nruns);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "-R 500");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].nruns != 500)
    {
        printf("Test failure on run_config suite: -R with argument 500 had incorrect value of %d.\n", rc[0].nruns);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "--runs=1000");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].nruns != 1000)
    {
        printf("Test failure on run_config suite: -R with argument 1000 had incorrect value of %d.\n", rc[0].nruns);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int o_tests(int argc_, char** argv_, int* nrc, struct run_config* rc)
{
    asprintf(&argv_[2], "-oCOPY");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].op != 0)
    {
        printf("Test failure on run_config suite: -o with argument COPY had incorrect value of %d.\n", rc[0].op);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "-o ACCUM");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].op != 1)
    {
        printf("Test failure on run_config suite: -o with argument ACCUM had incorrect value of %d.\n", rc[0].op);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "--op=COPY");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].op != 0)
    {
        printf("Test failure on run_config suite: -o with argument COPY had incorrect value of %d.\n", rc[0].op);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int z_tests(int argc_, char** argv_, int* nrc, struct run_config* rc)
{
    asprintf(&argv_[2], "-z100");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].local_work_size != 100)
    {
        printf("Test failure on run_config suite: -z with argument 100 had incorrect value of %zu.\n", rc[0].local_work_size);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "-z 500");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].local_work_size != 500)
    {
        printf("Test failure on run_config suite: -z with argument 500 had incorrect value of %zu.\n", rc[0].local_work_size);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "--local-work-size=1000");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].local_work_size != 1000)
    {
        printf("Test failure on run_config suite: -z with argument 1000 had incorrect value of %zu.\n", rc[0].local_work_size);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int m_tests(int argc_, char** argv_, int* nrc, struct run_config* rc)
{
    asprintf(&argv_[2], "-m100");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].shmem != 100)
    {
        printf("Test failure on run_config suite: -m with argument 100 had incorrect value of %u.\n", rc[0].shmem);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "-m 500");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].shmem != 500)
    {
        printf("Test failure on run_config suite: -m with argument 500 had incorrect value of %u.\n", rc[0].shmem);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "--shared-mem=1000");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].shmem != 1000)
    {
        printf("Test failure on run_config suite: -m with argument 1000 had incorrect value of %u.\n", rc[0].shmem);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int n_tests(int argc_, char** argv_, int* nrc, struct run_config* rc)
{
    asprintf(&argv_[2], "-nTestName");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (strcmp(rc[0].name, "TestName"))
    {
        printf("Test failure on run_config suite: -n with argument TestName had incorrect value of %s.\n", rc[0].name);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "-n TestName2");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (strcmp(rc[0].name, "TestName2"))
    {
        printf("Test failure on run_config suite: -n with argument TestName2 had incorrect value of %s.\n", rc[0].name);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "--name=TestName3");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (strcmp(rc[0].name, "TestName3"))
    {
        printf("Test failure on run_config suite: -n with argument TestName3 had incorrect value of %s.\n", rc[0].name);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int other_tests(int argc_, char** argv_, int* nrc, struct run_config* rc)
{
    asprintf(&argv_[2], "--morton=1");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].ro_morton != 1)
    {
        printf("Test failure on run_config suite: --morton with argument 1 had incorrect value of %d.\n", rc[0].ro_morton);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "--hilbert=1");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].ro_hilbert != 1)
    {
        printf("Test failure on run_config suite: --hilbert with argument 1 had incorrect value of %d.\n", rc[0].ro_hilbert);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "--roblock=1");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].ro_block != 1)
    {
        printf("Test failure on run_config suite: --roblock with argument 1 had incorrect value of %d.\n", rc[0].ro_block);
        return EXIT_FAILURE;
    }

    asprintf(&argv_[2], "--stride=1");
    parse_args(argc_, argv_, nrc, &rc);
    free(argv_[2]);

    if (rc[0].stride_kernel != 1)
    {
        printf("Test failure on run_config suite: --stride with argument 1 had incorrect value of %d.\n", rc[0].stride_kernel);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int main ()
{
    int argc_ = 3;
    char **argv_ = (char**)malloc(sizeof(char*) * argc_);

    int ret;
    ret = asprintf(&argv_[0], "./spatter");
    if (ret == -1)
        return EXIT_FAILURE;

    ret = asprintf(&argv_[1], "-p1,2,3,4");
    if (ret == -1)
        return EXIT_FAILURE;

    int nrc = 0;
    struct run_config *rc = NULL;

    // kernel-name k
    if (k_tests(argc_, argv_, &nrc, rc) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // delta d
    if (d_tests(argc_, argv_, &nrc, rc) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // count l
    if (l_tests(argc_, argv_, &nrc, rc) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // wrap w
    if (w_tests(argc_, argv_, &nrc, rc) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // vector-len v
    if (v_tests(argc_, argv_, &nrc, rc) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // runs R
    if (R_tests(argc_, argv_, &nrc, rc) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // op o
    if (o_tests(argc_, argv_, &nrc, rc) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // local-work-size z
    if (z_tests(argc_, argv_, &nrc, rc) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // shared-mem m
    if (m_tests(argc_, argv_, &nrc, rc) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // name n
    if (n_tests(argc_, argv_, &nrc, rc) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // morton, hilbert, roblock, and stride
    if (other_tests(argc_, argv_, &nrc, rc) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    free(argv_[0]);
    free(argv_[1]);
    free(argv_);

    free(rc);
    return EXIT_SUCCESS;
}
