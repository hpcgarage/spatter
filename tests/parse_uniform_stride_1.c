#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "parse-args.h"

#define STRLEN (1024)

int main (int argc, char **argv)
{
    int nrc = 0;
    struct run_config *rc;

    int argc_ = 2;
    char **argv_ = (char**)malloc(sizeof(char*) * (argc_ - 1));
    for (int i = 0; i < argc_; i++) {
        argv_[i] = (char*)malloc(sizeof(char)*STRLEN);
    }

    strcpy(argv_[0], "./spatter");
    strcpy(argv_[1], "-pUNIFORM:8:1");
    argv[2] = NULL;

    parse_args(argc_, argv_, &nrc, &rc);

    int gold[8] = {0,1,2,3,4,5,6,7};

    for (int i = 0; i < 8; i++) {
        if (gold[i] != rc[0].pattern[i]) {
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;

}
