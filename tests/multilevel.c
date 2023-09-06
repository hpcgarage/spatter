#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>

int uniform_multilevel_test() {
    int len1 = 8;
    int len2 = 8;
    for (int i = 1; i <= 8; i++) {
        char *command;
        int ret = asprintf(&command, "../spatter -kMultiGather -pUNIFORM:%d:%d -gUNIFORM:%d:%d ", len1, i, len2, 1);
        if (ret == -1 || system(command) != EXIT_SUCCESS) {
            printf("Test failure on %s", command);
            return EXIT_FAILURE;
        }
        len1 *= 2;
        //len2 = len2*(i % 2 == 0 ? 2 : 1);
        free(command);
    }
    len1 = 8;
    len2 = 8;
    for (int i = 1; i <= 8; i++) {
        char *command;
        int ret = asprintf(&command, "../spatter -kMultiScatter -pUNIFORM:%d:%d -hUNIFORM:%d:%d ", len1, i, len2, 1);
        if (ret == -1 || system(command) != EXIT_SUCCESS) {
            printf("Test failure on %s", command);
            return EXIT_FAILURE;
        }
        free(command);
        len1 *= 2;
        //len2 = len2*(i % 2 == 0 ? 2 : 1);
    }
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    if (uniform_multilevel_test() != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
