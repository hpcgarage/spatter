#include <stdio.h>
#include <stdlib.h>

int uniform_multilevel_test() {
    for (int i = 1; i <= 10; i++) {
        char *command;
        int ret = asprintf(&command, "../spatter -kMultiGather -pUNIFORM:%d:%d -gUNIFORM:%d:%d ", i, i, i, i);
        if (ret == -1 || system(command) != EXIT_SUCCESS) {
            printf("Test failure on %s", command);
            return EXIT_FAILURE;
        }
        free(command);
    }
    for (int i = 1; i <= 10; i++) {
        char *command;
        int ret = asprintf(&command, "../spatter -kMultiScatter -pUNIFORM:%d:%d -hUNIFORM:%d:%d ", i, i, i, i);
        if (ret == -1 || system(command) != EXIT_SUCCESS) {
            printf("Test failure on %s", command);
            return EXIT_FAILURE;
        }
        free(command);
    }
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    if (uniform_multilevel_test() != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
