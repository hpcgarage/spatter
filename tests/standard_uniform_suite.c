#include <stdlib.h>

int uniform_test_length_gap() {
    int length = 4, gap = 2;
    for (int i = 0; i < 10; i++) {
        const char *command = sprintf("./spatter -pUNIFORM:%d:%d", length, gap);
        if (system(command) == EXIT_FAILURE) {
            printf("Test failure on %s", command);
            return EXIT_FAILURE;
        }
        length *= 2;
        gap *= 2;
    }
    return EXIT_SUCCESS;
}

int uniform_test_delta() {
    for (int delta = 1; delta < 100; delta*=2) {
        const char *command = sprintf("./spatter -pUNIFORM:8:4 -d%d", delta);
        if (system(command) == EXIT_FAILURE) {
            printf("Test failure on %s", command);
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    if (uniform_test_length_gap() != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    if (uniform_test_delta() != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}