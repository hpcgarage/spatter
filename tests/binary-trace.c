#include <stdio.h>
#include <stdlib.h>

int gz_read_test() {
    char *command;
    int ret = asprintf(&command, "../gz_read -q -f %s/0.0.R.idx.gz", BINARY_TRACE_DIR);
    if (ret == -1 || system(command) != EXIT_SUCCESS) {
        printf("Test failure on %s", command);
        return EXIT_FAILURE;
    }
    free(command);
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
#ifndef BINARY_TRACE_DIR
    printf("BINARY TRACE Directory not defined!\n");
    return EXIT_FAILURE;
#else
    if (gz_read_test() != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
#endif
}
