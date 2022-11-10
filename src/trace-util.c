#define  _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include "sp_alloc.h"
#include "trace-util.h"

//TODO: Remove. No longer used.
#if 0
void read_trace (struct trace *t, const char *filename)
{
    FILE *fp = fopen(filename, "r");
    assert(fp);

    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    char *token;
    double cpct_accum = 0;

    int have_num_instructions = 0;
    int instructions_read = 0;

    while ((read = getline(&line, &len, fp)) != -1) {
        if (line[0] == '#')
            continue;

        if (have_num_instructions && instructions_read < t->length) {
            struct instruction tmp;

            token = strtok(line, " ");
            sscanf(token, "%d", &(tmp.type));

            token = strtok(NULL, " ");
            sscanf(token, "%zu", &(tmp.data_type_size));

            token = strtok(NULL, " ");
            sscanf(token, "%zu", &(tmp.count));

            token = strtok(NULL, " ");
            sscanf(token, "%lf", &(tmp.pct));

            cpct_accum += tmp.pct;
            tmp.cpct = cpct_accum;

            token = strtok(NULL, " ");
            sscanf(token, "%zu", &(tmp.length));

            tmp.delta = (sgsIdx_t *)malloc(tmp.length * sizeof(sgsIdx_t));

            for (int i = 0; i < tmp.length; i++) {
                token = strtok(NULL, " ");
                sscanf(token, SGS, &(tmp.delta[i]));
            }
            t->in[instructions_read] = tmp;
            instructions_read++;
        }
        else {
            sscanf(line, "%zu", &(t->length));
            t->in = (struct instruction *)malloc(t->length *sizeof(struct instruction));
            have_num_instructions = 1;
        }
    }
    fclose(fp);

}

void print_trace(struct trace t) {
    printf("%zu\n", t.length);
    for (int i = 0; i < t.length; i++) {
        struct instruction tmp = t.in[i];
        printf("%d %zu %zu %lf %zu ", tmp.type, tmp.data_type_size, tmp.count, tmp.pct, tmp.length);
        for (int j = 0; j < tmp.length; j++) {
            printf(SGS, tmp.delta[j]);
            if (j != tmp.length-1) {
                printf(" ");
            }
        }
        printf("\n");
    }
}

void reweight_trace(struct trace t){
    //rescale weights to be between 0 and 1
    double tot = 0;
    double cpct_accum = 0;
    for (int i = 0; i < t.length; i++) {
        tot += t.in[i].pct;
    }
    for (int i = 0; i < t.length; i++) {
        t.in[i].pct /= tot;
        cpct_accum += t.in[i].pct;
        t.in[i].cpct = cpct_accum;
    }
}
#endif
