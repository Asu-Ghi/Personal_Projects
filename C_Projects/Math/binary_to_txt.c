#include <stdio.h>
#include <stdlib.h>


int main(int* argc, char** argv) {
    if (argc < 3) {
        printf("Command usage %s %s %s \n", argv[0], "size", "dim");
    }

    int len = atoi(argv[1]);
    int dim = atoi(argv[2]);

    // read in binary
    __uint128_t* data = (__uint128_t*) calloc(len * dim, sizeof(__uint128_t));
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < dim; j++) {
            scanf("%u")
        }
    }
    


}
