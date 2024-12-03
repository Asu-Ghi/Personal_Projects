#include "network.h"
#include "test_network.h"

#ifndef NUM_THREADS
#define NUM_THREADS 1
#endif

/*
Find best betas for optimization
*/
void find_best_beta() {
    exit(1);
}

/*
Main Method
*/
int main(int argc, char** argv) {

    #ifdef ENABLE_PARALLEL
        omp_set_num_threads(NUM_THREADS);
        printf("NUMBER OF TOTAL CPU CORES: %d\n", NUM_THREADS);
    #else
        printf("Calculating Sequentially..\n");
    #endif 

    #ifdef STUDY_TIMING
    double start_time = omp_get_wtime();
    #endif
    test_mnist();
    #ifdef STUDY_TIMING
    double end_time = omp_get_wtime();
    printf ("(%d,%.4f),\n",NUM_THREADS,end_time - start_time);
    #endif  
}
