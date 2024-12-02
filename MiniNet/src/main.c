#include "network.h"
#include "test_network.h"

#define NUM_THREADS 8

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
    omp_set_num_threads(NUM_THREADS); // Set the number of threads to 8
    #endif 
    
    test_mnist();
}
