#include "vec.h" // Vector header file
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

typedef struct {
    int indx;
    double dist;
} dist_info;


dist_info calc_cost_sq(double* data, int center_indx, int len, int dim, int k) {

    dist_info* minimums = (dist_info*) malloc(k * sizeof(dist_info));
    int* center_indicies = (int*) malloc(k * sizeof(int));
    if (minimums == NULL || center_indicies == NULL) {
        printf("Error allocating memory for minimums or centers.\n");
        exit(1);
    }

    dist_info max_info = {-1, 0};

    // calculate diff for each ci, excluding itself
    for (int i = 0; i < k; i++) {
        center_indicies[i] = center_indx;
        dist_info* distances = (dist_info*) malloc(len * sizeof(dist_info));
        if (distances == NULL) {
            printf("Error allocating memory for distances.\n");
            exit(1);
        }

        // get distances
        for (int j = 0; j < len; j++) {
            distances[j].dist = vec_dist_sq(&data[center_indicies[i] * dim],
                                         &data[j * dim], dim);
            distances[j].indx = j;
        }

        // find min diff for each ci
        dist_info min_info = {-1, DBL_MAX};

        for (int j = 0; j < len; j++) {
            // skip center
            if(center_indicies[i] == j) {
                continue;
            }

            if (distances[j] < min_info.dist) {
                min_info.indx = distances[j].indx;
                min_info.dist = distances[j].dist;
            }
        }
        free(distances);
        // add minimum to minimums
        minimums[i] = min_info;

        // find maximum value in the min vals
        for (int d = 0; d < k; d++) {
            if (minimums[d].dist >= max_info.dist) {
                max_info.indx = minimums[d].indx;
                max_info.dist = minimums[d].dist;
            }
        }

        // assign next center
        center_indx = max_info.indx;
    }

    free(minimums);
    return(max_info);

}




int main(int argc, char** argv) {
    // Input handling
    if (argc < 2) {
        printf("Command usage %s %s\n",argv[0], "#k centers");
    }

    // get num centers
    int k = atoi(argv[1]);

    // Read in Data -> Prints perfectly 
    int len = 0;
    int dim = 0;
    int result1 = scanf("%d", &len);
    int result2 = scanf("%d", &dim);
    if (result1 != 1 || result2 != 1) {
        printf("Error reading in length and dim of dataset.\n");
        return 1;
    }
    double *data = (double*) calloc(len * dim, sizeof(double));
    vec_read_dataset(data, len, dim);

    // center indicie i.e row index of the vector thats a center
    int center_indx = 10;

    dist_info centers = calc_cost_sq(data, center_indx, len, dim, k);

    return 0;

}