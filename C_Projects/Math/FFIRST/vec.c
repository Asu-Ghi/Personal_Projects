#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vec.h"

// calculate the distance squared between dim dimensional vectors u and v
double vec_dist_sq (double* u, double* v, int dim) {
    double dist_sq = 0;
    for (int i=0;i<dim;i++) {
        dist_sq += (u[i]-v[i])*(u[i]-v[i]);
    }
    return dist_sq;
}

// read len vectors in dim dimensional space from stdin into data array
void vec_read_dataset (double* data, int len, int dim) {
    for (int i=0;i<len;i++) {
        for (int j=0;j<dim;j++) {
            if (scanf("%lf",&(data[i*dim+j])) != 1) {
                printf ("error reading dataset\n");
                exit(1);
            }
        }
    }
}
