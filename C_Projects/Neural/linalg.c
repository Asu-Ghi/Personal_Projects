#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


// scalar multiple of vector w
void scalar_mult(double* w, double scalar, int dim) {
    for (int i = 0; i < dim; i++){
        w[i] *= scalar;
    }
}

// (m x p) * (p x n) = (m x n)
double* matrix_mult(double* inputs, double* weights, int num_inputs, int num_neurons) {
    double* result = (double*)calloc(num_neurons, sizeof(double));
    for (int i = 0; i < num_neurons; i++) {
        for (int j = 0; j < num_inputs; j++) {
            result[i] += inputs[j] * weights[i * num_inputs + j];
        }
    }
    return result;
}

// print matrix
void print_matrix(double* M, int dim1, int dim2) {
    int m = dim1;  // Number of rows
    int n = dim2;  // Number of columns

    // Loop through the rows and columns of the matrix
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", M[i * n + j]);  // Print element at [i, j]
        }
        printf("\n");  // New line after each row
    }
}

// dot product, z, of vectors w and v
double dot_product(double* w, double* v, int dim) {
    double z = 0; 
    for (int i = 0; i < dim; i++){
        z += w[i] * v[i];
    }
    // return scalar dot product of w and v
    return z; 
}

// vector addition,z, of w and v
double* vector_sum(double* w, double* v, int dim) {

    // allocate memory for return vector
    double* z = (double*) calloc(dim, sizeof(double));
    if (z == NULL) {
        fprintf(stderr, "Memory allocation failed in vector_sum.\n");
        exit(1);
    }

    // sum w and v
    for (int i = 0; i < dim; i++) {
        z[i] = w[i] + v[i];
    }
    return z;
}