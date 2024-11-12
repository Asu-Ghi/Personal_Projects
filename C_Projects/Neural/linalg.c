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
double* matrix_mult(double *m1, double *m2, int* dim1, int* dim2) {

    // Allocate memory for C
    double* m3 = (double*) calloc(dim1[0] * dim2[1], sizeof(double));
    if (m3 == NULL) {
        fprintf(stderr, "Memory allocation failed in matrix mult. \n");
        exit(1);
    }

    // ensures m x n matrix
    if (dim1[1] != dim2[0]) {
        fprintf(stderr, "Wrong matrix multiplication dimensions. \n");
        exit(1);
    }

    int m = dim1[0]; // Rows of m1
    int p = dim1[1]; // Columns of m1 and rows of m2
    int n = dim2[1]; // Columns of m2
    // m x n
    // Matrix multiplication in column-major order
    for (int i = 0; i < m; i++) {        // Row of m1 and m3
        for (int j = 0; j < n; j++) {    // Column of m2 and m3
            double sum = 0.0;
            for (int k = 0; k < p; k++) { // Iterating through shared dimension
                sum += m1[i * p + k] * m2[k * n + j]; // Access in row-major order
            }
            m3[i * n + j] = sum; // Store result in row-major order
        }
    }
    return m3;
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