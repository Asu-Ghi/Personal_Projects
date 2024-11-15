#include "linalg.h"
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

// (rows_w x cols_w) * (rows_v x cols_v) = (rows_w x cols_v)
matrix* matrix_mult(matrix* w, matrix* v) {
    // Get dimensionality info
    int rows_w = w->dim1;
    int cols_w = w->dim2;
    int cols_v = v->dim2;

    // Check dimensions
    if (w->dim2 != v->dim1) {
        fprintf(stderr, "Error in matrix mult, dimensionality mismatch.\n");
        exit(1);
    }

    // Allocate result matrix with dimensions rows_w x cols_v
    matrix* result = malloc(sizeof(matrix));
    result->dim1 = rows_w;
    result->dim2 = cols_v;
    result->data = (double*) calloc(rows_w * cols_v, sizeof(double));
    
    // Check memory allocation
    if (result->data == NULL) {
        fprintf(stderr, "Error: Memory allocation failure in matrix_mult.\n");
        exit(1);
    }

    // Perform the matrix multiplication
    for (int i = 0; i < rows_w; i++) { // For each row in the result
        for (int j = 0; j < cols_v; j++) { // For each column in the result
            for (int k = 0; k < cols_w; k++) { // Shared dimension
                result->data[i * cols_v + j] += w->data[i * cols_w + k] * v->data[k * cols_v + j];
            }
        }
    }

    return result;
}

matrix* transpose_matrix(matrix* w){
    // Create a new matrix object to hold the transposed matrix
    matrix* transposed_matrix = (matrix*) malloc(sizeof(matrix));

    // Check memory allocation for the matrix struct
    if (transposed_matrix == NULL) {
        fprintf(stderr, "Error: Memory allocation failure for transposed_matrix struct.\n");
        exit(1);
    }

    // Allocate memory for the transposed data
    transposed_matrix->dim1 = w->dim2;  // Transposed matrix rows = original matrix cols
    transposed_matrix->dim2 = w->dim1;  // Transposed matrix cols = original matrix rows
    transposed_matrix->data = (double*) calloc(transposed_matrix->dim1 * transposed_matrix->dim2, sizeof(double));

    // Check memory allocation for the transposed data
    if (transposed_matrix->data == NULL) {
        fprintf(stderr, "Error: Memory allocation failure for transposed matrix data.\n");
        exit(1);
    }

    // Iterate through the original matrix and fill the transposed matrix
    for (int i = 0; i < w->dim1; i++) {
        for (int j = 0; j < w->dim2; j++) {
            // Swap row and column indices to transpose the matrix
            transposed_matrix->data[j * w->dim1 + i] = w->data[i * w->dim2 + j];
        }
    }

    // Return the pointer to the transposed matrix
    return transposed_matrix;
}

// print matrix
void print_matrix(matrix* M) {
    int m = M->dim1;  // Number of rows
    int n = M->dim2;  // Number of columns

    // Loop through the rows and columns of the matrix
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", M->data[i * n + j]);  // Print element at [i, j]
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