// linalg.h
#ifndef LINALG_H
#define LINALG_H

/*
matrix data structure 
holds dimensional data
holds data as a pointer to a 1d double array
used to store outputs and inputs.
*/
typedef struct {
    double* data;
    int dim1;
    int dim2;
} matrix;

void scalar_mult(double* w, double scalar, int dim);

matrix* transpose_matrix(matrix* w); 

void print_matrix(double* M, int dim1, int dim2);

matrix* matrix_mult(matrix* w, matrix* v);

double dot_product(double* w, double* v, int dim);

double* vector_sum(double* w, double* v, int dim);


#endif