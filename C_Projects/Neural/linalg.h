// linalg.h
#ifndef LINALG_H
#define LINALG_H

void scalar_mult(double* w, double scalar, int dim);

void print_matrix(double* M, int dim1, int dim2);

double* matrix_mult(double* inputs, double* weights, int num_inputs, int num_neurons);

double dot_product(double* w, double* v, int dim);

double* vector_sum(double* w, double* v, int dim);


#endif