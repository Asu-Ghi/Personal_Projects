/*
Asu Ghimire
11/17/2024

Utility "Class".
Provides basic utilites 
Handles 
    > Linear algebra functionalities
    > Data loading 
*/
#ifndef UTILS_H
#define UTILS_H

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h> 
#include <stdbool.h>
#include "cJSON.h" // Saving parameters
#include <unistd.h> // Socket work
#include <arpa/inet.h> // Socket work

//////////////////////////////////////////////////// Data Structures //////////////////////////////////////////////////////////////
/*
Matrix data structure 
holds dimensional data
holds data as a pointer to a 1d double array
*/
typedef struct {
    double* data;
    int dim1;
    int dim2;
} matrix;

/*
Optimization function enum structure
Enum to store what optimization function to use.
*/
typedef enum {
    SGD,
    SGD_MOMENTUM,
    ADA_GRAD,
    RMS_PROP,
    ADAM
}OptimizationType;

/*
Activation function enum structure
Enum to store what activation function is being used
*/
typedef enum {
    RELU,
    SOFTMAX,
    SIGMOID,
    LINEAR,
    TANH
} ActivationType;

/////////////////////////////////////////////////////// Misc. Methods /////////////////////////////////////////////////////////////////

/*
Loads IRIS Data in for training
Initializes X_train, Y_train, X_test, and Y_test.
Class Labels (One-Hot)
    > 0 : Iris-Setosa
    > 1 : Iris-versicolor
    > 2: Iris-virginica
*/
void load_iris_data(char* file_path, matrix* X_train, matrix* Y_train, matrix* X_test, 
                    matrix* Y_test, int num_batches, double train_ratio);

/*
Loads Spiral Data set
*/
void load_spiral_data(const char *filename, double *data, int start_row, int end_row, int cols);

/*
Loads Mnist Data
*/
void load_mnist_data(char* filename, double* X, double* Y, int num_samples);


/*
Convert Optimization enum to a string
*/
char* optimization_type_to_string(OptimizationType type);


/*
Convert Activation Enum to a String
*/
char* activation_type_to_string(ActivationType type);

/*
Allocates memory on the heap for a matrix object.
Checks memory allocation.
*/
matrix* allocate_matrix(int dim1, int dim2);

/*
Frees matrix struct. Checks for dangling pointers.
*/
void free_matrix(matrix* M);

/*
Shallow copies a select portion of the matrix src
*/
void shallow_cpy_matrix(matrix* src, matrix* dest, int start_row, int num_rows);

//////////////////////////////////////////////////// Linear Algebra Methods //////////////////////////////////////////////////////////////

/*
Returns a matrix object. Allocates memory on the heap for the object.
Transposes w, swaps dimension indicators in the matrix object.
*/
matrix* transpose_matrix(matrix* w); 

/*
Prints a matrix object. Dimensional information required to print is included in the object.
*/
void print_matrix(matrix* M);

/*
Returns a matrix object. Allocates memory on the heap for the object.
Returns the matrix product of w x v. Order matters.
Includes dimensionality checks.
*/
matrix* matrix_mult(matrix* w, matrix* v);


/*
Returns a matrix object. Allocates memory on the heap for the object.
Returns element by element matrix product of w x v. Order doesnt matter.
Includes dimensionality checks.
*/
matrix* element_matrix_mult(matrix* w, matrix* v);

/*
Returns a scalar for the dot product of w and v
Includes dimensionality checks.
*/
double vector_dot_product(matrix* w, matrix* v);

/*
Scales a matrix w by a scalar s.
*/
void matrix_scalar_mult(matrix* w, double s);


/*
Returns matrix sum of w and v
Includes dimensionality checks.
Allocates memory on the heap for the return matrix
*/
matrix* matrix_sum(matrix* w, matrix* v);

/*
Returns a matrix of scalar sum between matrix
w and scalar s. 
Allocates memory on the heap for the return matrix
*/
matrix* matrix_scalar_sum(matrix* w, double s, bool useAbs);


#endif