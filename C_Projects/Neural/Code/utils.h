/*
Asu Ghimire
11/17/2024

Utility "Class".
Provides basic utilites for forward.c, backward.c, network.c
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
    TANH
} ActivationType;

/*
Class label encoding enum structure
Enum to store what type of vector the class label is.
*/
typedef enum {
    ONE_HOT,
    SPARSE
} ClassLabelEncoding;

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
Loads X data set
*/
void load_data(const char *filename, double *data, int start_row, int end_row, int cols);

/*
Loads Y labels
*/
void load_labels(const char *filename, int *labels, int size);

/*
Convert Optimization enum to a string
*/
char* optimization_type_to_string(OptimizationType type);


/*
Convert Activation Enum to a String
*/
char* activation_type_to_string(ActivationType type);



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
Takes the matrix product of w x v. Order matters.
Includes dimensionality checks.
*/
matrix* matrix_mult(matrix* w, matrix* v);

#endif