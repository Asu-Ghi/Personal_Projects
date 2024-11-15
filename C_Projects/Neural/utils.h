// utils.h
#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h> 
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
Activation function enum structure
Enum to store what activation function is being used
*/
typedef enum {
    RELU,
    SOFTMAX
} ActivationType;

/*
layer_dense data structure 
Has matrix data structures that hold layer info
Includes information for forward and backward passes.
*/
typedef struct {
    char* id;
    int num_neurons;
    int num_inputs;
    matrix* weights;
    matrix* biases;
    matrix* inputs;
    matrix* pre_activation_output;
    matrix* post_activation_output;
    matrix* dweights;
    matrix* dbiases;
    matrix* dinputs;
    ActivationType activation;
} layer_dense;

/*
Class label encoding enum structure
Enum to store what type of vector the class label is.
*/
typedef enum {
    ONE_HOT,
    SPARSE
} ClassLabelEncoding;

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

//////////////////////////////////////////////////// Network Methods //////////////////////////////////////////////////////////////////

/*
Returns a layer dense object. Allocates memory on the heap for the object.
Takes in number of inputs and neurons for a given layer, as well as the activation function to be applied.
Assigns value for activation type.
Allocates and checks memory for:
    >weights
    >biases
    >dweights
    >dbiases
    >inputs
    >dinputs
    >pre_activation_outputs
    >post_activation_outputs
*/
layer_dense* init_layer(int num_inputs, int num_neurons, ActivationType activation, int batch_size);

/*
Frees layer dense memory. Deallocates memory for 
    >weights
    >biases
    >inputs
    >pre_activation_output
    >post_activation_output
    >dweights
    >dbiases
    >dinputs
and the layer itself.
*/
void free_layer(layer_dense* layer);

/*
Calculates the accuracy of the network.
Takes in class targets as either one hot or sparse vectors
Takes in the final layer (with the classification activation function)
Takes in the type of class target encoding (either one hot or sparse)
Returns a double 'Accuracy'
*/
double calculate_accuracy(matrix* class_targets, layer_dense* final_layer, ClassLabelEncoding encoding);

/*
Calculates the loss of the network. 
Takes in the true predictions as either one hot or sparse vectors.
Takes in the final layer of softmax outputs.
Takes in the type of class target encoding (either one hot or sparse)
Returns a matrix of loss coresponding to the output of softmax and the # of classes to be classified.
*/
double loss_categorical_cross_entropy(matrix* true_pred, layer_dense* last_layer, ClassLabelEncoding encoding);

/*
SGD Optimization
ADD INFO HERE
*/
void update_params_sgd(layer_dense* layer, double learning_rate);


/*
Loads IRIS Data in for training
*/
void load_iris_data(const char* file_path, matrix* X, matrix* Y);


#endif