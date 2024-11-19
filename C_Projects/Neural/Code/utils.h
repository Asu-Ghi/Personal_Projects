/*
Asu Ghimire
11/17/2024

Utility "Class".
Provides basic utilites for forward.c, backward.c, network.c
Handles 
    > Linear algebra functionalities
    > Data loading 
    > Layer initialization
    > Accuracy calculation
    > Loss calculation
    > Optimization
*/
#ifndef UTILS_H
#define UTILS_H

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
layer_dense data structure 
Has matrix data structures that hold layer info
Includes information for forward and backward passes.
*/
typedef struct {
    int id;
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
    matrix* w_velocity; // used to calculate momentums for weight
    matrix* b_velocity; // used to calculate momentums for bias
    matrix* cache_weights; // used to adjust gradients 
    matrix* cache_bias; // used in adagrad for adjusting gradients
    ActivationType activation; // Stores activation function
    OptimizationType optimization; // Store optimization function
    bool useRegularization; // Flag to determine if using regularization in forward and backwards.
    double lambda_l1;  // L1 regularization coefficient
    double lambda_l2;  // L2 regularization coefficient} 
}layer_dense;

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
layer_dense* init_layer(int num_inputs, int num_neurons, ActivationType activation, OptimizationType optimization, int batch_size);

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
matrix* loss_categorical_cross_entropy(matrix* true_pred, layer_dense* last_layer, ClassLabelEncoding encoding);

/*
Calculates regularization l1 and l2 for a given layer.
Checks to see if layer has property "useRegularization"
Returns a double of summed regularization costs for l1 and l2 weights and biases
*/
double calculate_regularization_loss(layer_dense* layer);

/*
SGD Optimization
Stochastic Gradient Descent with learning rate decay.
*/
void update_params_sgd(layer_dense* layer, double* learning_rate, int current_epoch, double decay_rate);

/*
SGD Optimization with momentum
Stochastic Gradient Descent with Momentum.
Uses momentum to help push out of local extrema when performing gradient descent through SGD.
Take in extra hyperparameters beta to calculate momentumns.
*/
void update_params_sgd_momentum(layer_dense* layer, double* learning_rate, int current_epoch, double decay_rate, double beta);


/*
ADA GRAD Optimization
Adaptive Gradient. 
Normalizes the layer gradients with a store local per parameter learning rate calculated from previous gradient changes.
*/
void update_params_adagrad(layer_dense* layer, double* learning_rate, double decay_rate, double epsilon);

/*
RMSPROP
Root mean Squared Propogation. 
Similar to Adaptive Gradient with local per parameter learning rate, but uses a different formula to calculate cache.
*/
void update_params_rmsprop(layer_dense* layer, double* learning_rate, double decay_rate, double epsilon);

/*
Adam Optimization
Adaptive Momentum.
For Batch Gradient Descent t = current epoch
For Mini Batch Gradient Descent t is incriminted after every mini batch.
Beta_1 and Beta_2 are hyperparameters affecting momentum and RMSProp caches respectively. 
*/
void update_params_adam (layer_dense* layer, double* learning_rate, double decay_rate, 
                    double beta_1, double beta_2, double epsilon, int t, bool correctBias);

/*
Clips gradients to a min and max value, useful if experiencing exploding gradients. Applied in backwards.
*/
void clip_gradients(double* gradients, int size, double min_value, double max_value);



#endif