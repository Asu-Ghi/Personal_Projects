#ifndef TEST_FUNCTIONS_H
#define TEST_FUNCTIONS_H

#include "network.h"

// Define constants for layer testing 
// NOTE (WEIGHT RANDOMIZATION MUST HAVE A SET SEED OF 42) //

#define NUM_NEURONS_1 4

#define NUM_INPUT_FEATURES_2 4
#define NUM_NEURONS_2 3

#define NUM_INPUT_FEATURES_3 3
#define NUM_NEURONS_3 3 // Matches num of batch features in classification

// Define constants for network testing 
#define BATCH_SIZE 3
#define NUM_BATCH_FEATURES 3
#define NUM_LAYERS 3
#define NUM_EPOCHS 40
#define NUM_NEURONS_LAYER_1 9
#define NUM_NEURONS_LAYER_2 6
#define NUM_NEURONS_LAYER_3 3
#define LEARNING_RATE 0.01

///////////////////////////////////////////////////LINEAR ALGEBRA FUNCTIONS////////////////////////////////////////////////////////////////

/*
Test Matrix Transpose.
    > Transpose a matrix
    > Ensure dimensons are flipped
    > Ensure data is copied properly
*/
void test_matrix_transpose();

/*
Test Matrix Multiplication.
Compares M1(4x1) * M2(1x4) 
M1- >[1, 2, 3, 4]
M2 -> 1, 2, 3, 4]
    > Ensure dimensons are valid
    > Ensure the multiplication is valid
*/
void test_matrix_mult();


///////////////////////////////////////////////////LAYER FUNCTIONS////////////////////////////////////////////////////////////////


/*
Test Init Layer
Check if layer objects are being initialized properly.
    > Initialize layers
    > Check their dimensions
    > Print out weight and bias matrices to verify and validate.
Expected Dimensions:
    >Weights/dWeights -> 3x4
    >Biases/dBiases -> 1x4
    >Inputs/dInputs -> 10x3
    >Pre/Post Activation Outputs -> 10x4
*/
void test_init_layer();

/*
Test Forward Pass
Input Data = [[1, 2, 3], [1, 2, 3]]
    > Call forward pass
    > Ensure each layer produces correct dimension outputs.
    > Print layer outputs to verify that matrix multiplication is valid
*/
void test_forward_pass();

/*
Test Loss (Categorical Cross Entropy)
    > Call forward pass
    > Call loss function at the end
    > Ensure loss produces valid outputs
*/
void test_loss_categorical();

/*
Test Accuracy
    > Call forward pass
    > Calculate accuaracy at the end
    > Ensure accuracy calculated is valid.
*/
void test_accuracy();

/*
Test Backwards Pass
    > Call forward pass
    > Call backward pass
    > Ensure layer gradient matrices are correct dimensions
    > Print out gradients for each layer in simple test case
*/
void test_backward_pass();

/*
Test Stochastic Gradient Descent Method
    > Check to see if losses are decreasing after each forward and backward pass
    > Check to see if gradients are being "exploded". 
*/
void test_update_params_sgd();

///////////////////////////////////////////////////NETWORK FUNCTIONS////////////////////////////////////////////////////////////////

/*
Test init_neural_network
*/
void test_init_neural_network();

/*
Test forward pass on a network
*/
void test_forward_pass_nn();

/*
Test backward pass on a network
*/
void test_backward_pass_nn();

/*
Test training the neural network.
*/
void test_train_nn();

/*
Test the predictions of a network
*/
void test_predict();


//////////////////////////////////////////////////TEST ALL METHODS/////////////////////////////////////////////////////////////////

/*
Tests Every Method defined above.
*/
void test_all_methods();


#endif
