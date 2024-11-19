/*
Asu Ghimire
11/17/2024

Network "Class".
Interfaces with forward.c, backward.c, utils.c to allow for the creation of a Neural Network.

*/

#ifndef NETWORK_H
#define NETWORK_H
#include "forward.h"
#include "backward.h"

/*
Neural Network data structure
Includes
    > Pointer to dense layers
    > Number of layers in the network
    > The size of the input batch
    > The learning rate of the network
    > The number of epochs for training iterations
    > An array of loss history per training iteration
*/
typedef struct {
    layer_dense** layers; // Layer dense array of layer dense pointers.
    ActivationType* activations_per_layer; // Array of activation type for each layer respectively
    OptimizationType* optimizations_per_layer; // Array of optimization type for each layer respectivley.
    bool* regularizations_per_layer; // Array of bools determining whether a layer uses regularizations.
    int* num_neurons_in_layer; // Array of num neurons in each layer respectively
    int num_layers; // Number of layers in network
    int batch_size; // Size of input batch
    int num_features; // Number of features in input data
    double learning_rate; // Optimization learning rate
    double decay_rate; // Optimization learning decay rate
    int num_epochs;  // Total number of epochs
    int current_epoch; // Current epoch iteration
    double* loss_history; // History over batch losses
    bool momentum; // Momentum flag
    double beta_1; // Momentum Hyperparameter
    double beta_2; // Cache Hyperparameter
    double epsilon; // Optimization Hyperparameter
    bool debug; // Prints learning vals after each epoch, otherwise just once at the end
    double accuracy; // Stores network accuracy after each epoch
    double loss; // Stores network loss after each epoch
    double val_accuracy; // Stores network validation accuracy after each epoch
    double val_loss; // Stores network validation loss after each epoch
    bool useBiasCorrection; // Flag to determine wheter to use bias correction in ADAM
} NeuralNetwork;

/*
Initializes the Neural Network architechture 
*/
NeuralNetwork* init_neural_network(int num_layers, int batch_size, int num_epochs, int* num_neurons_in_layer, double learning_rate,
                                   ActivationType* activations, OptimizationType* optimizations, bool* regularizations, int num_features);
/*
Frees neural network from memory
*/
void free_neural_network(NeuralNetwork* network);

/*
Prints information for the neural net.
*/
void print_nn_info(NeuralNetwork* network);

/*
Forward pass on the neural network
*/
void forward_pass_nn(NeuralNetwork* network, matrix* inputs);

/*
Backward pass through the neural network
*/
void backward_pass_nn(NeuralNetwork* network, matrix* y_true);

/*
Update the parameters of the neural network
*/
void update_parameters(NeuralNetwork* network);

/*
Train the neural network
*/
void train_nn(NeuralNetwork* network, matrix* X, matrix* Y, matrix* X_validate, matrix* Y_validate);


/*
Predict on the network
*/
// Function to predict a class label for new data (for classification)
void predict(NeuralNetwork* network, matrix* input_data);

/*
Validate a set of outputs on the model
*/
void validate_model(NeuralNetwork* network, matrix* validate_data, matrix* validate_pred, double* loss, double* accuracy);

#endif