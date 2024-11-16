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
    layer_dense** layers;
    int* num_neurons_in_layer;
    int num_layers;
    int batch_size;
    int num_features;
    double learning_rate;
    double decay_rate;
    int num_epochs;
    int current_epoch;
    double* loss_history;
    bool momentum;
    double beta;
    ActivationType activation;
} NeuralNetwork;

/*
Initializes the Neural Network architechture 
*/
NeuralNetwork* init_neural_network(int num_layers, int batch_size, int num_epochs, int* num_neurons_in_layer, double learning_rate,
                                   ActivationType activation, int num_features);
/*
Frees neural network from memory
*/
void free_neural_network(NeuralNetwork* network);

/*
Prints information for the neural net.
*/
void print_nn_info(NeuralNetwork* network);

/*
Train the neural network
*/
void train_nn(NeuralNetwork* network, matrix* X, matrix* Y);

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
Predict on the network
*/
// Function to predict a class label for new data (for classification)
void predict(NeuralNetwork* network, matrix* input_data);

#endif