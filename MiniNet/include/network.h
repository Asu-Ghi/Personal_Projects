#ifndef NETWORK_H
#define NETWORK_H
#include "layer_dense.h"

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
    double* drop_out_per_layer; // Array of dropout rates for each layer
    bool* regularizations_per_layer; // Array of bools determining whether a layer uses regularizations.
    int* num_neurons_in_layer; // Array of num neurons in each layer respectively
    int num_layers; // Number of layers in network
    int batch_size; // Size of input batch
    int mini_batch_size; // Size of mini batch (if using)
    int num_features; // Number of features in input data
    double learning_rate; // Optimization learning rate
    double decay_rate; // Optimization learning decay rate
    int num_epochs;  // Total number of epochs
    int current_epoch; // Current epoch iteration
    double* accuracy_history; // History of accuracy over training
    double beta_1; // Momentum Hyperparameter
    double beta_2; // Cache Hyperparameter
    double epsilon; // Optimization Hyperparameter
    bool debug; // Prints learning vals after each epoch, otherwise just once at the end
    double accuracy; // Stores network accuracy after each epoch
    double loss; // Stores network loss after each epoch
    double reg_loss; // Stores regularization loss for network
    double val_accuracy; // Stores network validation accuracy after each epoch
    double val_loss; // Stores network validation loss after each epoch
    bool useBiasCorrection; // Flag to determine whether to use bias correction in ADAM
    bool useWeightDecay; // Flag to determine whether to use weight decay for ADAM
    bool early_stopping; // Flag to determine wheter to stop training early based on val_loss ratio
    int send_ratio; // ratio of num epoch that determines when to send socket data
} NeuralNetwork;

typedef struct {
    matrix* X; 
    matrix* Y;
    matrix* X_pred;
    matrix* Y_pred;
} Training_Data;

/*
Initializes the Neural Network architechture 
*/
NeuralNetwork* init_neural_network(int num_layers, int* num_neurons_in_layer, double learning_rate,
                                   ActivationType* activations, OptimizationType* optimizations, bool* regularizations, int num_features);
/*
Frees neural network from memory
*/
void free_neural_network(NeuralNetwork* network);

/*
Frees uneeded memory from every layer in the network after a forward and backward pass.
*/
void free_layers_memory(NeuralNetwork* network);

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
Updates learning rate
*/
void update_learning_rate(NeuralNetwork* network);

/*
Train the neural network
*/
void train_full_batch(NeuralNetwork* network, int num_epochs, Training_Data* training_data);

/*
Trains on mini batches
*/
void train_mini_batch(NeuralNetwork* network, int num_epochs, int batch_size, Training_Data* training_data);

/*
Validate a set of outputs on the model
*/
void validate_model(NeuralNetwork* network, matrix* validate_data, matrix* validate_pred, double* loss, double* accuracy);

/*
Export network parameters into csv files
Takes in directory to save params
Creates csv files for Weights and Biases
s*/
void export_params(NeuralNetwork* network, char* dir_path);

/*
Import network params from file paths.
*/
void load_params(NeuralNetwork* network, char* dir_path);

/*
Function to setup socket once
*/
int setup_socket();

/*
Serialize and pass network data to socket
*/
void send_data(NeuralNetwork* network, int sockfd);

#endif