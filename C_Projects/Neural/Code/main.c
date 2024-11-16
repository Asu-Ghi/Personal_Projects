#include "network.h"
#include "test_functions.h"

#define NUM_FEATURES 4
#define NUM_CLASSES 3
/*
Main Method
*/
int main(int argc, char** argv) {
    // Check command-line arguments (same as before)
    // if (argc < 5) {
    //     printf("Command usage %s num_layers, batch_size, num_epochs, learning_rate", argv[0]);
    //     exit(1);
    // }

    char* file_path = "../DataSets/iris/iris.csv"; // Define file path for data
    matrix X_train, Y_train, X_test, Y_test; // Create matrix objects for data loading
    int num_batches = 150; // 120 training examples in Iris dataset
    double train_ratio = 0.8; // 80% of the data used for training
    
    // Load Iris Data Set    
    load_iris_data(file_path, &X_train, &Y_train, &X_test, &Y_test, num_batches, train_ratio);

    // Create model
    int num_epochs = 1000;
    int num_features = 4;
    int num_layers = 3;
    double learning_rate = .01;
    double decay_rate = 0.0001;
    double beta = 0.90;
    int num_neurons_in_layer[4] = {9, 6, 3};  // 3 output classifications 

    // Get number of examples in the batch for the training dataset.
    int model_num_batches = (int)num_batches*train_ratio;

    NeuralNetwork* network = init_neural_network(num_layers, model_num_batches, num_epochs, num_neurons_in_layer,
                                            learning_rate, RELU, num_features);

    network->beta = beta; // set beta for momentum calculations
    network->momentum = true; // set sgd optimization to use momentum calculations
    network->decay_rate = decay_rate; // set decay rate for network

    print_nn_info(network);
    // Train model
    train_nn(network, &X_train, &Y_train);

    // Free memory
    free_neural_network(network);

}
