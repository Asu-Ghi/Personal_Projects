#include "network.h"
#include "test_functions.h"

#define NUM_FEATURES 4
#define NUM_CLASSES 3

/*
Convert spiral data one hot
*/

matrix convert_spiral_labels(matrix* y, int num_samples) {
    matrix y_return;
    y_return.dim1 = y->dim1;
    y_return.dim2 = num_samples;

    for (int i = 0; i < y->dim1; i++) {
        for (int j = 0; j < y->dim2; j++) {
            if (y->data[i]== 0) {
                exit(1);
            }
        }
    }
}

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
    double train_ratio = 0.1; // 80% of the data used for training
    
    // Load Iris Data Set    
    load_iris_data(file_path, &X_train, &Y_train, &X_test, &Y_test, num_batches, train_ratio);

    // Create model
    int num_epochs = 1000;
    int num_features = 4;
    int num_layers = 3;
    double learning_rate = .001;
    double decay_rate = 0.00001;
    double beta_1 = 0.9; // Momentums
    double beta_2 = 0.999; // RMS PROP CACHE
    int num_neurons_in_layer[3] = {9, 6, 3};  // 3 output classifications 
    ActivationType activations_per_layer[3] = {RELU, RELU, SOFTMAX}; // size num layers
    OptimizationType optimizations_per_layer[3] = {SGD_MOMENTUM, SGD_MOMENTUM, SGD_MOMENTUM}; // size num layers
    bool regularization_per_layer[3] = {false, true, false}; // size num layers

    // ~.05 lr for rmsprop
    // ~.1 for sgd_momentum
    // ~.1 for ADAM
    // Get number of examples in the batch for the training dataset.
    int model_num_batches = (int)num_batches*train_ratio;

    NeuralNetwork* network = init_neural_network(num_layers, model_num_batches, num_epochs, num_neurons_in_layer, learning_rate,
                                            activations_per_layer, optimizations_per_layer, regularization_per_layer, num_features);

    network->epsilon = 1e-8; // ADAM epsilon 
    network->beta_1 = beta_1; // set beta1 for momentum calculations
    network->beta_2 = beta_2; // set beta2 for cache calculations

    network->momentum = true; // set sgd optimization to use momentum calculations
    network->decay_rate = decay_rate; // set decay rate for network

    print_nn_info(network);

    network->debug = false;

    // Find appropriate learning rate
    double initial_lr = 1e-5;
    double factor = 1.1;
    double max_lr = 1;
    int iter = 0;
    network->learning_rate = initial_lr;

    // Add regularization variables to hidden layers
    network->layers[1]->lambda_l2 = 5e-5;
    network->layers[1]->lambda_l1 = 5e-7;

    while (initial_lr < max_lr) {

        train_nn(network, &X_train, &Y_train);
        initial_lr *= factor;
        network->learning_rate = initial_lr;
        printf("Iter: %d, Initial: LR %f \n", iter, network->learning_rate);
        // Free network after training
        free_neural_network(network);
        // Re init network with new parameters
        network = init_neural_network(num_layers, model_num_batches, num_epochs, num_neurons_in_layer,
                                            initial_lr, activations_per_layer, optimizations_per_layer,regularization_per_layer, num_features);
        network->debug = false;
        iter++;
    }

    // Free memory
    free_neural_network(network);

}
