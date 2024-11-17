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

    // Load Spiral Data
    int samples = 100;
    int classes = 3;
    int rows = samples * classes;
    int cols = 2;
    matrix spiral_X, spiral_y;
    spiral_X.dim1 = rows;
    spiral_X.dim2 = cols;
    spiral_X.data = (double*) calloc(spiral_X.dim1 * spiral_X.dim2, sizeof(double));

    spiral_y.dim1 = samples;
    spiral_y.dim2 = classes;
    spiral_y.data = (double*) calloc(spiral_y.dim1 * spiral_y.dim2, sizeof(double));

    char* spiral_Xpath = "./X_data.csv";
    char* spiral_Ypath = "../y_labels.csv";

    load_data(spiral_Xpath, spiral_X.data, rows, cols);
    load_data(spiral_Ypath, spiral_y.data, samples, 1);

    // CONVERT Y TO ONE HOT->


    // Create model
    int num_epochs = 1000;
    int num_features = 4;
    int num_layers = 3;
    double learning_rate = .01;
    double decay_rate = 0.0001;
    double beta_1 = 0.95; // Momentums
    double beta_2 = 0.999; // RMS PROP CACHE
    int num_neurons_in_layer[3] = {9, 6, 3};  // 3 output classifications 
    ActivationType activations_per_layer[3] = {RELU, RELU, SOFTMAX};
    OptimizationType optimizations_per_layer[3] = {SGD_MOMENTUM, SGD_MOMENTUM, SGD_MOMENTUM};

    // Get number of examples in the batch for the training dataset.
    int model_num_batches = (int)num_batches*train_ratio;

    NeuralNetwork* network = init_neural_network(num_layers, model_num_batches, num_epochs, num_neurons_in_layer,
                                            learning_rate, activations_per_layer, optimizations_per_layer, num_features);


    network->beta_1 = beta_1; // set beta1 for momentum calculations
    network->beta_2 = beta_2; // set beta2 for cache calculations

    network->momentum = true; // set sgd optimization to use momentum calculations
    network->decay_rate = decay_rate; // set decay rate for network

    print_nn_info(network);
    // Train model
    // train_nn(network, &X_train, &Y_train);

    print_matrix(&spiral_X);

    // Free memory

    free(&spiral_X);
    free_neural_network(network);

}
