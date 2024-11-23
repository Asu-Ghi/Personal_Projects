#include "network.h"

#define NUM_FEATURES 4
#define NUM_CLASSES 3
#define NUM_THREADS 6

/*
Find best betas for optimization
*/
void find_best_beta() {
    exit(1);
}

/*
Main Method
*/
int main(int argc, char** argv) {

    omp_set_num_threads(NUM_THREADS); // Set the number of threads to 8

    char* file_path = "../data/iris/iris.csv"; // Define file path for data
    matrix X_train, Y_train, X_test, Y_test; // Create matrix objects for data loading
    int num_batches = 150; // 120 training examples in Iris dataset
    double train_ratio = 0.1; // 80% of the data used for training

    // Load Iris Data Set    
    // load_iris_data(file_path, &X_train, &Y_train, &X_test, &Y_test, num_batches, train_ratio);

    // Load spiral data
    // Init training
    matrix spiral_train, spiral_pred, spiral_test, spiral_test_pred;
    spiral_train.dim1 = 750;
    spiral_train.dim2 = 2;
    spiral_train.data = (double*) calloc(spiral_train.dim1 * spiral_train.dim2, sizeof(double));

    spiral_pred.dim1 = 750;
    spiral_pred.dim2 = 3;
    spiral_pred.data = (double*) calloc(spiral_pred.dim1 * spiral_pred.dim2, sizeof(double));

    // Init validating
    spiral_test.dim1 = 100;
    spiral_test.dim2 = 2;
    spiral_test.data = (double*) calloc(spiral_test.dim1 * spiral_test.dim2, sizeof(double));

    spiral_test_pred.dim1 = 100;
    spiral_test_pred.dim2 = 3;
    spiral_test_pred.data = (double*) calloc(spiral_test_pred.dim1 * spiral_test_pred.dim2, sizeof(double));

    // Load training
    // load_data("../DataSets/Spiral/train_data.csv", spiral_train.data, 0, 300, 2);
    // load_data("../DataSets/Spiral/train_labels.csv", spiral_pred.data, 0, 300, 3);

    load_data("data/Spiral/train_data_1000.csv", spiral_train.data, 0, 750, 2);
    load_data("data/Spiral/train_data_1000.csv", spiral_pred.data, 0, 750, 3);


    // Load validating
    load_data("data/Spiral/test_data.csv", spiral_test.data, 0, 100, 2);
    load_data("data/Spiral/test_labels.csv", spiral_test_pred.data, 0, 100, 3);

    int spiral_num_features = 2;
    int spiral_neurons_in_layer[3] = {512, 256, 3}; // Num neurons in a layer
    ActivationType spiral_activations_per_layer[3] = {RELU, RELU, SOFTMAX}; // size num layers
    OptimizationType spiral_optimizations_per_layer[3] = {ADAM, ADAM, ADAM}; // size num layers
    bool spiral_regularization_per_layer[3] = {true, true, true}; // size num layers

    // Find best lr
    int num_epochs = 10000;
    double init_lr = 0.05;
    double decay_rate = 1e-5;
    int max_lr = 1;
    double lr_factor = 1.01;
    double epsilon = 1e-7;
    double beta_1 = 0.85; // Momentums
    double beta_2 = 0.90; // RMS PROP CACHE
    double lambda1 = 1e-7;
    double lambda2 = 1e-6;

    NeuralNetwork* network_spiral = init_neural_network(3, spiral_neurons_in_layer, init_lr,
                                            spiral_activations_per_layer, spiral_optimizations_per_layer, spiral_regularization_per_layer, spiral_num_features);
    
    network_spiral->layers[0]->lambda_l1 = lambda1;
    network_spiral->layers[0]->lambda_l2 = lambda2;
    network_spiral->beta_1 = beta_1;
    network_spiral->beta_2 = beta_2;
    network_spiral->epsilon = epsilon;

    // network_spiral->layers[1]->lambda_l1 = lambda1;
    // network_spiral->layers[1]->lambda_l2 = lambda2;

    // network_spiral->layers[2]->lambda_l1 = lambda1;
    // network_spiral->layers[2]->lambda_l2 = lambda2;

    // network_spiral->layers[3]->lambda_l1 = lambda1;
    // network_spiral->layers[3]->lambda_l2 = lambda2;

    network_spiral->layers[0]->drop_out_rate = .3;
    // network_spiral->layers[1]->drop_out_rate = 0.3;
    // network_spiral->layers[2]->drop_out_rate = 0.3;
    // network_spiral->layers[3]->drop_out_rate = 0.3;


    network_spiral->layers[0]->clip_value = 0.0;
    // network_spiral->layers[1]->clip_value = 1;
    // network_spiral->layers[2]->clip_value = 1;
    // network_spiral->layers[3]->clip_value = 1;


    network_spiral->decay_rate = decay_rate;

    // print_nn_info(network_spiral);
    network_spiral->debug = false;
    network_spiral->useBiasCorrection = false; // works way better with adam for this dataset
    train_nn(network_spiral, num_epochs, &spiral_train, &spiral_pred, &spiral_test, &spiral_test_pred);

    // find_best_lr(network_spiral, &spiral_train, &spiral_pred, num_epochs,init_lr, max_lr,
    //                          lambda1, lambda2, beta_1, beta_2, epsilon, lr_factor, decay_rate, &spiral_test, &spiral_test_pred);



    // Free memory
    // free_neural_network(network_spiral);
    

    free(spiral_pred.data);
    free(spiral_train.data);

}
