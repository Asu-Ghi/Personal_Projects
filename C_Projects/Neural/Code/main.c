#include "network.h"
#include "test_functions.h"

#define NUM_FEATURES 4
#define NUM_CLASSES 3

/*
Convert spiral data one hot
*/
matrix convert_spiral_labels(matrix* y, int num_classes) {
    matrix y_return;
    y_return.dim1 = y->dim1;
    y_return.dim2 = num_classes;
    y_return.data = calloc(y_return.dim1*y_return.dim2,sizeof(double));

    // Check memory allocation
    if (!y_return.data) {
        fprintf(stderr, "Memory allocation failed in convert spiral labels.\n");
        exit(1);
    }

    for (int i = 0; i < y->dim1; i++) {
        int label = y->data[i];
        if(label < num_classes && label >= 0) {
            // Create one hot
            y_return.data[i * y_return.dim2 + label] = 1.0;
        }
    }
    return y_return;
}


/*
Find best LR
*/
void find_best_lr(NeuralNetwork* network, matrix* x, matrix* y_pred, int num_epochs, double init_lr, int max_lr, double lambda1,
                    double lambda2, double beta_1, double beta_2, double epsilon, double lr_factor, double decay_rate,
                    matrix* X_test, matrix* Y_test) {
    int num_batches = network->batch_size;
    int num_features = network->num_features;
    int* num_neurons_in_layer = network->num_neurons_in_layer;
    ActivationType* activations_in_layer = network->activations_per_layer;
    OptimizationType* optimizations_in_layer = network->optimizations_per_layer;
    bool* regularization_per_layer = network->regularizations_per_layer;
    int num_layers = network->num_layers;
    double best_acc = 0.0;
    double best_lr = 0.0;
    int iter = 0;
    while (init_lr < max_lr) {
        // Free network after training
        free_neural_network(network);

        // Re init network with new parameters
        network = init_neural_network(num_layers, num_batches, num_epochs, num_neurons_in_layer, init_lr,
                                            activations_in_layer, optimizations_in_layer, regularization_per_layer, num_features);

        network->momentum = true; // set sgd optimization to use momentum calculations
        network->decay_rate = decay_rate;
        network->debug = false;

        // retrain
        train_nn(network, x, y_pred, X_test, Y_test);

        // update best lr based on acc
        if (network->accuracy > best_acc) {
            best_lr = network->learning_rate;
            best_acc = network->accuracy;
        }

        // Debug
        printf("Iter: %d, Initial: LR %f \n", iter, network->learning_rate);

        init_lr *= lr_factor;

        iter++;
    }

    printf("Best learning rate = %f, Best Acc = %f\n", best_lr, best_acc);
}


void find_best_beta() {
    exit(1);
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

    // Load spiral data
    // Init training
    matrix spiral_train, spiral_pred, spiral_test, spiral_test_pred;
    spiral_train.dim1 = 64;
    spiral_train.dim2 = 2;
    spiral_train.data = (double*) calloc(spiral_train.dim1 * spiral_train.dim2, sizeof(double));

    spiral_pred.dim1 = 64;
    spiral_pred.dim2 = 3;
    spiral_pred.data = (double*) calloc(spiral_pred.dim1 * spiral_pred.dim2, sizeof(double));

    // Init validating
    spiral_test.dim1 = 64;
    spiral_test.dim2 = 2;
    spiral_test.data = (double*) calloc(spiral_test.dim1 * spiral_test.dim2, sizeof(double));

    spiral_test_pred.dim1 = 64;
    spiral_test_pred.dim2 = 3;
    spiral_test_pred.data = (double*) calloc(spiral_test_pred.dim1 * spiral_test_pred.dim2, sizeof(double));

    // Load training
    load_data("../DataSets/Spiral/train_data.csv", spiral_train.data, 0, 64, 2);
    load_data("../DataSets/Spiral/train_labels.csv", spiral_pred.data, 0, 64, 3);

    // Load validating
    load_data("../DataSets/Spiral/test_data.csv", spiral_test.data, 0, 64, 2);
    load_data("../DataSets/Spiral/test_labels.csv", spiral_test_pred.data, 0, 64, 3);

    int spiral_num_batches = 64;
    int spiral_num_features = 2;
    int spiral_neurons_in_layer[2] = {64, 3}; // Num neurons in a layer
    ActivationType spiral_activations_per_layer[2] = {RELU, SOFTMAX}; // size num layers
    OptimizationType spiral_optimizations_per_layer[2] = {ADAM, ADAM}; // size num layers
    bool spiral_regularization_per_layer[2] = {true, true}; // size num layers

    // Find best lr
    int num_epochs = 3000;
    double init_lr = 0.001;
    double decay_rate = 5e-7;
    int max_lr = 1;
    double lr_factor = 1.1;
    double epsilon = 1e-6;
    double beta_1 = 0.9; // Momentums
    double beta_2 = 0.999; // RMS PROP CACHE
    double lambda1 = 1e-6;
    double lambda2 = 1e-5;

    NeuralNetwork* network_spiral = init_neural_network(2, spiral_num_batches, num_epochs, spiral_neurons_in_layer, init_lr,
                                            spiral_activations_per_layer, spiral_optimizations_per_layer, spiral_regularization_per_layer, spiral_num_features);
    
    network_spiral->layers[0]->lambda_l1 = lambda1;
    network_spiral->layers[0]->lambda_l2 = lambda2;

    network_spiral->layers[1]->lambda_l1 = lambda1;
    network_spiral->layers[1]->lambda_l2 = lambda2;

    // network_spiral->layers[2]->lambda_l1 = lambda1;
    // network_spiral->layers[2]->lambda_l2 = lambda2;
    
    // network_spiral->layers[3]->lambda_l1 = lambda1;
    // network_spiral->layers[3]->lambda_l2 = lambda2;

    print_nn_info(network_spiral);
    network_spiral->useBiasCorrection = false; // works way better with adam for this dataset
    train_nn(network_spiral, &spiral_train, &spiral_pred, &spiral_test, &spiral_test_pred);

    // find_best_lr(network_spiral, &spiral_train, &spiral_pred, num_epochs,init_lr, max_lr,
    //                          lambda1, lambda2, beta_1, beta_2, epsilon, lr_factor, decay_rate, X_test, Y_test);



    // Free memory
    free_neural_network(network_spiral);
    

    free(spiral_pred.data);
    free(spiral_train.data);

}
