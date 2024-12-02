#include "test_network.h"


void test_mnist() {
matrix mnist_X, mnist_Y, mnist_pred_X, mnist_pred_Y;

    mnist_X.dim1 = 10000; // 10k samples
    mnist_X.dim2 = 784; // 785 pixel values
    mnist_X.data = (double*) calloc(mnist_X.dim1 * mnist_X.dim2,sizeof(double));

    mnist_Y.dim1 = 10000; // 10k samples
    mnist_Y.dim2 = 10; // 10 labels (digits 1 - 10)
    mnist_Y.data = (double*) calloc(mnist_Y.dim1 * mnist_Y.dim2,sizeof(double));


    mnist_pred_X.dim1 = 100; // 100 samples
    mnist_pred_X.dim2 = 784; // 785 pixel values
    mnist_pred_X.data = (double*) calloc(mnist_pred_X.dim1 * mnist_pred_X.dim2,sizeof(double));

    mnist_pred_Y.dim1 = 100; // 100 samples
    mnist_pred_Y.dim2 = 10; // 10 labels (digits 1 - 10)
    mnist_pred_Y.data = (double*) calloc(mnist_pred_Y.dim1 * mnist_pred_Y.dim2,sizeof(double));
    
    load_mnist_data("data/MNIST/mnist_test.csv", mnist_X.data, mnist_Y.data, 10000);
    load_mnist_data("data/MNIST/mnist_train.csv", mnist_pred_X.data, mnist_pred_Y.data, 100);
    int mnist_n_layers = 2;
    int mnist_n_features = 784;
    int num_neurons_mnist[2] = {784, 10};
    ActivationType activations_mnist[2] = {RELU, SOFTMAX};
    OptimizationType optimizations_mnist[3] = {ADAM, ADAM};
    bool regularizations_mnist[5] = {true, true};
    double mnist_lr = 0.05;
    int mnist_num_epochs = 100;
    double init_lr = 0.05;
    double decay_rate = 5e-7;
    int max_lr = 1;
    double lr_factor = 1.01;
    double epsilon = 1e-7;
    double beta_1 = 0.9; // Momentums
    double beta_2 = 0.925; // RMS PROP CACHE
    double lambda1 = 1e-5;
    double lambda2 = 5e-4;

    NeuralNetwork* mnist_network = init_neural_network(mnist_n_layers, num_neurons_mnist, mnist_lr,
                                            activations_mnist, optimizations_mnist, regularizations_mnist, mnist_n_features);

    int mnist_num_epochs = 100;
    mnist_network->beta_2 = beta_2;
    mnist_network->epsilon = epsilon;
    mnist_network->decay_rate = decay_rate;
    mnist_network->debug = true;
    mnist_network->useBiasCorrection = true;
    mnist_network->decay_rate = decay_rate;
    
    mnist_network->layers[0]->lambda_l1 = lambda1;
    mnist_network->layers[0]->lambda_l2 = lambda2;

    mnist_network->layers[1]->lambda_l1 = lambda1;
    mnist_network->layers[1]->lambda_l2 = lambda2;

    // mnist_network->layers[2]->lambda_l1 = lambda1;
    // mnist_network->layers[2]->lambda_l2 = lambda2;

    // mnist_network->layers[3]->lambda_l1 = lambda1;
    // mnist_network->layers[3]->lambda_l2 = lambda2;

    // mnist_network->layers[4]->lambda_l1 = lambda1;
    // mnist_network->layers[4]->lambda_l2 = lambda2;


    mnist_network->layers[0]->drop_out_rate = 0.3;
    // mnist_network->layers[1]->drop_out_rate = 0.3;

    train_nn(mnist_network, mnist_num_epochs, &mnist_X, &mnist_Y, &mnist_pred_X, &mnist_pred_Y);
    char* dir_path = "results/params/Model_1";
}

void test_spiral() {

    // Init Matrix Memory
    matrix spiral_train, spiral_pred, spiral_test, spiral_test_pred;
    spiral_train.dim1 = 10000;
    spiral_train.dim2 = 2;
    spiral_train.data = (double*) calloc(spiral_train.dim1 * spiral_train.dim2, sizeof(double));

    spiral_pred.dim1 = 10000;
    spiral_pred.dim2 = 3;
    spiral_pred.data = (double*) calloc(spiral_pred.dim1 * spiral_pred.dim2, sizeof(double));

    // Init validating
    spiral_test.dim1 = 300;
    spiral_test.dim2 = 2;
    spiral_test.data = (double*) calloc(spiral_test.dim1 * spiral_test.dim2, sizeof(double));

    spiral_test_pred.dim1 = 300;
    spiral_test_pred.dim2 = 3;
    spiral_test_pred.data = (double*) calloc(spiral_test_pred.dim1 * spiral_test_pred.dim2, sizeof(double));

    // Training
    load_spiral_data("data/Spiral/test_data_10000.csv", spiral_train.data, 0, 10000, 2);
    load_spiral_data("data/Spiral/test_labels_10000.csv", spiral_pred.data, 0, 10000, 3);   

    // Validation
    load_spiral_data("data/Spiral/test_data.csv", spiral_test.data, 0, 300, 2);
    load_spiral_data("data/Spiral/test_labels.csv", spiral_test_pred.data, 0, 300, 3);


    int spiral_num_features = 2;
    int spiral_neurons_in_layer[3] = {512, 3}; // Num neurons in a layer
    ActivationType spiral_activations_per_layer[2] = {RELU, SOFTMAX}; // size num layers
    OptimizationType spiral_optimizations_per_layer[2] = {ADAM, ADAM}; // size num layers
    bool spiral_regularization_per_layer[2] = {true, true}; // size num layers
    int num_epochs = 5000;
    double init_lr = 0.05;
    double decay_rate = 5e-7;
    int max_lr = 1;
    double lr_factor = 1.01;
    double epsilon = 1e-7;
    double beta_1 = 0.9; // Momentums
    double beta_2 = 0.925; // RMS PROP CACHE
    double lambda1 = 1e-5;
    double lambda2 = 5e-4;

    NeuralNetwork* network_spiral = init_neural_network(2, spiral_neurons_in_layer, init_lr,
                                        spiral_activations_per_layer, spiral_optimizations_per_layer, spiral_regularization_per_layer, spiral_num_features);
    network_spiral->layers[0]->id = 1;
    network_spiral->layers[1]->id = 2;
    network_spiral->layers[0]->lambda_l1 = lambda1;
    network_spiral->layers[0]->lambda_l2 = lambda2;
    network_spiral->layers[1]->lambda_l1 = lambda1;
    network_spiral->layers[1]->lambda_l2 = lambda2;
    network_spiral->layers[0]->drop_out_rate = 0.3;
    network_spiral->layers[0]->clip_value = 0;
    network_spiral->beta_1 = beta_1;
    network_spiral->beta_2 = beta_2;
    network_spiral->epsilon = epsilon;
    network_spiral->decay_rate = decay_rate;
    network_spiral->debug = true;
    network_spiral->useBiasCorrection = true;

    #ifdef ENABLE_PARALLEL
    printf("NUMBER OF CPU CORES: %d\n", NUM_THREADS);
    #else
    printf("Calculating Sequentially..\n");
    #endif  

    train_nn(network_spiral, num_epochs, &spiral_train, &spiral_pred, &spiral_test, &spiral_test_pred);
}

void test_iris() {

}