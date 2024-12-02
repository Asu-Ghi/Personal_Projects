#include "network.h"
#include "test_network.h"

#define NUM_THREADS 8

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

    #ifdef ENABLE_PARALLEL
    omp_set_num_threads(NUM_THREADS); // Set the number of threads to 8
    #endif 
    
    /*
    Testing network on mnist data
    */
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
    char* dir_path = "results/params/Model_1";
    // load_params(mnist_network, dir_path);
    train_nn(mnist_network, mnist_num_epochs, &mnist_X, &mnist_Y, &mnist_pred_X, &mnist_pred_Y);

    export_params(network_spiral, dir_path);
}
