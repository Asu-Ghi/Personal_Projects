#include "network.h"



/*
Initializes the neural network
*/
NeuralNetwork* init_neural_network(int num_layers, int batch_size, int num_epochs, int* num_neurons_in_layer, double learning_rate,
                                    ActivationType activation) {

    // Allocate memory for the network
    NeuralNetwork* n_network = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));

    // Check Memory Allocation
    if (n_network == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for Neural Network Architecture.\n");
        exit(1);
    }

    // Initialize the network
    n_network->num_layers = num_layers;
    n_network->batch_size = batch_size;
    n_network->learning_rate = learning_rate;
    n_network->num_epochs = num_epochs;
    n_network->activation = activation;
    n_network->num_neurons_in_layer = num_neurons_in_layer;

    // Allocate memory for loss history
    n_network->loss_history = (double*) calloc(n_network->num_epochs, sizeof(double));

    // Check memory allocation for loss history
    if (n_network->loss_history == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for Neural Networks loss history.\n");
        free(n_network);
        exit(1);   
    }

    // Allocate and check memory for layers
    n_network->layers = (layer_dense**) malloc(n_network->num_layers * sizeof(layer_dense*));

    if (n_network->layers == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for layers in Neural Network.\n");
        exit(1);
    }

    // Allocate memory for invidual layers

    // Allocate and check memory for first layer
    n_network->layers[0] = init_layer(n_network->batch_size, n_network->num_neurons_in_layer[0], 
                                    n_network->activation, n_network->batch_size);

    if (n_network->layers[0] == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for the first layer.\n");
        free_layer(n_network->layers[0]);
    }

    // Allocate memory for hidden layers
    for (int i = 1; i < n_network->num_layers - 1; i++) {
        n_network->layers[i] = init_layer(n_network->layers[i-1]->num_neurons, n_network->num_neurons_in_layer[i], 
                                    n_network->activation, n_network->batch_size);
        // Check memory allocation
        if (n_network->layers[i] == NULL) {
            fprintf(stderr, "Error: Memory allocation failed for the %dth layer.\n", i);
            // Free all previous layers
            for (int j = 1; j < i; j++) {
                free_layer(n_network->layers[j]);
            }
            // Free current layer
            free_layer(n_network->layers[i]);
            // Exit program
            exit(1);
        }
    }

    // Allocate and check memory for last layer
    n_network->layers[num_layers-1] = init_layer(n_network->layers[num_layers-2]->num_neurons, n_network->num_neurons_in_layer[num_layers-1], 
                                    SOFTMAX, n_network->batch_size);

    if (n_network->layers[num_layers-1] == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for the last layer.\n");
        free_layer(n_network->layers[num_layers-1]);
        exit(1);
    }

    // Return the neural network pointer
    return(n_network);
}

/*
Displays Neural Net Information
*/
void print_nn_info(NeuralNetwork* network) {
    // Print network constraints
    printf("NUMBER OF LAYERS: %d\n", network->num_layers);
    printf("BATCH SIZE: %d\n", network->batch_size);
    printf("NUMBER OF EPOCHS: %d\n", network->num_epochs);
    printf("LEARNING RATE: %f\n", network->learning_rate);
}

/*
Forward pass on the neural network
*/
void forward_pass_nn(NeuralNetwork* network, matrix* inputs) {

    // First forward pass
    forward_pass(inputs, network->layers[0]);

    // Forward pass for hidden layers
    for(int i = 1; i < network->num_layers - 1; i++) {
        forward_pass(network->layers[i - 1]->post_activation_output, network->layers[i]);
    }

    // Last forward pass
    forward_pass(network->layers[network->num_layers - 2]->post_activation_output, network->layers[network->num_layers - 1]);
}

/*
Backward Pass on the Neural Nework.
*/
void backward_pass_nn(NeuralNetwork* network, matrix* y_pred) {

    // Start with the backward pass for softmax and loss
    backwards_softmax_and_loss(y_pred, network->layers[network->num_layers-1]);

    // Go backwards through every hidden layer
    for (int i = network->num_layers - 2; i >= 0; i--) {
        backward_reLu(network->layers[i+1]->dinputs, network->layers[i]);
    }
    
}


/*
Update the parameters in the neural network
*/
void update_parameters(NeuralNetwork* network) {
    // Loop through the first and all hidden layers
    for (int i = 0; i < network->num_layers - 1; i++) {
        // Update weights and biases for each layer(currently uses SGD)
        update_params_sgd(network->layers[i], network->learning_rate);
    }
}

/*
Train the neural network 
*/
void train_nn(NeuralNetwork* network, matrix* X, matrix* Y) {

    for (int epoch = 0; epoch < network->num_epochs; epoch++) {

        // Step 1: Forward Pass
        printf("Arrived at forward pass.\n");
        forward_pass_nn(network, X);

        // Step 2: Calculate Loss and Accuracy
        printf("Arrived at accuracy.\n");
        double accuracy = calculate_accuracy(Y, network->layers[network->num_layers-1], ONE_HOT);
        double loss = loss_categorical_cross_entropy(Y, network->layers[network->num_layers], ONE_HOT);

        // Add loss to the loss history
        network->loss_history[epoch] = loss;

        // Print training data
        printf("Epoch %d: Loss = %f, Accuracy = %f\n", epoch, loss, accuracy);

        // Step 3: Backward Pass
        printf("Arrived at backwards pass.\n");
        backward_pass_nn(network, Y);

        // Step 4: Update Weights
        printf("Arrived at update weights.\n");
        update_parameters(network);
    }
}
