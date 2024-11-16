#include "network.h"


NeuralNetwork* init_neural_network(int num_layers, int batch_size, int num_epochs, int* num_neurons_in_layer, double learning_rate,
                                   ActivationType activation, int num_batch_features) {

    // Allocate memory for the network
    NeuralNetwork* n_network = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));

    // Initialize network parameters
    n_network->num_layers = num_layers;
    n_network->batch_size = batch_size;
    n_network->learning_rate = learning_rate;
    n_network->num_epochs = num_epochs;
    n_network->activation = activation;
    n_network->num_neurons_in_layer = num_neurons_in_layer;

    // Allocate memory for loss history and layers
    n_network->loss_history = (double*) calloc(n_network->num_epochs, sizeof(double));
    n_network->layers = (layer_dense**) malloc(n_network->num_layers * sizeof(layer_dense*));

    // Allocate memory for the first layer with `num_features`
    n_network->layers[0] = init_layer(num_batch_features, n_network->num_neurons_in_layer[0], n_network->activation, n_network->batch_size);

    // Allocate memory for hidden layers
    for (int i = 1; i < n_network->num_layers - 1; i++) {
        n_network->layers[i] = init_layer(n_network->layers[i-1]->num_neurons, n_network->num_neurons_in_layer[i], 
                                          n_network->activation, n_network->batch_size);
    }

    // Allocate memory for the output layer
    n_network->layers[num_layers - 1] = init_layer(n_network->layers[num_layers - 2]->num_neurons, n_network->num_neurons_in_layer[num_layers - 1], 
                                                   SOFTMAX, n_network->batch_size);

    return n_network;
}

void free_neural_network(NeuralNetwork* network) {
    for (int i = 0; i < network->num_layers; i++) {
        free_layer(network->layers[i]);
    }
    free(network->loss_history);
    free(network->layers);
    free(network);
}

void print_nn_info(NeuralNetwork* network) {
    // Print network constraints
    printf("NUMBER OF LAYERS: %d\n", network->num_layers);
    printf("BATCH SIZE: %d\n", network->batch_size);
    printf("NUMBER OF EPOCHS: %d\n", network->num_epochs);
    printf("LEARNING RATE: %f\n", network->learning_rate);
}

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

void backward_pass_nn(NeuralNetwork* network, matrix* y_pred) {

    // Start with the backward pass for softmax and loss
    backwards_softmax_and_loss(y_pred, network->layers[network->num_layers-1]);

    // Go backwards through every hidden layer
    for (int i = network->num_layers - 2; i >= 0; i--) {
        backward_reLu(network->layers[i+1]->dinputs, network->layers[i]);
    }
    
}

void update_parameters(NeuralNetwork* network) {
    // Loop through the first and all hidden layers
    for (int i = 0; i < network->num_layers - 1; i++) {
        // Update weights and biases for each layer(currently uses SGD)
        update_params_sgd(network->layers[i], network->learning_rate);
    }
}

void train_nn(NeuralNetwork* network, matrix* X, matrix* Y) {

    for (int epoch = 0; epoch < network->num_epochs; epoch++) {

        // Step 1: Forward Pass
        forward_pass_nn(network, X);

        // Step 2: Calculate Loss and Accuracy
        double accuracy = calculate_accuracy(Y, network->layers[network->num_layers-1], ONE_HOT);
        matrix* example_losses = loss_categorical_cross_entropy(Y, network->layers[network->num_layers-1], ONE_HOT);

        // calculate batch loss
        double batch_loss = 0.0;
        for (int i = 0; i < network->batch_size; i++) {
            batch_loss+= example_losses->data[i];
        }
        batch_loss = batch_loss/network->batch_size;

        // Add loss to the loss history
        network->loss_history[epoch] = batch_loss;

        // Print training data
        printf("Epoch %d: Loss = %f, Accuracy = %f\n", epoch, batch_loss, accuracy);

        // Step 3: Backward Pass
        backward_pass_nn(network, Y);

        // Step 4: Update Weights
        update_parameters(network);
    }
}

void predict(NeuralNetwork* network, matrix* input_data) {
    // Check if the input data dimensions match the expected input size
    if (input_data->dim2 != network->layers[0]->num_inputs) {
        printf("Error: Input data dimension does not match network input size.\n");
        return;
    }

    // Perform a forward pass through the network for prediction
    forward_pass_nn(network, input_data);

    // After the forward pass, output should contain the network's predictions
    matrix* output = network->layers[network->num_layers - 1]->post_activation_output;

    // Apply softmax to the output layer to interpret the logits as probabilities
    forward_softMax(output);

    // Find the index of the class with the highest probability
    int predicted_class = 0;
    double max_prob = output->data[0];
    for (int i = 1; i < output->dim2; i++) {
        if (output->data[i] > max_prob) {
            max_prob = output->data[i];
            predicted_class = i;
        }
    }

    // Print the predicted class
    printf("Predicted class: %d (Probability: %f)\n", predicted_class, max_prob);
}
