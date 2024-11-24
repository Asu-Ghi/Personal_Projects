#include "network.h"



NeuralNetwork* init_neural_network(int num_layers, int* num_neurons_in_layer, double learning_rate,
                                   ActivationType* activations, OptimizationType* optimizations, bool* regularizations, int num_batch_features) {

    // Allocate memory for the network
    NeuralNetwork* n_network = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));

    // Initialize network parameters
    n_network->num_layers = num_layers;
    n_network->num_features = num_batch_features;
    n_network->learning_rate = learning_rate;
    n_network->decay_rate = 0.0; 
    n_network->num_epochs = 150; // by default
    n_network->activations_per_layer = activations; 
    n_network->optimizations_per_layer = optimizations;
    n_network->regularizations_per_layer = regularizations;
    n_network->num_neurons_in_layer = num_neurons_in_layer;
    n_network->current_epoch = 100; 
    n_network->momentum = false; 
    n_network->beta_1 = 0.90; // initializes to 0.9 -> Used for Momentum
    n_network->beta_2 = 0.999; // initializes to 0.999 -> Used for Cachce
    n_network->epsilon = 1e-7; // epsilon(ADA GRAD, RMSPROP)
    n_network->debug = true; // init to true
    n_network->accuracy = 0.0; // init to 0
    n_network->loss = 0.0; // init to 0
    n_network->useBiasCorrection = true; // Set by default, set to false if ADAM not performing well

    // Allocate memory for layers
    n_network->layers = (layer_dense**) malloc(n_network->num_layers * sizeof(layer_dense*));

    // Init dropout rates to 0 on start
    n_network->drop_out_per_layer = (double*) calloc(n_network->num_layers, sizeof(double));

    // Allocate memory for the first layer with `num_features`
    n_network->layers[0] = init_layer(num_batch_features, n_network->num_neurons_in_layer[0], 
                                    n_network->activations_per_layer[0], n_network->optimizations_per_layer[0]);

    // Adjust layers "useRegularization" variable
    n_network->layers[0]->useRegularization = regularizations[0];

    // Allocate memory for hidden layers
    for (int i = 1; i < n_network->num_layers - 1; i++) {
        n_network->layers[i] = init_layer(n_network->layers[i-1]->num_neurons, n_network->num_neurons_in_layer[i], 
                                     n_network->activations_per_layer[i], n_network->optimizations_per_layer[i]);
        // Adjust layers "useRegularization" variable
        n_network->layers[i]->useRegularization = regularizations[i];

    }

    // Allocate memory for the output layer
    n_network->layers[num_layers - 1] = init_layer(n_network->layers[num_layers - 2]->num_neurons, n_network->num_neurons_in_layer[num_layers - 1], 
                                    n_network->activations_per_layer[num_layers-1], n_network->optimizations_per_layer[num_layers-1]);

    // Adjust layers "useRegularization" variable
    n_network->layers[num_layers-1]->useRegularization = regularizations[num_layers-1];

    // Return network object
    return n_network;
}

void free_neural_network(NeuralNetwork* network) {
    for (int i = 0; i < network->num_layers; i++) {
        free_layer(network->layers[i]);
    }
    network->loss = 0.0;
    network->accuracy = 0.0;
    network->current_epoch = 0;
    free(network->loss_history);
    free(network->layers);
    free(network->drop_out_per_layer);
    free(network);
    network = NULL;
}

void print_nn_info(NeuralNetwork* network) {
    // Print network constraints
    printf("#############################################\n");
    printf("#############################################\n");
    printf("NUMBER OF LAYERS: %d\n", network->num_layers);
    printf("BATCH SIZE: %d\n", network->batch_size);
    printf("NUMBER OF INPUT FEATURES: %d\n",network->num_features);
    printf("NUMBER OF EPOCHS: %d\n", network->num_epochs);
    printf("LEARNING RATE: %f\n", network->learning_rate);
    printf("DECAY RATE(RHO): %f\n", network->decay_rate);
    printf("BETA_1 (MOMENTUMS): %f\n", network->beta_1);
    printf("BETA_2 (RMSPROP CACHES): %f\n", network->beta_2);
    printf("EPSILON: %.8f\n", network->epsilon);

    if (network->useBiasCorrection){
        printf("USING BIAS CORRECTION(ADAM)\n");
    }
    else {
        printf("NOT USING BIAS CORRECTION(ADAM)\n");
    }

    printf("#############################################\n");
    printf("#############################################\n");
}

void forward_pass_nn(NeuralNetwork* network, matrix* inputs) {

    // First forward pass
    forward_pass(inputs, network->layers[0]);

    // Forward pass for hidden layers
    for(int i = 1; i < network->num_layers; i++) {
        forward_pass(network->layers[i - 1]->post_activation_output, network->layers[i]);
    }

    // Last forward pass
    forward_pass(network->layers[network->num_layers - 2]->post_activation_output, network->layers[network->num_layers - 1]);
}

void pred_forward_pass_nn(NeuralNetwork* network, matrix* inputs) {

    // First forward pass
    pred_forward_pass(inputs, network->layers[0]);

    // Forward pass for hidden layers
    for(int i = 1; i < network->num_layers; i++) {
        pred_forward_pass(network->layers[i - 1]->pred_outputs, network->layers[i]);
    }

    // Last forward pass
    pred_forward_pass(network->layers[network->num_layers - 2]->pred_outputs, network->layers[network->num_layers - 1]);
}

void backward_pass_nn(NeuralNetwork* network, matrix* y_pred) {

    // Start with the backward pass for softmax and loss
    if (network->layers[network->num_layers-1]->activation == SOFTMAX) {
        backwards_softmax_and_loss(y_pred, network->layers[network->num_layers-1]);

        // test later. apply to loop

        // free(network->layers[network->num_layers-1]->post_activation_output->data);
        // free(network->layers[network->num_layers-1]->post_activation_output);

        // free(network->layers[network->num_layers-1]->pre_activation_output->data);
        // free(network->layers[network->num_layers-1]->pre_activation_output);

        // free(network->layers[network->num_layers-1]->inputs->data);
        // free(network->layers[network->num_layers-1]->inputs);
    }
    else {
        printf("Not handled.\n");
        exit(1);
    }

    // Go backwards through every hidden layer
    // Free outputs for the layer
    for (int i = network->num_layers - 2; i >= 0; i--) {

        if (network->layers[i]->activation == RELU){
            backward_reLu(network->layers[i+1]->dinputs, network->layers[i]);

            // free(network->layers[i]->post_activation_output->data);
            // free(network->layers[i]->post_activation_output);

            // free(network->layers[i]->pre_activation_output->data);
            // free(network->layers[i]->pre_activation_output);

            // free(network->layers[i]->inputs->data);
            // free(network->layers[i]->inputs);
        }
        else {
            printf("Not handled.\n");
            exit(1);
        }
    }
    
}

void update_parameters(NeuralNetwork* network) {
    // Loop through the first and all hidden layers
    for (int i = 0; i < network->num_layers; i++) {
        // Optimize layer
        optimization_dense(network->layers[i], &network->learning_rate, network->decay_rate, network->current_epoch,
                            network->beta_1, network->beta_2, network->epsilon, network->useBiasCorrection);
    }   
}

void train_nn(NeuralNetwork* network, int num_epochs, matrix* X, matrix* Y, matrix* X_validate, matrix* Y_validate) {

    // Set num epochs
    network->num_epochs = num_epochs;
    // Allocate memory for loss_history
    network->loss_history = (double*) calloc(network->num_epochs, sizeof(double));

    // Print layer optimizations (if debug = true)
    if (network->debug) {
        for (int i = 0; i < network->num_layers; i++) {
            printf("Layer: %d, Optimization: %s, Activation: %s\n", i, 
            optimization_type_to_string(network->layers[i]->optimization), activation_type_to_string(network->layers[i]->activation));
        }
    }
    // calculate batch loss
    double batch_loss = 0.0;
    // calculate accuracy
    double accuracy = 0.0;
    // validate loss and accuracy
    double val_loss, val_accuracy;
    // best val loss for training
    double best_val_loss = DBL_MAX;

    // DEBUGING TIME
    double backward_time = 0.0;
    double forward_time = 0.0;
    double optimization_time = 0.0;
    double accuracy_time = 0.0;
    double loss_time = 0.0;
    double regularization_time = 0.0;
    double validate_time = 0.0;
    // Epoch Iterations
    for (int epoch = 0; epoch < network->num_epochs; epoch++) {
        // reset batch loss
        batch_loss = 0.0;
        // reset accuracy
        accuracy = 0.0;

        // Step 1: Forward Pass
        #ifdef ENABLE_PARALLEL
        double forward_start_time = omp_get_wtime();
        #endif
        forward_pass_nn(network, X);

        #ifdef ENABLE_PARALLEL
        double forward_end_time = omp_get_wtime();
        forward_time += (forward_end_time - forward_start_time);
        #endif

        // Step 2: Calculate Loss and Accuracy
        #ifdef ENABLE_PARALLEL
        double accuracy_start_time = omp_get_wtime();
        #endif

        accuracy = calculate_accuracy(Y, network->layers[network->num_layers-1], ONE_HOT);

        #ifdef ENABLE_PARALLEL
        double accuracy_end_time = omp_get_wtime();
        accuracy_time += (accuracy_end_time - accuracy_start_time);
        #endif

        #ifdef ENABLE_PARALLEL
        double loss_start_time = omp_get_wtime();
        #endif
        matrix* example_losses = loss_categorical_cross_entropy(Y, network->layers[network->num_layers-1], ONE_HOT);


        for (int i = 0; i < Y->dim1; i++) {
            batch_loss+= example_losses->data[i];
        }
        batch_loss = batch_loss/Y->dim1;

        #ifdef ENABLE_PARALLEL
        double loss_end_time = omp_get_wtime();
        loss_time += (loss_end_time - loss_start_time);
        #endif

        // Calculate regularization for l1 and l2 
        #ifdef ENABLE_PARALLEL
        double regularization_start_time = omp_get_wtime();
        #endif

        double regularization_val = 0.0;
        for (int i = 0; i < network->num_layers; i++) {
            regularization_val += calculate_regularization_loss(network->layers[i]);
        } 

        #ifdef ENABLE_PARALLEL
        double regularization_end_time = omp_get_wtime();
        regularization_time += (regularization_end_time - regularization_start_time);
        #endif

        // Sum regularization to loss
        batch_loss += regularization_val;

        // Add loss to the loss history
        network->loss_history[epoch] = batch_loss;

        // Step 3: Backward Pass
        #ifdef ENABLE_PARALLEL
        double backward_start_time = omp_get_wtime();
        #endif

        backward_pass_nn(network, Y);

        #ifdef ENABLE_PARALLEL
        double backward_end_time = omp_get_wtime();
        backward_time += (backward_end_time - backward_start_time);
        #endif

        // Step 4: Update Weights

        #ifdef ENABLE_PARALLEL
        double optimization_start_time = omp_get_wtime();
        #endif
        update_parameters(network);

        #ifdef ENABLE_PARALLEL
        double optimization_end_time = omp_get_wtime();
        optimization_time += (optimization_end_time - optimization_start_time);
        #endif

        // Step 5: Update current epoch (for learning rate decay)
        network->current_epoch += 1;
        
        // Update loss and accuracy of network
        network->loss = batch_loss;
        network->accuracy = accuracy;

        // Validate Network every 50 epochs
        if (epoch % 50 == 0) {
            #ifdef ENABLE_PARALLEL
            double validate_start_time = omp_get_wtime();
            #endif

            validate_model(network, X_validate, Y_validate, &val_loss, &val_accuracy);
            network->val_accuracy = val_accuracy;
            network->val_loss = val_loss;

            #ifdef ENABLE_PARALLEL
            double validate_end_time = omp_get_wtime();
            validate_time += (validate_end_time - validate_start_time);
            #endif

        }
        
        // Print training data (if debug = TRUE)
        if (network->debug) {
            printf("Epoch %d: Model Loss = %f, Regularization Loss = %f, Model Accuracy = %f, LR = %f \n", epoch, batch_loss, 
                                regularization_val, accuracy, network->learning_rate);
            printf("Validate Loss = %f, Validate Accuracy = %f\n", val_loss, val_accuracy);
        }

        // Free temp memory
        free(example_losses->data);
        free(example_losses);
    }
    // Print Final Accuracy
    printf("Epoch %d: Loss = %f, Accuracy = %f, LR = %f \n", network->current_epoch, batch_loss, accuracy, network->learning_rate);
    
    // Print time debugs
    #ifdef ENABLE_PARALLEL
    printf("forward time = %f\n", forward_time / network->num_epochs);
    printf("validate time = %f\n", validate_time / (network->num_epochs / 50));
    printf("accuracy time = %f\n", accuracy_time / network->num_epochs);
    printf("loss time = %f\n", loss_time / network->num_epochs);
    printf("backward time = %f\n", backward_time / network->num_epochs);
    printf("regularization loss time = %f\n", regularization_time / network->num_epochs);
    printf("optimization time = %f\n", optimization_time / network->num_epochs);
    #endif
}

void predict(NeuralNetwork* network, matrix* input_data) {
    // Check if the input data dimensions match the expected input size
    if (input_data->dim2 != network->layers[0]->num_inputs) {
        fprintf(stderr, "Error: Input data dimension does not match network input size.\n");
        printf("(%d x %d) != (%d x %d)\n", 
                    input_data->dim1, input_data->dim2, network->layers[0]->num_inputs, network->layers[0]->num_neurons);
        return;
    }
    // Perform a forward pass through the network for prediction    
    forward_pass_nn(network, input_data);

    // After the forward pass, output should contain the network's predictions
    matrix* output = network->layers[network->num_layers - 1]->post_activation_output;

    // Apply softmax to the output layer to interpret the logits as probabilities
    forward_softMax(output);

    // Find the index of the class with the highest probability
    int predicted_class = -1;
    double max_prob = output->data[0];
    print_matrix(network->layers[network->num_layers-1]->post_activation_output);
    matrix* outputs = network->layers[network->num_layers-1]->post_activation_output;

    double max_pred = 0.0;
    for (int i = 0; i < outputs->dim1; i++) {
        printf("Sample = %d.\n ", i);
        max_pred = 0.0;
        predicted_class = -1;
        for (int j = 0; j < outputs->dim2; j++) {
            if (outputs->data[i * outputs->dim2 + j] > max_pred) {
                max_pred = outputs->data[i * outputs->dim2 + j];
                predicted_class = j;
            }
        }
        printf("Predicted Class = %d, Probability = %f\n", predicted_class, max_pred);
    }

    // Print the predicted class

}

void validate_model(NeuralNetwork* network, matrix* validate_data, matrix* validate_pred, double* loss, double* accuracy) {
    // Perform a forward pass on validate data
    pred_forward_pass_nn(network, validate_data);

    // Get loss
    double batch_loss = 0.0;
    matrix* example_losses = pred_loss_categorical_cross_entropy(validate_pred, network->layers[network->num_layers-1], ONE_HOT);

    // Sum all example losses
    for (int i = 0; i < validate_data->dim1; i++) {
        batch_loss+= example_losses->data[i];
    }

    // Get average loss
    *loss = batch_loss / validate_data->dim1;

    // Get Accuracy
    *accuracy = pred_calculate_accuracy(validate_pred, network->layers[network->num_layers-1], ONE_HOT);
} 

