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
    n_network->num_epochs = 0; // by default
    n_network->activations_per_layer = activations; 
    n_network->optimizations_per_layer = optimizations;
    n_network->regularizations_per_layer = regularizations;
    n_network->num_neurons_in_layer = num_neurons_in_layer;
    n_network->current_epoch = 0; 
    n_network->momentum = false; 
    n_network->beta_1 = 0.90; // initializes to 0.9 -> Used for Momentum
    n_network->beta_2 = 0.90; // initializes to 0.999 -> Used for Cachce
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

void free_layers_memory(NeuralNetwork* network) {
    for (int i = 0; i < network->num_layers; i++) {
        free_memory(network->layers[i]);
    }
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
        optimization_dense(network->layers[i], &network->learning_rate, network->decay_rate, network->current_epoch, network->num_epochs,
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
    // calc regularization loss
    double regularization_loss = 0.0;
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
        // reset regularization loss
        regularization_loss = 0.0;

        // Step 1: Forward Pass
        double forward_start_time = omp_get_wtime();
        forward_pass_nn(network, X);
        double forward_end_time = omp_get_wtime();
        forward_time += (forward_end_time - forward_start_time);

        // Step 2: Calculate Loss and Accuracy
        double accuracy_start_time = omp_get_wtime();
        accuracy = calculate_accuracy(Y, network->layers[network->num_layers-1], ONE_HOT);
        double accuracy_end_time = omp_get_wtime();
        accuracy_time += (accuracy_end_time - accuracy_start_time);
    

        double loss_start_time = omp_get_wtime();
        matrix* example_losses = loss_categorical_cross_entropy(Y, network->layers[network->num_layers-1], ONE_HOT);
        for (int i = 0; i < Y->dim1; i++) {
            batch_loss+= example_losses->data[i];
        }
        batch_loss = batch_loss/Y->dim1;
        example_losses = NULL;
        double loss_end_time = omp_get_wtime();
        loss_time += (loss_end_time - loss_start_time);

        // Free examples losses
        free(example_losses->data);
        free(example_losses);       

        // Calculate regularization for l1 and l2 
        double regularization_start_time = omp_get_wtime();
        for (int i = 0; i < network->num_layers; i++) {
            regularization_loss += calculate_regularization_loss(network->layers[i]);
        } 
        double regularization_end_time = omp_get_wtime();
        regularization_time += (regularization_end_time - regularization_start_time);

        // Step 3: Backward Pass
        double backward_start_time = omp_get_wtime();
        backward_pass_nn(network, Y);
        double backward_end_time = omp_get_wtime();
        backward_time += (backward_end_time - backward_start_time);

        // Step 4: Update Weights

        double optimization_start_time = omp_get_wtime();
        update_parameters(network);
        double optimization_end_time = omp_get_wtime();
        optimization_time += (optimization_end_time - optimization_start_time);

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
            printf("Epoch %d: Loss = %f (data_loss: %f, reg_loss: %f), Accuracy = %f, LR = %f \n", network->current_epoch, batch_loss+regularization_loss,
                        batch_loss, regularization_loss, accuracy, network->learning_rate);
            printf("Validate Loss = %f, Validate Accuracy = %f\n", val_loss, val_accuracy);
        }

        // Sum regularization to loss
        batch_loss += regularization_loss;

        // Add loss to the loss history
        network->loss_history[epoch] = batch_loss;

        // Free temp memory
        free(example_losses->data);
        free(example_losses);

        // Free undeeded memory in layers
        free_layers_memory(network);

    }
    // Print Final Accuracy
    printf("Epoch %d: Loss = %f , Accuracy = %f, LR = %f \n", network->current_epoch, batch_loss,
            accuracy, network->learning_rate);
    
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

void validate_model(NeuralNetwork* network, matrix* validate_data, matrix* validate_pred, double* loss, double* accuracy) {
    // Perform a forward pass on validate data
    // pred_forward_pass_nn(network, validate_data);
    pred_forward_pass_nn(network, validate_data);
    // Get loss
    double batch_loss = 0.0;
    // matrix* example_losses = pred_loss_categorical_cross_entropy(validate_pred, network->layers[network->num_layers-1], ONE_HOT);
    matrix* example_losses = pred_loss_categorical_cross_entropy(validate_pred, network->layers[network->num_layers-1], ONE_HOT);

    // Sum all example losses
    for (int i = 0; i < validate_data->dim1; i++) {
        batch_loss+= example_losses->data[i];
    }

    // Get average loss
    *loss = batch_loss / validate_data->dim1;

    // Get Accuracy
    // *accuracy = pred_calculate_accuracy(validate_pred, network->layers[network->num_layers-1], ONE_HOT);
    *accuracy = pred_calculate_accuracy(validate_pred, network->layers[network->num_layers-1], ONE_HOT);

    // Free examples losses
    free(example_losses->data);
    free(example_losses);
    example_losses = NULL;
} 

void export_params(NeuralNetwork* network, char* dir_path) {
    // Create file paths for parameters
    char w_file_path[0xFF];
    char b_file_path[0xFF];

    strcpy(w_file_path, dir_path);
    stpcpy(b_file_path, dir_path);

    // Add on identifier for file
    strcat(w_file_path, "/weights.csv\n");
    strcat(b_file_path, "/biases.csv\n");

    // Open files for writing
    FILE* w_file = fopen(w_file_path, "w");
    FILE* b_file = fopen(b_file_path, "w");

    if (w_file == NULL || b_file == NULL) {
        perror("Error opening file");
        exit(1);
    }

    // Write out weights and biases
    for (int i = 0; i < network->num_layers; i++) {
        int rows = network->layers[i]->weights->dim1;
        int cols = network->layers[i]->weights->dim2;

        // Write weights to file
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {
                if (fprintf(w_file, "%.10f", network->layers[i]->weights->data[j * cols + k]) < 0) {
                    perror("Error writing weights");
                }
                if (k < cols - 1) {
                    fprintf(w_file, ",");
                }
            }
            fprintf(w_file, "\n");
        }

        // Write biases to file
        cols = network->layers[i]->biases->dim2;
        for (int j = 0; j < cols; j++) {
            if (fprintf(b_file, "%.10f", network->layers[i]->biases->data[j]) < 0) {
                perror("Error writing biases");
            }
            if (j < cols - 1) {
                fprintf(b_file, ",");
            }
        }
        fprintf(b_file, "\n");
    }

    printf("Saved Network Params..\n");
    // Close files
    fclose(w_file);
    fclose(b_file);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void load_params(NeuralNetwork* network, char* dir_path) {

    // Create file path for weights and biases
    char w_file_path[0xFFF];
    char b_file_path[0xFFF];

    strcpy(w_file_path, dir_path);
    strcpy(b_file_path, dir_path);

    strcat(w_file_path, "/weights.csv");
    strcat(b_file_path, "/biases.csv");

    // Open files for reading
    FILE* w_file = fopen(w_file_path, "r");
    FILE* b_file = fopen(b_file_path, "r");

    if (w_file == NULL || b_file == NULL) {
        perror("Error opening file");
        exit(1);
    }

    // Load Weights and Biases
    for (int i = 0; i < network->num_layers; i++) {

        // Get dimenstionality info 
        int rows = network->layers[i]->weights->dim1;
        int cols = network->layers[i]->weights->dim2;

        // Ensure layer params are free
        if (network->layers[i]->weights != NULL) {
            free(network->layers[i]->weights->data);
            free(network->layers[i]->weights);
        }
        if (network->layers[i]->biases != NULL) {
            free(network->layers[i]->biases->data);
            free(network->layers[i]->biases);
        }       

        // Reallocate memory
        network->layers[i]->weights = malloc(sizeof(matrix));
        network->layers[i]->weights->dim1 = rows; // Num Inputs
        network->layers[i]->weights->dim2 = cols; // Num Neurons
        network->layers[i]->weights->data = (double*) calloc(rows * cols, sizeof(double));

        network->layers[i]->biases = malloc(sizeof(matrix));
        network->layers[i]->biases->dim1 = 1;
        network->layers[i]->biases->dim2 = cols; // Num Neurons
        network->layers[i]->biases->data = (double*) calloc(1 * cols, sizeof(double));

        // Load weights from file
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {
                if (fscanf(w_file, "%lf", &network->layers[i]->weights->data[j * cols + k]) != 1) {
                    printf("Error reading weights\n");
                    fclose(w_file);
                    fclose(b_file);
                    exit(1);
                }
                if (k < cols - 1) {
                    fscanf(w_file, ","); // Skip comma
                }
            }
            fscanf(w_file, "\n"); // Skip newline
        }

        // Load biases from file
        cols = network->layers[i]->biases->dim2;
        for (int j = 0; j < cols; j++) {
            if (fscanf(b_file, "%lf", &network->layers[i]->biases->data[j]) != 1) {
                printf("Error reading biases\n");
                fclose(w_file);
                fclose(b_file);
                exit(1);
            }
            if (j < cols - 1) {
                fscanf(b_file, ","); // Skip comma
            }
        }
        fscanf(b_file, "\n"); // Skip newline
    }

    printf("Loaded Network Params..\n");

    // Close files
    fclose(w_file);
    fclose(b_file);
}


