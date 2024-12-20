#include "network.h"
#define DIR_PATH "results/params/Model_1"
////////////////////////////////////////////////// MISC METHODS ///////////////////////////////////////////////////////////////////////////

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
    n_network->beta_1 = 0.90; // initializes to 0.9 -> Used for Momentum
    n_network->beta_2 = 0.90; // initializes to 0.999 -> Used for Cachce
    n_network->epsilon = 1e-7; // epsilon(ADA GRAD, RMSPROP)
    n_network->debug = true; // init to true
    n_network->accuracy = 0.0; // init to 0
    n_network->loss = 0.0; // init to 0
    n_network->useBiasCorrection = true; // Set by default, set to false if ADAM not performing well
    n_network->useWeightDecay = false; // Set by default to false, used in ADAM mini batch training.
    n_network->early_stopping = false; // Set by default to be false.
    n_network->send_ratio = 50; // set to send socket data every 50 epochs by default.
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
    printf("Freeing network memory...\n");
    for (int i = 0; i < network->num_layers; i++) {
        free_layer(network->layers[i]);
    }
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

////////////////////////////////////////////////// FORWARD METHODS ///////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////// BACKWARD METHODS ///////////////////////////////////////////////////////////////////////////

void backward_pass_nn(NeuralNetwork* network, matrix* y_pred) {

    // Start with the backward pass for softmax and loss
    if (network->layers[network->num_layers-1]->activation == SOFTMAX) {
        backwards_softmax_and_loss(y_pred, network->layers[network->num_layers-1]);
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

        else if (network->layers[i]->activation == LINEAR) {
            backward_linear(network->layers[i+1]->dinputs, network->layers[i]);
        }

        else {
            printf("Not handled.\n");
            exit(1);
        }
    }
    
}

////////////////////////////////////////////////// OPTIMIZER METHODS ///////////////////////////////////////////////////////////////////////////

void update_parameters(NeuralNetwork* network) {
    // Loop through the first and all hidden layers
    for (int i = 0; i < network->num_layers; i++) {
        // Optimize layer
        optimization_dense(network->layers[i], &network->learning_rate, network->decay_rate, network->current_epoch, network->num_epochs,
                            network->beta_1, network->beta_2, network->epsilon, network->useBiasCorrection, network->useWeightDecay);
    }   
}

void update_learning_rate(NeuralNetwork* network) {
    // Apply learning rate decay (if decay factor is specified)
    if (network->decay_rate > 0.0) {
        network->learning_rate = fmax(network->learning_rate * (1 / (1 + network->decay_rate * network->current_epoch)), 1e-6);
    }
}
/////////////////////////////////////////////// TRAINING/TESTING METHODS ///////////////////////////////////////////////////////////////////////////

void train_full_batch(NeuralNetwork* network, int num_epochs, Training_Data* training_data) {

    // Set num epochs
    network->num_epochs = num_epochs;

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
    // best val acc for training
    double best_val_acc = 0;
    // Early stopping variables
    double prev_val_loss = 0; // Previous validate loss
    int max_time = 5; // Maximum number of validate iterations to wait before seeing loss improvements
    int current_time = 0; // Current wait iteration

    // DEBUGING TIME
    double backward_time = 0.0;
    double forward_time = 0.0;
    double optimization_time = 0.0;
    double accuracy_time = 0.0;
    double loss_time = 0.0;
    double regularization_time = 0.0;
    double validate_time = 0.0;

    // Setup the socket once
    #ifdef ENABLE_SOCKET
    printf("Setting up socket...\n");
    int sockfd = setup_socket();
    #endif

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
        forward_pass_nn(network, training_data->X);
        double forward_end_time = omp_get_wtime();
        forward_time += (forward_end_time - forward_start_time);

        // Step 2: Calculate Loss and Accuracy
        double accuracy_start_time = omp_get_wtime();
        accuracy = calculate_accuracy(training_data->Y, network->layers[network->num_layers-1]);
        double accuracy_end_time = omp_get_wtime();
        accuracy_time += (accuracy_end_time - accuracy_start_time);
    
        double loss_start_time = omp_get_wtime();
        batch_loss = loss_categorical_cross_entropy(training_data->Y, network->layers[network->num_layers-1]);
        double loss_end_time = omp_get_wtime();
        loss_time += (loss_end_time - loss_start_time);

        // Calculate regularization for l1 and l2 
        double regularization_start_time = omp_get_wtime();
        for (int i = 0; i < network->num_layers; i++) {
            regularization_loss += calculate_regularization_loss(network->layers[i]);
        } 

        double regularization_end_time = omp_get_wtime();
        regularization_time += (regularization_end_time - regularization_start_time);

        // Step 3: Backward Pass
        double backward_start_time = omp_get_wtime();
        backward_pass_nn(network, training_data->Y);
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
        network->reg_loss = regularization_loss;

        // Validate Network every 50 epochs
        if (epoch % 50 == 0) {
            free_layers_memory(network); // Free uneeded memory before validation forward pass

            double validate_start_time = omp_get_wtime();
            validate_model(network, training_data->X_pred, training_data->Y_pred, &val_loss, &val_accuracy); // Forward pass for validation
            network->val_accuracy = val_accuracy; // Update network validation accuracy
            network->val_loss = val_loss; // Update network validation loss

            // Check to update best loss
            if (network->val_loss < best_val_loss) {
                best_val_loss = network->val_loss;
            }
            if (network->val_accuracy > best_val_acc) {
                best_val_acc = network->val_accuracy;
            }

            // Check for improvement (ratio between previous and current loss greater than factor f)
            if (prev_val_loss/network->val_loss < 1.1 && network->early_stopping) {
                current_time += 1;
                if (current_time > max_time) {
                    printf("Early Stopping Training.\n");
                    // Send the final data before exiting to ensure the plots don't reset
                    #ifdef ENABLE_SOCKET
                    send_data(network, sockfd);
                    #endif
                    
                    // Send final validation loss and accuracy data (for plotting)
                    #ifdef ENABLE_SOCKET
                    send_data(network, sockfd);
                    usleep(100000); // Sleep for 100 ms (or adjust as needed)
                    close(sockfd); // Ensure socket is closed before exiting  
                    #endif
                    exit(0);
              }
            }
            else {
                current_time = 0; // Reset stop time.
            }

            // Update current validate loss
            prev_val_loss = network->val_loss;
            double validate_end_time = omp_get_wtime();
            validate_time += (validate_end_time - validate_start_time);
        }

        // Send data every 50 epochs
        #ifdef ENABLE_SOCKET
        if (epoch % network->send_ratio == 0) {
            send_data(network, sockfd);
        }        
        #endif

        if (network->debug) {
            printf("Epoch %d: Loss = %f (data_loss: %f, reg_loss: %f), Accuracy = %f, LR = %f \n", network->current_epoch, 
                    network->loss + network->reg_loss, network->loss, network->reg_loss, network->accuracy, network->learning_rate);
            printf("Validate Loss = %f, Validate Accuracy = %f\n", network->val_loss, network->val_accuracy);
        }

        // Update lr
        update_learning_rate(network);

        // Free undeeded memory in layers
        free_layers_memory(network);
    }

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

    // Close socket
    #ifdef ENABLE_SOCKET
    close(sockfd);
    #endif

}

void train_mini_batch(NeuralNetwork* network, int num_epochs, int batch_size, Training_Data* training_data) {
    // Set num epochs
    network->num_epochs = num_epochs;
    // Adam Weight Decay
    network->useWeightDecay = true;
    // Adjust Lr based off mini batch size
    network->learning_rate *= ((double)batch_size / training_data->X->dim1);

    // Check to see if batch size cleanly divides num of samples in training batch
    if (training_data->X->dim1 % batch_size != 0) {
        fprintf(stderr, "Training Data indivisble by batch size.");
    }

    int batch_iterations = training_data->X->dim1 / batch_size;
    matrix* mini_X = malloc(sizeof(matrix));
    matrix* mini_Y = malloc(sizeof(matrix));

    // Setup the socket once
    #ifdef ENABLE_SOCKET
    printf("Setting up socket...\n");
    int sockfd = setup_socket();
    #endif

    // Iter every epoch
    for (int i = 0; i < num_epochs; i++) {  

        // Iter every mini batch
        double batch_loss = 0.0;
        double batch_accuracy = 0.0;
        double reg_loss = 0.0;
        for (int j = 0; j < batch_iterations; j++) {
            // Only correct bias once (ADAM)
            if (j == 0) {
                network->useBiasCorrection = true;
            }
            else {
                network->useBiasCorrection = false;
            }
            // Shallow copy mini batch data
            shallow_cpy_matrix(training_data->X, mini_X, j * batch_size, batch_size);            
            shallow_cpy_matrix(training_data->Y, mini_Y, j * batch_size, batch_size);
            
            // Forward Pass
            forward_pass_nn(network, mini_X);

            // Calculate Loss and Accuracy
            batch_loss += loss_categorical_cross_entropy(mini_Y, network->layers[network->num_layers - 1]);
            batch_accuracy += calculate_accuracy(mini_Y, network->layers[network->num_layers - 1]);

            for (int k = 0; k < network->num_layers; k++) {
                reg_loss += calculate_regularization_loss(network->layers[k]);
            }
            // Backward Pass
            backward_pass_nn(network, mini_Y);

            // Optimizations
            update_parameters(network);

            free_layers_memory(network);
        }


        update_learning_rate(network); // update lr

        // Update loss and accuracy after each batch iteration is complete
        network->loss = batch_loss / batch_iterations;
        network->accuracy = batch_accuracy / batch_iterations;
        network->reg_loss = reg_loss / batch_iterations;
        network->current_epoch = i;
        // Validate Model
        if (i % 5 == 0) {
            free_layers_memory(network); // Free network memory before validation
            validate_model(network, training_data->X_pred, training_data->Y_pred, &network->val_loss, &network->val_accuracy);
            free_layers_memory(network); // Free network memory after validation
        }
        // Print metrics
        if (network->debug) {
            printf("Epoch %d: Loss = %f (data_loss: %f, reg_loss: %f), Accuracy = %f, LR = %f \n", network->current_epoch, 
                    network->loss + network->reg_loss, network->loss, network->reg_loss, network->accuracy, network->learning_rate);
            printf("Validate Loss = %f, Validate Accuracy = %f\n", network->val_loss, network->val_accuracy);
        }   

        // Send Data

        #ifdef ENABLE_SOCKET
        send_data(network, sockfd);   
        #endif
    }

    #ifdef ENABLE_SOCKET
    close(sockfd);
    #endif
    free(mini_X);
    free(mini_Y);
}

void validate_model(NeuralNetwork* network, matrix* validate_data, matrix* validate_pred, double* loss, double* accuracy) {
    // Set validation flags for all layers
    for(int i = 0; i < network->num_layers; i++) {
        network->layers[i]->is_training = false;
    }

    // Perform a forward pass on validate data
    forward_pass_nn(network, validate_data);
    // Get loss
    *loss = loss_categorical_cross_entropy(validate_pred, network->layers[network->num_layers-1]);

    // Get Accuracy
    *accuracy = calculate_accuracy(validate_pred, network->layers[network->num_layers-1]);

    // Set training flags for all layers
    for(int i = 0; i < network->num_layers; i++) {
        network->layers[i]->is_training = true;
    }
} 

void predict_nn(NeuralNetwork* network, matrix* X) {

    // Check to see what type of activation is used for output
    if (network->layers[network->num_layers - 1]->activation == SOFTMAX) {
        exit(1);
    }
    if (network->layers[network->num_layers - 1]->activation == SIGMOID) {
        exit(1);
    }

}

///////////////////////////////////////////////// HYPERPARAM METHODS ////////////////////////////////////////////////////////////////////////////

void find_best_lr(NeuralNetwork* network, double init_lr, double lr_factor, int num_epochs) {
    /*
    Steps:
        > Break 
        > Train Network
        > Increase Lr by lr factor after every epoch
        > Send Data to Python
    */
}


///////////////////////////////////////////////// SAVE/LOAD METHODS ////////////////////////////////////////////////////////////////////////////

void export_params(NeuralNetwork* network, char* dir_path) {
    // Create file paths for parameters
    char w_file_path[0xFF]; // Layer Dense params
    char b_file_path[0xFF];

    char w_momentums_file_path[0xFF]; // Momentums
    char b_momentums_file_path[0xFF];

    char w_caches_file_path[0xFF]; // Caches
    char b_caches_file_path[0xFF];

    char hp_file_path[0xFF]; // Hyper params

    strcpy(w_file_path, dir_path);
    strcpy(b_file_path, dir_path);

    strcpy(w_momentums_file_path, dir_path);
    strcpy(b_momentums_file_path, dir_path);

    strcpy(w_caches_file_path, dir_path);
    strcpy(b_caches_file_path, dir_path);  

    strcpy(hp_file_path, dir_path);

    // Add on identifier for files
    strcat(w_file_path, "/weights.csv");
    strcat(b_file_path, "/biases.csv");

    strcat(w_momentums_file_path, "/weight_momentums.csv");
    strcat(b_momentums_file_path, "/bias_momentums.csv");

    strcat(w_caches_file_path, "/weight_caches.csv");
    strcat(b_caches_file_path, "/bias_caches.csv");

    strcat(hp_file_path, "/hyper_params.json");

    // Open files for writing
    FILE* w_file = fopen(w_file_path, "w");
    FILE* b_file = fopen(b_file_path, "w");

    if (w_file == NULL || b_file == NULL) {
        perror("Error opening file");
        exit(1);
    }

    FILE* wm_file = fopen(w_momentums_file_path, "w");
    FILE* bm_file = fopen(b_momentums_file_path, "w");

    if (wm_file == NULL || bm_file == NULL) {
        perror("Error opening file");
        exit(1);
    }

    FILE* wc_file = fopen(w_caches_file_path, "w");
    FILE* bc_file = fopen(b_caches_file_path, "w");

    if (wc_file == NULL || bc_file == NULL) {
        perror("Error opening file");
        exit(1);
    }

    // Save hyperparams first
    cJSON* json = cJSON_CreateObject(); // Json file to save hyperparams
    cJSON_AddNumberToObject(json, "learning_rate", network->learning_rate);
    cJSON_AddNumberToObject(json, "current_epoch", network->current_epoch);
    cJSON_AddNumberToObject(json, "beta1", network->beta_1);
    cJSON_AddNumberToObject(json, "beta2", network->beta_2);
    cJSON_AddNumberToObject(json, "epsilon", network->epsilon);
    cJSON_AddNumberToObject(json, "l2_lambda", network->layers[0]->lambda_l2); // same lambda l2 for all layers

    // Write JSON to file
    FILE* hyper_file = fopen(hp_file_path, "w");
    if (hyper_file == NULL) {
        perror("Error opening hyper param json file for writing");
        cJSON_Delete(json);
        return;
    }

    // Convert to JSON string
    char *json_string = cJSON_Print(json);


    // Check if JSON string was created
    if (json_string == NULL) {
        printf("Error creating JSON string\n");
        exit(1);
    }

    // Write to file
    fprintf(hyper_file, "%s\n", json_string);
    fclose(hyper_file);
    cJSON_Delete(json);


    // Write out weights biases and optimization params
    for (int i = 0; i < network->num_layers; i++) {
        int rows = network->layers[i]->weights->dim1;
        int cols = network->layers[i]->weights->dim2;

        // Write weight related params to file
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {
                if (fprintf(w_file, "%.10f", network->layers[i]->weights->data[j * cols + k]) < 0) {
                    perror("Error writing weights");
                }
                if (fprintf(wm_file, "%.10f", network->layers[i]->w_velocity->data[j * cols + k]) < 0) {
                    perror("Error writing weight momentums");
                }
                if (fprintf(wc_file, "%.10f", network->layers[i]->cache_weights->data[j * cols + k]) < 0) {
                    perror("Error writing weight caches");
                }
                if (k < cols - 1) {
                    fprintf(w_file, ",");
                    fprintf(wm_file, ",");
                    fprintf(wc_file, ",");
                }
            }
            fprintf(w_file, "\n");
            fprintf(wm_file, "\n");
            fprintf(wc_file, "\n");
        }

        // Write biase related params to file
        cols = network->layers[i]->biases->dim2;
        for (int j = 0; j < cols; j++) {
            if (fprintf(b_file, "%.10f", network->layers[i]->biases->data[j]) < 0) {
                perror("Error writing biases");
            }
            if (fprintf(bm_file, "%.10f", network->layers[i]->b_velocity->data[j]) < 0) {
                perror("Error writing bias momentums");
            }
            if (fprintf(bc_file, "%.10f", network->layers[i]->cache_bias->data[j]) < 0) {
                perror("Error writing bias caches");
            }
            if (j < cols - 1) {
                fprintf(b_file, ",");
                fprintf(bm_file, ",");
                fprintf(bc_file, ",");
            }
        }
        fprintf(b_file, "\n");
        fprintf(bm_file, "\n");
        fprintf(bc_file, "\n");
    }

    printf("Saved Network Params..\n");

    // Close files
    fclose(w_file);
    fclose(b_file);
    fclose(wm_file);
    fclose(bm_file);
    fclose(wc_file);
    fclose(bc_file);
}

void load_params(NeuralNetwork* network, char* dir_path) {

    // Create file path for weights and biases
    // Create file paths for parameters
    char w_file_path[0xFF];
    char b_file_path[0xFF];

    char w_momentums_file_path[0xFF];
    char b_momentums_file_path[0xFF];

    char w_caches_file_path[0xFF];
    char b_caches_file_path[0xFF];

    char hp_file_path[0xFF];

    strcpy(w_file_path, dir_path);
    stpcpy(b_file_path, dir_path);

    strcpy(w_momentums_file_path, dir_path);
    strcpy(b_momentums_file_path, dir_path);

    strcpy(w_caches_file_path, dir_path);
    strcpy(b_caches_file_path, dir_path);  

    strcpy(hp_file_path, dir_path);

    // Add on identifier for file
    strcat(w_file_path, "/weights.csv");
    strcat(b_file_path, "/biases.csv");

    strcat(w_momentums_file_path, "/weight_momentums.csv");
    strcat(b_momentums_file_path, "/bias_momentums.csv");

    strcat(w_caches_file_path, "/weight_caches.csv");
    strcat(b_caches_file_path, "/bias_caches.csv");

    strcat(hp_file_path, "/hyper_params.json");

    // Open files for reading

    FILE* w_file = fopen(w_file_path, "r");
    FILE* b_file = fopen(b_file_path, "r");
    if (w_file == NULL || b_file == NULL) {
        perror("Error opening file");
        exit(1);
    }

    FILE* wm_file = fopen(w_momentums_file_path, "r");
    FILE* bm_file = fopen(b_momentums_file_path, "r");
    if (wm_file == NULL || bm_file == NULL) {
        perror("Error opening file");
        exit(1);
    }

    FILE* wc_file = fopen(w_caches_file_path, "r");
    FILE* bc_file = fopen(b_caches_file_path, "r");
    if (wc_file == NULL || bc_file == NULL) {
        perror("Error opening file");
        exit(1);
    }

    // Load hyperparams
    FILE* hyper_file = fopen(hp_file_path, "r");
    if (hyper_file == NULL) {
        perror("Error opening file for reading");
        return;
    }

    fseek(hyper_file, 0, SEEK_END);
    long length = ftell(hyper_file);
    fseek(hyper_file, 0, SEEK_SET);

    char* buffer = (char*)malloc(length + 1);
    fread(buffer, 1, length, hyper_file);
    buffer[length] = '\0';
    fclose(hyper_file);

    cJSON* json = cJSON_Parse(buffer);
    if (!json) {
        printf("Error parsing JSON\n");
        free(buffer);
        return;
    }

    network->learning_rate = cJSON_GetObjectItem(json, "learning_rate")->valuedouble;
    network->current_epoch = cJSON_GetObjectItem(json, "current_epoch")->valueint;
    network->beta_1 = cJSON_GetObjectItem(json, "beta1")->valuedouble;
    network->beta_2 = cJSON_GetObjectItem(json, "beta2")->valuedouble;
    network->epsilon = cJSON_GetObjectItem(json, "epsilon")->valuedouble;

    // Iterate through every layer
    for (int i = 0; i < network->num_layers; i++) {
        // Set lambda from json
        network->layers[i]->lambda_l2 = cJSON_GetObjectItem(json, "l2_lambda")->valuedouble;   

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

        if (network->layers[i]->w_velocity != NULL) {
            free(network->layers[i]->w_velocity->data);
            free(network->layers[i]->w_velocity);        
        }   
        if (network->layers[i]->b_velocity != NULL) {
            free(network->layers[i]->b_velocity->data);
            free(network->layers[i]->b_velocity);        
        } 

        if (network->layers[i]->cache_weights != NULL) {
            free(network->layers[i]->cache_weights->data);
            free(network->layers[i]->cache_weights);        
        }             
        if (network->layers[i]->cache_bias != NULL) {
            free(network->layers[i]->cache_bias->data);
            free(network->layers[i]->cache_bias);        
        } 

        // Allocate Weight and Bias Memory
        network->layers[i]->weights = malloc(sizeof(matrix));
        network->layers[i]->weights->dim1 = rows; // Num Inputs
        network->layers[i]->weights->dim2 = cols; // Num Neurons
        network->layers[i]->weights->data = (double*) calloc(rows * cols, sizeof(double));

        network->layers[i]->biases = malloc(sizeof(matrix));
        network->layers[i]->biases->dim1 = 1;
        network->layers[i]->biases->dim2 = cols; // Num Neurons
        network->layers[i]->biases->data = (double*) calloc(1 * cols, sizeof(double));

        // Allocate Momentum memory
        network->layers[i]->w_velocity = malloc(sizeof(matrix));
        network->layers[i]->w_velocity->dim1 = rows;
        network->layers[i]->w_velocity->dim2 = cols; // Num Neurons
        network->layers[i]->w_velocity->data = (double*) calloc(rows * cols, sizeof(double));

        network->layers[i]->b_velocity = malloc(sizeof(matrix));
        network->layers[i]->b_velocity->dim1 = 1;
        network->layers[i]->b_velocity->dim2 = cols; // Num Neurons
        network->layers[i]->b_velocity->data = (double*) calloc(1 * cols, sizeof(double));

        // Allocate cache memory
        network->layers[i]->cache_weights = malloc(sizeof(matrix));
        network->layers[i]->cache_weights->dim1 = rows;
        network->layers[i]->cache_weights->dim2 = cols; // Num Neurons
        network->layers[i]->cache_weights->data = (double*) calloc(rows * cols, sizeof(double));

        network->layers[i]->cache_bias = malloc(sizeof(matrix));
        network->layers[i]->cache_bias->dim1 = 1;
        network->layers[i]->cache_bias->dim2 = cols; // Num Neurons
        network->layers[i]->cache_bias->data = (double*) calloc(1 * cols, sizeof(double));

        // Load weight related parameters from file
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {
                // Weights
                if (fscanf(w_file, "%lf", &network->layers[i]->weights->data[j * cols + k]) != 1) {
                    printf("Error reading weights\n");
                    fclose(w_file);
                    fclose(b_file);
                    fclose(wm_file);
                    fclose(wc_file);
                    fclose(bm_file);
                    fclose(bc_file);
                    exit(1);
                }
                // Weight Momentums
                if (fscanf(wm_file, "%lf", &network->layers[i]->w_velocity->data[j * cols + k]) != 1) {
                    printf("Error reading weight momenutms\n");
                    fclose(w_file);
                    fclose(b_file);
                    fclose(wm_file);
                    fclose(wc_file);
                    fclose(bm_file);
                    fclose(bc_file);
                    exit(1);
                }
                // Weight Caches
                if (fscanf(wc_file, "%lf", &network->layers[i]->cache_weights->data[j * cols + k]) != 1) {
                    printf("Error reading weights\n");
                    fclose(w_file);
                    fclose(b_file);
                    fclose(wm_file);
                    fclose(wc_file);
                    fclose(bm_file);
                    fclose(bc_file);
                    exit(1);
                }
                if (k < cols - 1) {
                    fscanf(w_file, ","); // Skip comma
                    fscanf(wm_file, ","); // Skip comma
                    fscanf(wc_file, ","); // Skip comma
                }
            }
            fscanf(w_file, "\n"); // Skip newline
            fscanf(wm_file, "\n"); // Skip newline
            fscanf(wc_file, "\n"); // Skip newline
        }

        // Load biase related parameters from file
        cols = network->layers[i]->biases->dim2;
        for (int j = 0; j < cols; j++) {
            if (fscanf(b_file, "%lf", &network->layers[i]->biases->data[j]) != 1) {
                printf("Error reading biases\n");
                fclose(w_file);
                fclose(b_file);
                fclose(wm_file);
                fclose(wc_file);
                fclose(bm_file);
                fclose(bc_file);
                exit(1);
            }
            if (fscanf(bm_file, "%lf", &network->layers[i]->b_velocity->data[j]) != 1) {
                printf("Error reading bias momentums\n");
                fclose(w_file);
                fclose(b_file);
                fclose(wm_file);
                fclose(wc_file);
                fclose(bm_file);
                fclose(bc_file);
                exit(1);
            }
            if (fscanf(bc_file, "%lf", &network->layers[i]->cache_bias->data[j]) != 1) {
                printf("Error reading bias caches\n");
                fclose(w_file);
                fclose(b_file);
                fclose(wm_file);
                fclose(wc_file);
                fclose(bm_file);
                fclose(bc_file);
                exit(1);
            }
            if (j < cols - 1) {
                fscanf(b_file, ","); // Skip comma
                fscanf(bm_file, ","); // Skip comma
                fscanf(bc_file, ","); // Skip comma
            }
        }
        fscanf(b_file, "\n"); // Skip newline
        fscanf(bm_file, "\n"); // Skip newline
        fscanf(bc_file, "\n"); // Skip newline
    }

    printf("Loaded Network Params..\n");

    // Close files
    fclose(w_file);
    fclose(b_file);
    cJSON_Delete(json);
    free(buffer);
}

///////////////////////////////////////////////// NETWORKING METHODS ////////////////////////////////////////////////////////////////////////////
int setup_socket() {
    int sockfd;
    struct sockaddr_in server_addr;

    // Create socket
    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(8080);  // Port number
    server_addr.sin_addr.s_addr = INADDR_ANY;  // Localhost

    // Connect to the server
    if (connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connection failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    return sockfd;
}

void send_data(NeuralNetwork* network, int sockfd) {
    cJSON *json = cJSON_CreateObject();
    cJSON_AddNumberToObject(json, "current_epoch", network->current_epoch);
    cJSON_AddNumberToObject(json, "num_epochs", network->num_epochs);
    cJSON_AddNumberToObject(json, "accuracy", network->accuracy);
    cJSON_AddNumberToObject(json, "loss", network->loss);
    cJSON_AddNumberToObject(json, "val_accuracy", network->val_accuracy);
    cJSON_AddNumberToObject(json, "val_loss", network->val_loss);

    char *json_string = cJSON_PrintUnformatted(json);
    send(sockfd, json_string, strlen(json_string), 0);

    free(json_string);
    cJSON_Delete(json);
}

