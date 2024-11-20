#include "layer_dense.h"
/*
C File for a Layer Dense "Object"
Author: Asutosh Ghimire
*/
//////////////////////////////////////////////////// LAYER METHODS //////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////// MISC METHODS ///////////////////////////////////////////////////////////////////////////

void clip_gradients(double* gradients, int size, double min_value, double max_value) {
    for (int i = 0; i < size; i++) {
        if (gradients[i] < min_value) {
            gradients[i] = min_value;  // Clipping to the lower bound
        }
        if (gradients[i] > max_value) {
            gradients[i] = max_value;  // Clipping to the upper bound
        }
    }
}

void apply_drop_out(layer_dense* layer, double drop_out_rate) {

    // Allocate memory for binary mask if it doesnt exist
    if (layer->binary_mask == NULL) {
        layer->binary_mask = malloc(sizeof(matrix));
        layer->binary_mask->dim1 = layer->post_activation_output->dim1;
        layer->binary_mask->dim2 = layer->post_activation_output->dim2;
        layer->binary_mask->data = (double*) calloc(layer->binary_mask->dim1 * layer->binary_mask->dim2, sizeof(double));

        // Check memory allocation
        if (layer->binary_mask->data == NULL) {
            fprintf(stderr,"Error: Memory allocation for binary mask failed.\n");
            exit(1);
        }

    }

    // Iterate through every batch example in layer output
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < layer->post_activation_output->dim1; i++) {
        // Iterate through every neuron 
        for (int j = 0; j < layer->post_activation_output->dim2; j++) {
            // Generate random num between 0 and 1
            double r = (double)rand() / RAND_MAX;
            // If less than drop out rate, drop neuron output
            if (r < drop_out_rate) {
                layer->post_activation_output->data[i * layer->post_activation_output->dim2 + j] = 0;
                layer->binary_mask->data[i * layer->post_activation_output->dim2 + j] = 0;
            }
            // Scale output by 1/(1- drop_out_rate)
            else {
                layer->post_activation_output->data[i * layer->post_activation_output->dim2 + j] /= (1-drop_out_rate);
                layer->binary_mask->data[i * layer->post_activation_output->dim2 + j] = 1 / drop_out_rate;
            }
        }
    }
}

layer_dense* init_layer(int num_inputs, int num_neurons, ActivationType activation, OptimizationType optimization) {

    // initialize a layer dense object
    layer_dense* layer_ = malloc(sizeof(layer_dense));
    layer_->num_inputs = num_inputs;
    layer_->num_neurons = num_neurons;

    // point memory to null for vars that are instantiated in forwards
    layer_->inputs = NULL;
    layer_->dinputs = NULL;
    layer_->pre_activation_output = NULL;
    layer_->post_activation_output = NULL;

    // point memory to null for pred inputs
    layer_->pred_inputs = NULL;

    // point memory to null for pred outputs
    layer_->pred_outputs = NULL;

    // init layer id
    layer_->id = -1;

    // Allocate memory for weights
    layer_->weights = (matrix*) malloc(sizeof(matrix));
    layer_->weights->data = (double*) calloc(num_inputs * num_neurons, sizeof(double));
    layer_->weights->dim1 = num_inputs; 
    layer_->weights->dim2 = num_neurons; 

    // Check memory allocation
    if (layer_->weights->data == NULL) {
        fprintf(stderr, "Error in memory allocation for weights in forward pass.\n");
        exit(1);
    }

    // Allocate memory for biases
    layer_->biases = (matrix*) malloc(sizeof(matrix));
    layer_->biases->data = (double*)calloc(num_neurons, sizeof(double));
    layer_->biases->dim1 = 1;
    layer_->biases->dim2 = num_neurons;

    // Check memory allocation
    if (layer_->biases->data == NULL) {
        fprintf(stderr, "Error in memory allocation for biases in forward pass.\n");
        exit(1);
    }

    // randomize weights
    srand(time(NULL));  // Seed random number with current time
    // srand(42);
    //  n_inputs x n_neurons matrix
    for (int i = 0; i < num_neurons * num_inputs; i++){
        // Random between -1 and 1 scaled by sqrt(1/n)
        // He initialization
        layer_->weights->data[i] = sqrt(1.0 / num_inputs) * ((double)rand() / RAND_MAX * 2.0 - 1.0);  

        // Xavier init
        // layer_->weights->data[i] = sqrt(1.0 / (num_inputs + num_neurons)) * ((double)rand() / RAND_MAX * 2.0 - 1.0);

    }

    // initialize other matrix objects, also allocate memory
    // derivative of weights
    layer_->dweights = (matrix*) malloc(sizeof(matrix));
    layer_->dweights->data = (double*) calloc(num_inputs * num_neurons, sizeof(double));
    layer_->dweights->dim1 = num_inputs;
    layer_->dweights->dim2 = num_neurons;

    // derivative of biases
    layer_->dbiases = (matrix*) malloc(sizeof(matrix));
    layer_->dbiases->data = (double*) calloc(num_neurons, sizeof(double));
    layer_->dbiases->dim1 = 1;
    layer_->dbiases->dim2 = num_neurons;

    // initialize activation function for the layer
    layer_->activation = activation;

    // initialize optimization function for the layer
    layer_->optimization = optimization;

    // Init use regularization to false
    layer_->useRegularization = false;

    // Init drop out rate to 0
    layer_->drop_out_rate = 0.0;

    // Initialize binary mask to NULL (init in apply dropout)
    layer_->binary_mask = NULL;

    // Init clip gradients to false
    layer_->clipGradients = false;

    // Init lambda for regularization
    layer_->lambda_l1 = 0.001;
    layer_->lambda_l2 = 0.01;

    // Initialize velocity for weights
    layer_->w_velocity = (matrix*)malloc(sizeof(matrix));
    layer_->w_velocity->dim1 = layer_->weights->dim1;
    layer_->w_velocity->dim2 = layer_->weights->dim2;
    layer_->w_velocity->data = (double*)calloc(layer_->w_velocity->dim1 * layer_->w_velocity->dim2, sizeof(double));

    // Check memory
    if (layer_->w_velocity->data == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for layer weight velocity.\n");
        free_layer(layer_);
        exit(1);
    } 

    // Initialize velocity for biases
    layer_->b_velocity = (matrix*)malloc(sizeof(matrix));
    layer_->b_velocity->dim1 = layer_->biases->dim1;
    layer_->b_velocity->dim2 = layer_->biases->dim2;
    layer_->b_velocity->data = (double*)calloc(layer_->b_velocity->dim1 * layer_->b_velocity->dim2, sizeof(double));

    // Check memory
    if (layer_->b_velocity->data == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for layer bias velocity.\n");
        free_layer(layer_);
        exit(1);
    } 

    // Initialize cache weights and biases for the layer
    layer_->cache_bias = malloc(sizeof(matrix));
    layer_->cache_bias->dim1 = layer_->biases->dim1;
    layer_->cache_bias->dim2 = layer_->biases->dim2;
    layer_->cache_bias->data = (double*) calloc(layer_->cache_bias->dim1 * layer_->cache_bias->dim2, sizeof(double));

    layer_->cache_weights = malloc(sizeof(matrix));
    layer_->cache_weights->dim1 = layer_->weights->dim1;
    layer_->cache_weights->dim2 = layer_->weights->dim2;
    layer_->cache_weights->data = (double*) calloc(layer_->cache_weights->dim1 * layer_->cache_weights->dim2, sizeof(double));
    
    // Check memory
    if (layer_->cache_bias == NULL || layer_->cache_bias->data == NULL) {
        fprintf(stderr, "Error: Memory allocation failure with cache_bias in init layer.\n");
        free_layer(layer_);
        exit(1);
    }

    if (layer_->cache_weights == NULL || layer_->cache_weights->data == NULL) {
        fprintf(stderr, "Error: Memory allocation failure with cache_weights in init layer.\n");
        free_layer(layer_);
        exit(1);
    }

    // Check dimensions
    if (layer_->cache_weights->dim1 != layer_->dweights->dim1 || layer_->cache_weights->dim2 != layer_->dweights->dim2) {
        fprintf(stderr, "Error: Dimensionality mismatch between cache weights and dweights in update params adagrad.\n");
        free_layer(layer_);
        exit(1);
    }

    // Check dimensions
    if (layer_->cache_bias->dim1 != layer_->dbiases->dim1 || layer_->cache_bias->dim2 != layer_->dbiases->dim2) {
        fprintf(stderr, "Error: Dimensionality mismatch between cache bias and dbiases in update params adagrad.\n");
        free(layer_->cache_bias->data);
        free(layer_->cache_bias);
        exit(1);
    }

    // return layer dense object
    return layer_;
}

void free_layer(layer_dense* layer) {
    free(layer->weights->data);
    free(layer->weights);
    free(layer->biases->data);
    free(layer->biases);
    free(layer->dweights->data);
    free(layer->dweights);
    free(layer->dbiases->data);
    free(layer->dbiases);
    // If inputs != NULL so do these other variables
    if (layer->inputs != NULL) {
        free(layer->inputs->data);
        free(layer->inputs);
        free(layer->dinputs->data);
        free(layer->dinputs);
        free(layer->pre_activation_output->data);
        free(layer->pre_activation_output);
        free(layer->post_activation_output->data);
        free(layer->post_activation_output);    
    }
    free(layer->w_velocity->data);
    free(layer->w_velocity);
    free(layer->b_velocity->data);
    free(layer->b_velocity);
    free(layer->cache_weights->data);
    free(layer->cache_weights);
    free(layer->cache_bias->data);
    free(layer->cache_bias);
    free(layer);
    layer = NULL;
}

//////////////////////////////////////////////////// ACCURACY METHODS ///////////////////////////////////////////////////////////////////////////

double calculate_accuracy(matrix* class_targets, layer_dense* final_layer, ClassLabelEncoding encoding) {

    // handles mismatching first dimensions 
    if (class_targets->dim1 != final_layer->post_activation_output->dim1) {
        fprintf(stderr, "Error: Mismatching dimensions in calculate accuracy, dim1 for class targets and predictions.\n");
        exit(1);
    } 

    // stores correct prediction count
    int correct_count = 0;

    // stores number of samples
    int num_samples = final_layer->post_activation_output->dim1;

    // handles one hot encoded vectors
    if (encoding == ONE_HOT) {

         // handles mismatching second dimensions 
        if (class_targets->dim2 != final_layer->post_activation_output->dim2) {
            fprintf(stderr, "Error: Mismatching dimensions in calculate accuracy, dim2 for class targets and predictions.\n");
            exit(1);
        } 

        // iter through every prediction
        for (int i = 0; i < final_layer->post_activation_output->dim1; i++) {

            // find max value, ie the prediction in each input in the batch
            int max_indx = -1;
            double max = -DBL_MAX;
            for (int j = 0; j < final_layer->post_activation_output->dim2; j++) {
                if (final_layer->post_activation_output->data[i * final_layer->post_activation_output->dim2 + j] > max) {
                    max = final_layer->post_activation_output->data[i * final_layer->post_activation_output->dim2 + j];
                    max_indx = j;
                }
            }

            // incriment correct count if the predictions match the one hot encoded vector
            if (class_targets->data[i * class_targets->dim2 + max_indx] == 1) {
                correct_count += 1;
            }
        }    
    }
    
    // handles sparse true label vectors
    else if (encoding == SPARSE) {

        // iter through every prediction
        for (int i = 0; i < final_layer->post_activation_output->dim1; i++) {
            int max_indx = -1;
            double max = -DBL_MAX;
            for (int j = 0; j < final_layer->post_activation_output->dim2; j++) {
                if (final_layer->post_activation_output->data[i * final_layer->post_activation_output->dim2 + j] > max) {
                    max = final_layer->post_activation_output->data[i * final_layer->post_activation_output->dim2 + j];
                    max_indx = j;
                }
            }

            // incriment correct count if the predictions match the sparse vector
            if (class_targets->data[i] == max_indx) {
                correct_count+=1;
            }
        }    
    }

    // handles encoding type input error
    else {
        fprintf(stderr, "Error: Incorrect encoding type provided.\n");
        exit(1);
    }

    // calculate and return accuracy
    double accuracy = (1.0)*correct_count / num_samples;

    return(accuracy);
}

double pred_calculate_accuracy(matrix* class_targets, layer_dense* final_layer, ClassLabelEncoding encoding) {

    // handles mismatching first dimensions 
    if (class_targets->dim1 != final_layer->pred_outputs->dim1) {
        fprintf(stderr, "Error: Mismatching dimensions in calculate accuracy, dim1 for class targets and predictions.\n");
        exit(1);
    } 

    // stores correct prediction count
    int correct_count = 0;

    // stores number of samples
    int num_samples = final_layer->pred_outputs->dim1;

    // handles one hot encoded vectors
    if (encoding == ONE_HOT) {

         // handles mismatching second dimensions 
        if (class_targets->dim2 != final_layer->pred_outputs->dim2) {
            fprintf(stderr, "Error: Mismatching dimensions in calculate accuracy, dim2 for class targets and predictions.\n");
            exit(1);
        } 

        // iter through every prediction
        for (int i = 0; i < final_layer->pred_outputs->dim1; i++) {

            // find max value, ie the prediction in each input in the batch
            int max_indx = -1;
            double max = -DBL_MAX;
            for (int j = 0; j < final_layer->pred_outputs->dim2; j++) {
                if (final_layer->pred_outputs->data[i * final_layer->pred_outputs->dim2 + j] > max) {
                    max = final_layer->pred_outputs->data[i * final_layer->pred_outputs->dim2 + j];
                    max_indx = j;
                }
            }

            // incriment correct count if the predictions match the one hot encoded vector
            if (class_targets->data[i * class_targets->dim2 + max_indx] == 1) {
                correct_count += 1;
            }
        }    
    }
    
    // handles sparse true label vectors
    else if (encoding == SPARSE) {

        // iter through every prediction
        for (int i = 0; i < final_layer->pred_outputs->dim1; i++) {
            int max_indx = -1;
            double max = -DBL_MAX;
            for (int j = 0; j < final_layer->pred_outputs->dim2; j++) {
                if (final_layer->pred_outputs->data[i * final_layer->pred_outputs->dim2 + j] > max) {
                    max = final_layer->pred_outputs->data[i * final_layer->pred_outputs->dim2 + j];
                    max_indx = j;
                }
            }

            // incriment correct count if the predictions match the sparse vector
            if (class_targets->data[i] == max_indx) {
                correct_count+=1;
            }
        }    
    }

    // handles encoding type input error
    else {
        fprintf(stderr, "Error: Incorrect encoding type provided.\n");
        exit(1);
    }

    // calculate and return accuracy
    double accuracy = (1.0)*correct_count / num_samples;

    return(accuracy);
}


//////////////////////////////////////////////////// LOSS METHODS ///////////////////////////////////////////////////////////////////////////

matrix* loss_categorical_cross_entropy(matrix* true_pred, layer_dense* last_layer, ClassLabelEncoding encoding) {

    // check if predictions and true values dim1 match in size
    if(last_layer->post_activation_output->dim1 != true_pred->dim1) {
        fprintf(stderr, "Mismatch in prediction batch size and true value size. \n");
        exit(1);
    }

    // initialize losses data.
    matrix* losses = malloc(sizeof(matrix));
    losses->data = (double*) calloc(last_layer->post_activation_output->dim1, sizeof(double));
    losses->dim1 = last_layer->post_activation_output->dim1;
    losses->dim2 = 1;

    // one hot encoded assumption
    if(encoding == ONE_HOT) {
        
        // check if one hot is the correct size
        if (last_layer->post_activation_output->dim2 != true_pred->dim2) {
            fprintf(stderr, "Error: Dimension 2 for one hot vectors and predictions do not match.\n");
            print_matrix(last_layer->post_activation_output);
            print_matrix(true_pred);
            free(losses->data);
            exit(1);
        }

        // iterate over every vector in the prediction batch
        for (int i = 0; i < last_layer->post_activation_output->dim1; i++) {

            // find true class in one hot vector
            int true_class = -1;
            for (int j = 0; j < true_pred->dim2; j++) {
                if (true_pred->data[i * true_pred->dim2 + j] == 1.0) {
                    true_class = j;
                    break;
                }
            }

            // error handling if no true class is found
            if(true_class == -1) {
                fprintf(stderr, "Error: No true class found in one hot vectors. \n");
                free(losses->data);
                exit(1);
            }

            // get predicted sample in question with relation to true class
            double predicted_sample = last_layer->post_activation_output->data[i * last_layer->post_activation_output->dim2 + true_class];

            // clip value so we never calculate log(0)
            if(predicted_sample < 1e-15) {
                predicted_sample = 1e-15;
            }
            
            // calcuale -log loss for the sample in question and append to loss matrix
            double loss = -log(predicted_sample);
            losses->data[i] = loss;
        }
    }

    // sparse encoded assumption
    else if (encoding == SPARSE) {

        // iterate through every true classification in the sparse vector
        for (int i = 0; i < true_pred->dim1; i++) {

            // get true class value from 1d sparse vector.
            int true_class = (int)true_pred->data[i];

            // Error handling, check if true class is in bounds of prediction vectors
            if (true_class < 0 || true_class >= last_layer->post_activation_output->dim2) {
                fprintf(stderr,"Error: True class dimensions out of bounds. \n");
                free(losses->data);
                exit(1);
            }  

            // get predicted sample from batch data 
            double predicted_sample = last_layer->post_activation_output->data[i * last_layer->post_activation_output->dim2 + true_class];
            
            // clip value so we never calculate log(0)
            if (predicted_sample < 1e-15) {
                predicted_sample = 1e-15;
            }

            // calcuale -log loss for the sample in question and append to loss matrix
            double loss = -log(predicted_sample);
            
            losses->data[i] = loss;
        }
    }

    // error handling
    else {
        fprintf(stderr, "Error: Incorrect type encoding provided. \n");
        free(losses->data);
        exit(1);
    }

    // return losses
    return(losses);
}

matrix* pred_loss_categorical_cross_entropy(matrix* true_pred, layer_dense* last_layer, ClassLabelEncoding encoding) {

    // check if predictions and true values dim1 match in size
    if(last_layer->pred_outputs->dim1 != true_pred->dim1) {
        fprintf(stderr, "Mismatch in prediction batch size and true value size. \n");
        exit(1);
    }

    // initialize losses data.
    matrix* losses = malloc(sizeof(matrix));
    losses->data = (double*) calloc(last_layer->pred_outputs->dim1, sizeof(double));
    losses->dim1 = last_layer->pred_outputs->dim1;
    losses->dim2 = 1;

    // one hot encoded assumption
    if(encoding == ONE_HOT) {
        
        // check if one hot is the correct size
        if (last_layer->pred_outputs->dim2 != true_pred->dim2) {
            fprintf(stderr, "Error: Dimension 2 for one hot vectors and predictions do not match.\n");
            print_matrix(last_layer->pred_outputs);
            print_matrix(true_pred);
            free(losses->data);
            exit(1);
        }

        // iterate over every vector in the prediction batch
        for (int i = 0; i < last_layer->pred_outputs->dim1; i++) {

            // find true class in one hot vector
            int true_class = -1;
            for (int j = 0; j < true_pred->dim2; j++) {
                if (true_pred->data[i * true_pred->dim2 + j] == 1.0) {
                    true_class = j;
                    break;
                }
            }

            // error handling if no true class is found
            if(true_class == -1) {
                fprintf(stderr, "Error: No true class found in one hot vectors. \n");
                free(losses->data);
                exit(1);
            }

            // get predicted sample in question with relation to true class
            double predicted_sample = last_layer->pred_outputs->data[i * last_layer->pred_outputs->dim2 + true_class];

            // clip value so we never calculate log(0)
            if(predicted_sample < 1e-15) {
                predicted_sample = 1e-15;
            }
            
            // calcuale -log loss for the sample in question and append to loss matrix
            double loss = -log(predicted_sample);
            losses->data[i] = loss;
        }
    }

    // sparse encoded assumption
    else if (encoding == SPARSE) {

        // iterate through every true classification in the sparse vector
        for (int i = 0; i < true_pred->dim1; i++) {

            // get true class value from 1d sparse vector.
            int true_class = (int)true_pred->data[i];

            // Error handling, check if true class is in bounds of prediction vectors
            if (true_class < 0 || true_class >= last_layer->pred_outputs->dim2) {
                fprintf(stderr,"Error: True class dimensions out of bounds. \n");
                free(losses->data);
                exit(1);
            }  

            // get predicted sample from batch data 
            double predicted_sample = last_layer->pred_outputs->data[i * last_layer->pred_outputs->dim2 + true_class];
            
            // clip value so we never calculate log(0)
            if (predicted_sample < 1e-15) {
                predicted_sample = 1e-15;
            }

            // calcuale -log loss for the sample in question and append to loss matrix
            double loss = -log(predicted_sample);
            
            losses->data[i] = loss;
        }
    }

    // error handling
    else {
        fprintf(stderr, "Error: Incorrect type encoding provided. \n");
        free(losses->data);
        exit(1);
    }

    // return losses
    return(losses);
}

double calculate_regularization_loss(layer_dense* layer) {

    // Check if using regularization
    if (!layer->useRegularization) {
        return 0.0;
    }

    double l1_w = 0.0; // L1 weight regularization
    double l2_w = 0.0; // L2 weight regularization
    double l1_b = 0.0; // L1 bias regularization
    double l2_b = 0.0; // L2 bias regularization

    // Weight regularization L1 and L2
    #pragma omp for schedule(dynamic)  // Paralellize it (schedule dynamic helps allocate resources)
    for (int i = 0; i < layer->weights->dim1 * layer->weights->dim2; i++) {
        l1_w += fabs(layer->weights->data[i]);
        l2_w += layer->weights->data[i] * layer->weights->data[i];
    }

    // Bias regularization L1 and L2
    #pragma omp for schedule(dynamic)  // Paralellize it (schedule dynamic helps allocate resources)
    for (int i = 0; i <layer->biases->dim1 * layer->biases->dim2; i++) {
        l1_b += fabs(layer->biases->data[i]);
        l2_b += layer->biases->data[i] * layer->biases->data[i];
    }

    // Multiply regularizations by lambda and return sum of all regularizations
    l1_w *= layer->lambda_l1;
    l2_w *= layer->lambda_l2;
    l1_b *= layer->lambda_l1;
    l2_b *= layer->lambda_l2;

    return (l1_w + l2_w + l1_b + l2_b);
}


////////////////////////////////////////////////// FORWARD METHODS ///////////////////////////////////////////////////////////////////////////

void forward_pass(matrix* inputs, layer_dense* layer) {

     // Allocate memory for layer input and dinput data
    if(layer->inputs == NULL) {
        layer->inputs = malloc(sizeof(matrix));
        layer->inputs->dim1 = inputs->dim1;
        layer->inputs->dim2 = inputs->dim2;
        layer->inputs->data = (double*) calloc(layer->inputs->dim1 * layer->inputs->dim2, sizeof(double));
        // Check memory allocation
        if (layer->inputs->data == NULL) {
            fprintf(stderr, "Error in memory allocation for inputs in forward pass.\n");
            exit(1);
        }   
    } 

    // derivative of inputs
    if (layer->dinputs == NULL) {
        layer->dinputs = (matrix*) malloc(sizeof(matrix));
        layer->dinputs->dim1 = inputs->dim1;
        layer->dinputs->dim2 = inputs->dim2;
        layer->dinputs->data = (double*) calloc(layer->dinputs->dim1 * layer->dinputs->dim2, sizeof(double));
        // Check memory allocation
        if (layer->dinputs->data == NULL) {
            fprintf(stderr, "Error in memory allocation for dinputs in forward pass.\n");
            exit(1);
        }
    }

    // Copy inputs into layer structure
    memcpy(layer->inputs->data, inputs->data, layer->inputs->dim1 * layer->inputs->dim2 * sizeof(double));


    // Allocate memory for pre activation outputs
    if (layer->pre_activation_output == NULL) {
        layer->pre_activation_output = malloc(sizeof(matrix));
        layer->pre_activation_output->dim1 = inputs->dim1;
        layer->pre_activation_output->dim2 = layer->num_neurons;
        layer->pre_activation_output->data = (double*) calloc(layer->pre_activation_output->dim1*
                                                        layer->pre_activation_output->dim2, sizeof(double));
        // Check memory allocation
        if (layer->pre_activation_output->data == NULL) {
            fprintf(stderr, "Error in memory allocation for pre_activation_outputs in forward pass.\n");
            exit(1);
        }
    }

    // Allocate memory for post activation outputs
    if (layer->post_activation_output == NULL) {
        layer->post_activation_output = malloc(sizeof(matrix));
        layer->post_activation_output->dim1 = inputs->dim1;
        layer->post_activation_output->dim2 = layer->num_neurons;
        layer->post_activation_output->data = (double*) calloc(layer->post_activation_output->dim1*
                                                        layer->post_activation_output->dim2, sizeof(double));
    
       // Check memory allocation
        if (layer->post_activation_output->data == NULL) {
            fprintf(stderr, "Error in memory allocation for post_activation_output in forward pass.\n");
            exit(1);
        }
    }
  
    // Allocate output memory
    matrix* output = malloc(sizeof(matrix));
    output->dim1 = inputs->dim1; // number of vectors in batch
    output->dim2 = layer->weights->dim2; // number of neurons in weights
    output->data = (double*) calloc(output->dim1  * output->dim2, sizeof(double));

    // Check memory allocation
    if (output->data == NULL) {
        fprintf(stderr, "Error in matrix mult for forward pass.\n");
        exit(1); 
    }

    /* 
    num_inputs x num_neurons
    Perform matrix multiplication between inputs (dim1 x dim2) and weights (dim2 x dim3)
    eturns batch_size x num_neuron matrix for the layer 
    */
    matrix* mult_matrix = matrix_mult(inputs, layer->weights);

    // Add biases for the layer to the batch output data
    // batch_size x num_neurons, where output dim1 -> batch size

    #pragma omp for collapse(2) schedule(dynamic) // Paralellize it (schedule dynamic helps allocate resources)
    for (int i = 0; i < output->dim1; i++) {
        // output dim2-> num neurons
        for (int j = 0; j < output->dim2; j++) {
            output->data[i * output->dim2 + j] = mult_matrix->data[i * output->dim2 + j] + layer->biases->data[j];
        }
    }

    // Update pre activation outputs
    memcpy(layer->pre_activation_output->data,  output->data, output->dim1 * output->dim2 * sizeof(double));

    // relu activation
    if (layer->activation == RELU) {
        forward_reLu(output);
    } 
    // softmax activation
    else if(layer->activation == SOFTMAX) {
        forward_softMax(output);
    }

    // Update post activation outputs
    memcpy(layer->post_activation_output->data,  output->data, output->dim1 * output->dim2 * sizeof(double));

    // Apply dropout
    if (layer->drop_out_rate > 0.0) {
        apply_drop_out(layer, layer->drop_out_rate);
    }

    // Free unused memory
    free(output->data);
    free(output);
    free(mult_matrix->data);
    free(mult_matrix);
}

void pred_forward_pass(matrix* inputs, layer_dense* layer) {

     // Allocate memory for prediction inputs
    if(layer->pred_inputs == NULL) {
        layer->pred_inputs = malloc(sizeof(matrix));
        layer->pred_inputs->dim1 = inputs->dim1;
        layer->pred_inputs->dim2 = inputs->dim2;
        layer->pred_inputs->data = (double*) calloc(layer->pred_inputs->dim1 * layer->pred_inputs->dim2, sizeof(double));

        // Check memory allocation
        if (layer->pred_inputs->data == NULL) {
            fprintf(stderr, "Error in memory allocation for pred_inputs in pred_forward pass.\n");
            exit(1);
        }   
    }

    // Allocate memory for prediction outputs
    if (layer->pred_outputs == NULL) {
        layer->pred_outputs = malloc(sizeof(matrix));
        layer->pred_outputs->dim1 = inputs->dim1;
        layer->pred_outputs->dim2 = layer->num_neurons;
        layer->pred_outputs->data = (double*) calloc(layer->pred_outputs->dim1*
                                                        layer->pred_outputs->dim2, sizeof(double));
    
       // Check memory allocation
        if (layer->pred_outputs->data == NULL) {
            fprintf(stderr, "Error in memory allocation for pred_outputs in pred_forward pass.\n");
            exit(1);
        }
    }
    
    // Check if weights and biases exist
    if(layer->weights->data == NULL || layer->biases->data == NULL) {
        fprintf(stderr, "Error: Weights and Biases not initialized in pred_forward.\n");
        exit(1);
    }

    // Multiply inputs and weights
    matrix* z = matrix_mult(inputs, layer->weights);

    // Add biases
    #pragma omp for collapse(2) schedule(dynamic) // Paralellize it (schedule dynamic helps allocate resources)
    for (int i = 0; i < layer->pred_outputs->dim1; i++) {
        for (int j = 0; j < layer->pred_outputs->dim2; j++) {
            layer->pred_outputs->data[i * layer->pred_outputs->dim2 + j] = z->data[i * layer->pred_outputs->dim2 + j] + layer->biases->data[j];
        }
    }

    // Apply Activation
    
    // relu activation
    if (layer->activation == RELU) {
        forward_reLu(layer->pred_outputs);
    } 
    // softmax activation
    else if(layer->activation == SOFTMAX) {
        forward_softMax(layer->pred_outputs);
    }
}

void forward_reLu(matrix* batch_input) {
    // iterate through every point in the batch input

    #pragma omp for schedule(dynamic) // Paralellize it (schedule dynamic helps allocate resources)
    for (int i = 0; i < batch_input->dim1 * batch_input->dim2; i++){

        // if the input value is <= 0, rectify it to 0 (otherwise, leave it unchanged)
        if(batch_input->data[i] <= 0) {
            batch_input->data[i] = 0;
        }
    }
}

void forward_softMax(matrix* batch_input) {
    // iterate over the batch
    #pragma omp for schedule(dynamic) // Paralellize it (schedule dynamic helps allocate resources)
    for(int i = 0; i < batch_input -> dim1; i++) {

        //step 1: Subtract maximum value from each value in the input batch to ensure numerical stability (no large exponentiations)
        double max = -DBL_MAX;
        for(int j = 0; j < batch_input->dim2; j++){
            if (batch_input->data[i*batch_input->dim2 + j] > max) {
                max = batch_input->data[i*batch_input->dim2 + j];
            }
        }

        // step 2: calculate exponentials and sum them
        double* exp_values = (double*) calloc(batch_input->dim2, sizeof(double));
        double sum = 0.0;
        for(int j = 0; j < batch_input -> dim2; j++) {
            exp_values[j] = exp(batch_input->data[i * batch_input->dim2 + j] - max);
            sum += exp_values[j];
        }

        // step 3: normalize exponentials by dividing by the sum to get probabilities
        for(int j = 0; j < batch_input->dim2; j++) {
            batch_input->data[i * batch_input->dim2 + j] = exp_values[j] / sum;
        }

        // step 4: free temp exp values 
        free(exp_values);
    }
}


////////////////////////////////////////////////// BACKWARD METHODS ///////////////////////////////////////////////////////////////////////////

void backward_reLu(matrix* input_gradients, layer_dense* layer) {
    /*
    Find Gradient of the ReLu of the layer.
    STEP 1: Find Gradient of ReLu based off of pre activation output of the layer (ie. data that hasnt had ReLU applied yet.)
    STEP 2: Perform an element by element multiplication of the input gradients from the layer before to the gradients found in step 1.
    */

    // Allocate memory for ReLU gradient
    matrix* relu_gradients = malloc(sizeof(matrix));
    relu_gradients->dim1 = layer->pre_activation_output->dim1;
    relu_gradients->dim2 = layer->pre_activation_output->dim2;
    relu_gradients->data = (double*) calloc(relu_gradients->dim1 * relu_gradients->dim2, sizeof(double));

    // Check memory allocation
    if (relu_gradients->data == NULL) {
        fprintf(stderr, "Error in memory allocation for relu gradient in backward relu.\n");
        exit(1);
    }

    // Iterate through every value in layer post activation output to get relu gradients
    for (int i = 0; i < layer->pre_activation_output->dim1 * layer->pre_activation_output->dim2; i++) {
        if (layer->pre_activation_output->data[i] >= 0) {
            relu_gradients->data[i] = 1;
        }
        else {
            relu_gradients->data[i] = 0;
        }
    }

    // Check dimensions for element by element multiplication of input gradients and relu gradients.
    if (input_gradients->dim1 != relu_gradients->dim1 || input_gradients->dim2 != relu_gradients->dim2) {
        fprintf(stderr,"Error: Dimensionality mismatch between relu gradients and input gradients in backwards relu.\n");
        printf("input_gradients(%d x %d) != relu_gradients(%d x %d)\n", input_gradients->dim1, input_gradients->dim2,
        relu_gradients->dim1, relu_gradients->dim2);
        free(relu_gradients->data);
        exit(1);
    }

    // Element by element mult of input gradients and relu gradients
    #pragma omp for collapse(2) schedule(dynamic) // Paralellize it (schedule dynamic helps allocate resources)
    for(int i = 0; i < relu_gradients->dim1; i++) {
        for (int j = 0; j < relu_gradients->dim2; j++) {
            relu_gradients->data[i * relu_gradients->dim2 + j] = relu_gradients->data[i * relu_gradients->dim2 + j] * 
            input_gradients->data[i * relu_gradients->dim2 + j];
        }
    }

    /*
    Find Gradient of the Weights and Biases of the layer.
    STEP 1: Find the gradient of weights
        -> inputs coming from the layer before transposed dot the ReLu gradients calculated earlier. 
    STEP 2: Find the gradients of biases
        -> summation of the ReLu gradients across eatch example in the batch: returns dimensions 1 x 5
    */

    // Calculate weight gradients

    // Transpose inputs
    matrix* inputs_transposed = transpose_matrix(layer->inputs);

    // Check dimensions
    if(inputs_transposed->dim2 != relu_gradients-> dim1) {
        fprintf(stderr, "Error: Dimensionality mismatch between inputs_transposed and relu_gradients in weight calculation.\n");
        free(inputs_transposed);
        free(relu_gradients);
        exit(1);
    }
    // Define max gradient for exploding gradients
    double max_gradient = 100.0;

    // Perform the dot product
    layer->dweights = matrix_mult(inputs_transposed, relu_gradients);

    // If clipping gradients, apply
    if (layer->clipGradients) {
        clip_gradients(layer->dweights->data, layer->dweights->dim1 * layer->dweights->dim2, -1, 1);
    }

    // Calculate bias gradients
    // Sum the relu gradients for each example in the batch of inputs
    #pragma omp for collapse(2) schedule(dynamic) // Paralellize it (schedule dynamic helps allocate resources)
    for (int j = 0; j < layer->dbiases->dim2; j++) {
        for(int i = 0; i < relu_gradients->dim1; i++) {
            // sum across rows
            layer->dbiases->data[j] += relu_gradients->data[i * relu_gradients->dim2 + j];
            if (layer->dbiases->data[j] > max_gradient) {
                layer->dbiases->data[j] = max_gradient;
            }
        }
    }

    // If clipping gradients, apply
    if (layer->clipGradients) {
        clip_gradients(layer->dbiases->data, layer->dbiases->dim1 * layer->dbiases->dim2, -1, 1);
    }

    if (layer->useRegularization) {
        // weights
        for (int i = 0; i < layer->dweights->dim1 * layer->dweights->dim2; i++) {
            // L2 gradients
            layer->dweights->data[i] += 2 * layer->lambda_l2 * layer->weights->data[i];

            // L1 gradients (1 if > 0, -1 if < 0)
            layer->dweights->data[i] += layer->lambda_l1 * (layer->weights->data[i] >= 0.0 ? 1.0 : -1.0);
        }
        // biases
        for (int i = 0; i < layer->dbiases->dim1 * layer->dbiases->dim2; i++) {
            // L2 gradients
            layer->dbiases->data[i] += 2 * layer->lambda_l2 * (layer->biases->data[i]);

            // L1 gradients (1 if > 0, -1 if < 0)
            layer->dbiases->data[i] += layer->lambda_l1 * (layer->biases->data[i] >= 0 ? 1.0: -1.0);
        }
    }

    // Calculate gradients for the input

    // Transpose weights
    matrix* weights_transposed = transpose_matrix(layer->weights);

    // Check dimensions
    if (relu_gradients->dim2 != weights_transposed->dim1) {
        fprintf(stderr, "Error: Dimensionality mismatch between relu gradients and weights transposed in backwards RELU\n");
        free(weights_transposed->data);
        free(weights_transposed);
        free(relu_gradients->data);
        free(relu_gradients);
        free(inputs_transposed->data);
        free(inputs_transposed);
        exit(1);
    }

    // Dot product of relu_gradients and weights transposed
    matrix* output_gradients= matrix_mult(relu_gradients, weights_transposed);

    // If using dropout, apply
    if (layer->drop_out_rate > 0.0) {
        for (int i = 0; i < output_gradients->dim1 * output_gradients->dim2; i++) {
            output_gradients->data[i] *= layer->binary_mask->data[i];
        }
    }

    // Copy to dinputs
    memcpy(layer->dinputs->data, output_gradients->data, layer->dinputs->dim1 * layer->dinputs->dim2 * sizeof(double));

    // If clipping gradients, apply
    if (layer->clipGradients) {
        clip_gradients(layer->dinputs->data, layer->dinputs->dim1 * layer->dinputs->dim2, -1, 1);
    }

    // Final dimensionality check
    if (layer->weights->dim1 != layer->dweights->dim1 || layer->weights->dim2 != layer->dweights->dim2) {
        fprintf(stderr, "Error. Dimensionality mismatch between dweights and weights in backwards ReLu.\n");
        free(output_gradients->data);
        free(output_gradients);
        free(weights_transposed->data);
        free(weights_transposed);
        free(relu_gradients->data);
        free(relu_gradients);
        free(inputs_transposed->data);
        free(inputs_transposed);
    }

    if (layer->biases->dim1 != layer->dbiases->dim1 || layer->biases->dim2 != layer->dbiases->dim2) {
        fprintf(stderr, "Error. Dimensionality mismatch between dbiases and biases in backwards ReLu.\n");
        free(output_gradients->data);
        free(output_gradients);
        free(weights_transposed->data);
        free(weights_transposed);
        free(relu_gradients->data);
        free(relu_gradients);
        free(inputs_transposed->data);
        free(inputs_transposed);
    }

    if (layer->inputs->dim1 != layer->dinputs->dim1 || layer->inputs->dim2 != layer->dinputs->dim2) {
        fprintf(stderr, "Error. Dimensionality mismatch between dinputs and inputs in backwards ReLu.\n");
        free(output_gradients->data);
        free(output_gradients);
        free(weights_transposed->data);
        free(weights_transposed);
        free(relu_gradients->data);
        free(relu_gradients);
        free(inputs_transposed->data);
        free(inputs_transposed);
    }

    // free unused memory
    free(output_gradients->data);
    free(output_gradients);
    free(weights_transposed->data);
    free(weights_transposed);
    free(relu_gradients->data);
    free(relu_gradients);
    free(inputs_transposed->data);
    free(inputs_transposed);
}

void backwards_softmax_and_loss(matrix* true_labels, layer_dense* layer) {

    // Check dimensionality
    if (layer->post_activation_output->dim1 != true_labels->dim1 || layer->post_activation_output->dim2 != true_labels->dim2) {
        fprintf(stderr, "Error: Dimensionality mismatch between true labels and predictions in backwards softmax.\n");
        printf("Inputs:(%d x %d) != True:(%d x %d)\n", layer->post_activation_output->dim1, layer->post_activation_output->dim2, 
        true_labels->dim1, true_labels->dim2);
        exit(1);
    }

    // Calculate softmax loss partial derivatives

    // Allocate memory for loss gradients
    matrix* loss_gradients = malloc(sizeof(matrix));

    // same dimensions as the input (batch_size x neurons)
    // same dimensions as the true_labels
    loss_gradients->dim1 = layer->post_activation_output->dim1;
    loss_gradients->dim2 = layer->post_activation_output->dim2;

    loss_gradients->data = (double*) calloc(loss_gradients->dim1 * loss_gradients->dim2, sizeof(double));

    // Check memory allocation
    if(loss_gradients->data == NULL) {
        fprintf(stderr, "Error in memory allocation for loss gradients in softmax backprop.\n");
        free(loss_gradients);
        exit(1);
    }

    // For each example in the input batch
    #pragma omp for collapse(2) schedule(dynamic) // Paralellize it (schedule dynamic helps allocate resources)
    for (int i = 0; i < loss_gradients->dim1; i++){
        // For each neuron in the input vector
        for(int j = 0; j < loss_gradients->dim2; j++) {
            loss_gradients->data[i * loss_gradients->dim2 + j] = layer->post_activation_output->data[i * loss_gradients->dim2 + j] - 
            true_labels->data[i * loss_gradients->dim2 + j];
        }
    }

    // Calculate layer weight derivatives
    // dot product of inputs for the layer and loss_gradients calculated above.

    // Transpose layer inputs
    matrix* inputs_T = transpose_matrix(layer->inputs);

    // Check dimensions
    if(inputs_T->dim2 != loss_gradients->dim1) {
        fprintf(stderr, "Error: Dimensionality mismatch for inputs transposed in backwards softmax.\n");
        free(loss_gradients->data);
        free(loss_gradients);
        free(inputs_T->data);
        exit(1);
    }

    /*
    dweights and dbiases not used in backprop here, but later in optimization.
    */

    // Calculate dweights -> dont need to allocate memory as matrix_mult does that.
    layer->dweights = matrix_mult(inputs_T, loss_gradients);

    // If clipping gradients, apply
    if (layer->clipGradients) {
        clip_gradients(layer->dweights->data, layer->dweights->dim1 * layer->dweights->dim2, -1, 1);
    }

    // Calculate layer bias derivatives

    // Check memory allocation
    if (layer->dbiases->data == NULL) {
        fprintf(stderr, "Error in memory allocation for dbiases in softmax backprop. \n");
        free(loss_gradients->data);
        free(loss_gradients);
        free(inputs_T->data);
        exit(1);    
    }

    // Sum the loss gradients for each example in the batch of inputs
    #pragma omp for collapse(2) schedule(dynamic) // Paralellize it (schedule dynamic helps allocate resources)
    for (int j = 0; j < layer->dbiases->dim2; j++) {
        for(int i = 0; i < layer->post_activation_output->dim1; i++) {
            // sum across rows
            layer->dbiases->data[j] += loss_gradients->data[i * loss_gradients->dim2 + j];
        }
    }

    // If clipping gradients, apply
    if (layer->clipGradients) {
        clip_gradients(layer->dbiases->data, layer->dbiases->dim1 * layer->dbiases->dim2, -1, 1);
    }

    // Add regularization derivatives to dweights and dbiases

    // Check if using regularization
    if (layer->useRegularization) {
        // weights
        for (int i = 0; i < layer->dweights->dim1 * layer->dweights->dim2; i++) {
            // L2 gradients
            layer->dweights->data[i] += 2 * layer->lambda_l2 * layer->weights->data[i];

            // L1 gradients (1 if > 0, -1 if < 0)
            layer->dweights->data[i] += layer->lambda_l1 * (layer->weights->data[i] >= 0.0 ? 1.0 : -1.0);
        }
        // biases
        for (int i = 0; i < layer->dbiases->dim1 * layer->dbiases->dim2; i++) {
            // L2 gradients
            layer->dbiases->data[i] += 2 * layer->lambda_l2 * (layer->biases->data[i]);

            // L1 gradients (1 if > 0, -1 if < 0)
            layer->dbiases->data[i] += layer->lambda_l1 * (layer->biases->data[i] >= 0 ? 1.0: -1.0);
        }
    }

    // Backpropogate derivatives for previous layer

    // Transpose weights for layer
    matrix* weights_transposed = transpose_matrix(layer->weights);

    // Check dimensions
    if(loss_gradients->dim2 != weights_transposed->dim1) {
        fprintf(stderr, "Error: Dimensionality mismatch for weights transposed in backprop softmax function.\n");
        free(loss_gradients->data);
        free(loss_gradients);
        free(inputs_T->data);
        free(layer->dbiases->data);
        free(layer->dbiases);
        free(layer->dweights->data);
        free(layer->dweights);
        exit(1);    
    }

    // Calculate backprop derivative to pass to layer previous
    matrix* output_gradients = matrix_mult(loss_gradients, weights_transposed);

    // Ensure dimensions and sizes match
    if (output_gradients->dim1 != layer->dinputs->dim1 || output_gradients->dim2 != layer->dinputs->dim2) {
        fprintf(stderr, "Error, dinputs does not match calculated gradient dimensions in backwards softmax.\n");
        free(loss_gradients->data);
        free(loss_gradients);
        free(inputs_T->data);
        free(layer->dbiases->data);
        free(layer->dbiases);
        free(layer->dweights->data);
        free(layer->dweights);
        exit(1);    
    }

    // If using dropout, apply
    if (layer->drop_out_rate > 0.0) {
        for (int i = 0; i < output_gradients->dim1 * output_gradients->dim2; i++) {
            output_gradients->data[i] *= layer->binary_mask->data[i];
        }
    }

    // Save to layer data structure
    memcpy(layer->dinputs->data, output_gradients->data, layer->dinputs->dim1 * layer->dinputs->dim2 * sizeof(double));
    
    // If clipping gradients, apply
    if (layer->clipGradients) {
        clip_gradients(layer->dinputs->data, layer->dinputs->dim1 * layer->dinputs->dim2, -1, 1);
    }

    // Final dimensionality check
    if (layer->weights->dim1 != layer->dweights->dim1 || layer->weights->dim2 != layer->dweights->dim2) {
        fprintf(stderr, "Error. Dimensionality mismatch between dweights and weights in backwards ReLu.\n");
        free(output_gradients->data);
        free(output_gradients);
        free(weights_transposed->data);
        free(weights_transposed);
        free(loss_gradients->data);
        free(loss_gradients);
    }

    if (layer->biases->dim1 != layer->dbiases->dim1 || layer->biases->dim2 != layer->dbiases->dim2) {
        fprintf(stderr, "Error. Dimensionality mismatch between dbiases and biases in backwards ReLu.\n");
        free(output_gradients->data);
        free(output_gradients);
        free(weights_transposed->data);
        free(weights_transposed);
        free(loss_gradients->data);
        free(loss_gradients);
    }

    if (layer->inputs->dim1 != layer->dinputs->dim1 || layer->inputs->dim2 != layer->dinputs->dim2) {
        fprintf(stderr, "Error. Dimensionality mismatch between dinputs and inputs in backwards ReLu.\n");
        free(output_gradients->data);
        free(output_gradients);
        free(weights_transposed->data);
        free(weights_transposed);
        free(loss_gradients->data);
        free(loss_gradients);
    }

    // free unused memory
    free(output_gradients->data);
    free(output_gradients);
    free(weights_transposed->data);
    free(weights_transposed);
    free(loss_gradients->data);
    free(loss_gradients);
}


////////////////////////////////////////////////// OPTIMIZER METHODS ///////////////////////////////////////////////////////////////////////////

void update_params_sgd(layer_dense* layer, double* learning_rate, int current_epoch, double decay_rate) {

    // Decay the learning rate after each epoch
    // *learning_rate = *learning_rate / (1 + decay_rate * current_epoch);
    // fmax ensures min learning rate of 0.000001
    *learning_rate = fmax(*learning_rate * exp(-decay_rate * current_epoch), 0.000001);

    // Update weights
    #pragma omp for collapse(2) schedule(dynamic) // Paralellize it (schedule dynamic helps allocate resources)
    for (int i = 0; i < layer->num_neurons; i++) {
        for (int j = 0; j < layer->num_inputs; j++) {
            // W = W - learning_rate * dL/dW
            layer->weights->data[i * layer->num_inputs + j] -= *learning_rate * layer->dweights->data[i * layer->num_inputs + j];
            
        }
    }

    // Update biases
    for(int i = 0; i < layer->num_neurons; i++) {
        // b = b - learning_rate * dL/dB
        layer->biases->data[i] -= *learning_rate * layer->dbiases->data[i];
    }
}

void update_params_sgd_momentum(layer_dense* layer, double* learning_rate, int current_epoch, double decay_rate, double beta) {

    // Decay the learning rate after each epoch
    // *learning_rate = *learning_rate / (1 + decay_rate * current_epoch);
    // fmax ensures min learning rate of 0.000001
    *learning_rate = fmax(*learning_rate * exp(-decay_rate * current_epoch), 0.000001);
    
    // Update weights
    #pragma omp for collapse(2) schedule(dynamic) // Paralellize it (schedule dynamic helps allocate resources)
    for (int i = 0; i < layer->num_neurons; i++) {
        for (int j = 0; j < layer->num_inputs; j++) {
            // v_t = beta * v_(t-1) + (1 - beta) * dL/dW
            layer->w_velocity->data[i*layer->num_inputs+j] = beta * layer->w_velocity->data[i*layer->num_inputs+j] 
                                                            + (1-beta) * layer->dweights->data[i*layer->num_inputs+j];
            // W = W - learning_rate * v_t
            layer->weights->data[i*layer->num_inputs+j] -= *learning_rate * layer->w_velocity->data[i*layer->num_inputs+j];
            
        }
    }

    // Update biases
    #pragma omp for schedule(dynamic) // Paralellize it (schedule dynamic helps allocate resources)
    for(int i = 0; i < layer->num_neurons; i++) {
        // v_t = beta * v_(t-1) + (1 - beta) * dL/dB
        layer->b_velocity->data[i] = beta * layer->b_velocity->data[i] + (1 - beta) * layer->dbiases->data[i];
        // b = b - learning_rate * v_t
        layer->biases->data[i] -= *learning_rate * layer->b_velocity->data[i];
    }
}

void update_params_adagrad(layer_dense* layer, double* learning_rate, double decay_rate, double epsilon) {
    // WEIGHTS

    // Square every element in dweights, add to cache_weights
    #pragma omp for schedule(dynamic) // Paralellize it (schedule dynamic helps allocate resources)
    for (int i = 0; i < layer->cache_weights->dim1 * layer->cache_weights->dim2; i++) {
        // Calculate cache
        layer->cache_weights->data[i] += layer->dweights->data[i] * layer->dweights->data[i];

        layer->weights->data[i] -= *learning_rate * layer->dweights->data[i] / (sqrt(layer->cache_weights->data[i]) + epsilon);
    }


    // BIASES

    // Square every element in dbiases, add to cache_biases
    #pragma omp for schedule(dynamic) // Paralellize it (schedule dynamic helps allocate resources)
    for (int i = 0; i < layer->cache_bias->dim1 * layer->cache_bias->dim2; i++) {
        // Calculate cache
        layer->cache_bias->data[i] += layer->dbiases->data[i] * layer->dbiases->data[i];

        layer->biases->data[i] -= *learning_rate * layer->dbiases->data[i] / (sqrt(layer->cache_bias->data[i]) + epsilon);
    }
}

void update_params_rmsprop(layer_dense* layer, double* learning_rate, double decay_rate, double epsilon) {
    // WEIGHTS
    #pragma omp for schedule(dynamic) // Paralellize it (schedule dynamic helps allocate resources)
    for (int i = 0; i < layer->weights->dim1 * layer->weights->dim2; i++) {
        // Update cache for weights
        layer->cache_weights->data[i] = decay_rate * layer->cache_weights->data[i] +
                                                (1.0 - decay_rate) * layer->dweights->data[i] * layer->dweights->data[i];
        // Update weights
        layer->weights->data[i] -= *learning_rate * layer->dweights->data[i] /
                                           (sqrt(layer->cache_weights->data[i]) + epsilon);
    }

    // BIASES
    #pragma omp for schedule(dynamic) // Paralellize it (schedule dynamic helps allocate resources)
    for (int i = 0; i < layer->biases->dim1 * layer->biases->dim2; i++) {
        // Update cache for biases
        layer->cache_bias->data[i] = decay_rate * layer->cache_bias->data[i] +
                                                (1.0 - decay_rate) * layer->dbiases->data[i] * layer->dbiases->data[i];
        // Update biases
        layer->biases->data[i] -= *learning_rate * layer->dbiases->data[i] /
                                           (sqrt(layer->cache_bias->data[i]) + epsilon);
    }
}

void update_params_adam (layer_dense* layer, double* learning_rate, double decay_rate, double beta_1, 
                double beta_2, double epsilon, int t, bool correctBias) {
    
    // Check memory allocation for momentums and cache
    if (layer->w_velocity->data == NULL || layer->b_velocity->data == NULL) {
        fprintf(stderr, "Error: Momentum data in adam optimizer not initialized.\n");
        exit(1);
    }

    if (layer->cache_weights->data == NULL || layer->cache_bias->data == NULL) {
        fprintf(stderr, "Error: Cache data in adam optimizer not initialized. \n");
        exit(1);
    }

    // Check Dimensions
    if (layer->w_velocity->dim1 != layer->dweights->dim1 || layer->w_velocity->dim2 != layer->dweights->dim2) {
        fprintf(stderr, "Error: w_velocity dimensions do not match dweights.\n");
        exit(1);
    }

    if (layer->b_velocity->dim1 != layer->dbiases->dim1 || layer->b_velocity->dim2 != layer->dbiases->dim2) {
        fprintf(stderr, "Error: b_velocity dimensions do not match dbiases.\n");
        exit(1);
    }

    if (layer->cache_weights->dim1 != layer->dweights->dim1 || layer->cache_weights->dim2 != layer->dweights->dim2) {
        fprintf(stderr, "Error: cach_weights dimensions do not match dweights.\n");
        exit(1);
    }

    if (layer->cache_bias->dim1 != layer->dbiases->dim1 || layer->cache_bias->dim2 != layer->dbiases->dim2) {
        fprintf(stderr, "Error: cach_bias dimensions do not match dbiases.\n");
        exit(1);
    }

    // Apply learning rate decay (if decay factor is specified)
    if (decay_rate > 1e-10) {
        *learning_rate = *learning_rate / (1.0 + decay_rate * t);
    }

    // Weights
    #pragma omp parallel num_threads(8) 
    {
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < layer->dweights->dim1 * layer->dweights->dim2; i++) {
            // Update Momentum
            layer->w_velocity->data[i] = beta_1 * layer->w_velocity->data[i] + (1.0 - beta_1) * layer->dweights->data[i];
            
            // Correct Momentum
            if (correctBias) {
                layer->w_velocity->data[i] = layer->w_velocity->data[i] / (1.0 - pow(beta_1, t + 1)); // Bias correction for weights momentum
            }

            // Update cache 
            layer->cache_weights->data[i] = beta_2 * layer->cache_weights->data[i] + (1.0 - beta_2) * layer->dweights->data[i] * layer->dweights->data[i];
            
            // Correct cache
            if (correctBias) {
                layer->cache_weights->data[i] = layer->cache_weights->data[i] / (1.0 - pow(beta_2, t + 1)); // Bias correction for weight cache
            }

            // Update Weights using corrected moments and cache
            layer->weights->data[i] -= (*learning_rate) * layer->w_velocity->data[i] / (sqrt(layer->cache_weights->data[i]) + epsilon);

        }
    }

    // Biases
    #pragma omp parallel num_threads(8) 
    {   
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < layer->dbiases->dim1 * layer->dbiases->dim2; i++) {

                // Update Momentum
                layer->b_velocity->data[i] = beta_1 * layer->b_velocity->data[i] + (1.0 - beta_1) * layer->dbiases->data[i];
                
                // Correct Momentum
                if (correctBias) {
                    layer->b_velocity->data[i] = layer->b_velocity->data[i] / (1.0 - pow(beta_1, t + 1)); // Bias correction for bias momentum
                }
                
                // Update cache 
                layer->cache_bias->data[i] = beta_2 * layer->cache_bias->data[i] + (1.0 - beta_2) * layer->dbiases->data[i] * layer->dbiases->data[i];
                
                // Correct cache
                if (correctBias) {
                    layer->cache_bias->data[i] = layer->cache_bias->data[i] / (1.0 - pow(beta_2, t + 1)); // Bias correction for bias cache
                }

                // Update Bias using corrected moments and cache
                layer->biases->data[i] -= (*learning_rate) * layer->b_velocity->data[i] / (sqrt(layer->cache_bias->data[i]) + epsilon);

            }
    }
}


