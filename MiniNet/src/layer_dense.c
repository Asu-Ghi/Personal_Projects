#include "layer_dense.h"

#define VALIDATE_FLAG // Flag to determine if in training or validation pass.

//////////////////////////////////////////////////// MISC METHODS ///////////////////////////////////////////////////////////////////////////

void clip_gradients(matrix* gradients, double clip_value) {

    // Calculate l2 norm for gradients 
    double norm = vector_dot_product(gradients, gradients); // already supports parallel
    // normalize the dot product
    norm = sqrt(norm);
    // if norm > clip value, scale gradients
    if (norm > clip_value) {
        // scale each gradient by the ratio of clip_value / norm
        double scaling_factor = clip_value / norm;
        // scale gradients by factor
        matrix_scalar_mult(gradients, scaling_factor); // already supports parallel
    }
}

void apply_drop_out(layer_dense* layer, double drop_out_rate) {

    // Free and allocate memory for binary mask if it exists
    if (layer->binary_mask != NULL) {
        free_matrix(layer->binary_mask);
        layer->binary_mask = allocate_matrix(layer->post_activation_output->dim1,
                                    layer->post_activation_output->dim2);
    }

    // Allocate memory if it doesnt exist
    else {
        layer->binary_mask = allocate_matrix(layer->post_activation_output->dim1,
                                    layer->post_activation_output->dim2);        
    }

    // Sanity check
    if (layer->post_activation_output->dim1 != layer->binary_mask->dim1 ||
        layer->post_activation_output->dim2 != layer->binary_mask->dim2) {
            fprintf(stderr, "Dimension mismatch, apply dropouts.\n");
            exit(1);
    }

#ifdef ENABLE_PARALLEL // Parallel Approach
int rows_output = layer->post_activation_output->dim1;
int cols_output = layer->post_activation_output->dim2;

#pragma omp parallel
{
    unsigned int seed = omp_get_thread_num();  // Generate a unique seed per thread
    int thread_id = omp_get_thread_num(); // Get current thread id
    int total_threads = omp_get_num_threads(); // Get total num threads
    int rows_per_thread = (rows_output + total_threads - 1) / total_threads; // Get num rows to calc per each thread
    int start_row = rows_per_thread * thread_id; // Get start row for unique thread
    int end_row = rows_per_thread * thread_id + rows_per_thread; // Get end row for unique thread

    // Check to see if in bounds of thread calculations
    if (end_row > rows_output) {
        end_row = rows_output;
    }
    // Iterate through every batch example in layer output
    for (int i = start_row; i < end_row; i++) {
        // Iterate through every neuron 
        for (int j = 0; j < cols_output; j++) {

            // Generate random num between 0 and 1
            double r = (double)rand_r(&seed) / RAND_MAX;

            // If less than drop out rate, drop neuron output
            if (r < drop_out_rate) {
                layer->post_activation_output->data[i * cols_output + j] = 0;
                layer->binary_mask->data[i * cols_output + j] = 0;
            }

            // Scale output by 1/(1- drop_out_rate)
            else {
                layer->post_activation_output->data[i * cols_output + j] /= (1-drop_out_rate);
                layer->binary_mask->data[i * cols_output + j] = 1 ;
            }
        }
    }

}

#else // Sequential Approach
    unsigned int seed = time(NULL);

    for (int i = 0; i < layer->post_activation_output->dim1; i++) {
        // Iterate through every neuron 
        for (int j = 0; j < layer->post_activation_output->dim2; j++) {

            // Generate random num between 0 and 1
            double r = (double)rand_r(&seed) / RAND_MAX;

            // If less than drop out rate, drop neuron output
            if (r < drop_out_rate) {
                layer->post_activation_output->data[i * layer->post_activation_output->dim2 + j] = 0;
                layer->binary_mask->data[i * layer->post_activation_output->dim2 + j] = 0;
            }
            
            // Scale output by 1/(1- drop_out_rate)
            else {
                layer->post_activation_output->data[i * layer->post_activation_output->dim2 + j] /= (1-drop_out_rate);
                layer->binary_mask->data[i * layer->post_activation_output->dim2 + j] = 1 ;
            }
        }
    }

#endif

}

layer_dense* init_layer(int num_inputs, int num_neurons, ActivationType activation, OptimizationType optimization) {
    
    // initialize a layer dense object
    layer_dense* layer_ = malloc(sizeof(layer_dense));
    layer_->num_inputs = num_inputs;
    layer_->num_neurons = num_neurons;

    // point memory to null for vars that are instantiated in forwards and backwards
    layer_->inputs = NULL;
    layer_->dinputs = NULL;
    layer_->pre_activation_output = NULL;
    layer_->post_activation_output = NULL;
    layer_->dweights = NULL;
    layer_->dbiases = NULL;

    // init layer id
    layer_->id = -1;

    // initi clip val to 0
    layer_->clip_value = 0;

    // init layer is training
    layer_->is_training = true;

    // Allocate memory for weights
    layer_->weights = allocate_matrix(num_inputs, num_neurons);

    // Allocate memory for biases
    layer_->biases = allocate_matrix(1, num_neurons);

    // randomize weights
    srand(time(NULL));  // Seed random number with current time
    // srand(42);
    for (int i = 0; i < num_neurons * num_inputs; i++){
        // Random between -1 and 1 scaled by sqrt(1/n)
        // He initialization
        layer_->weights->data[i] = sqrt(1.0 / num_inputs) * ((double)rand() / RAND_MAX * 2.0 - 1.0);  

        // Xavier init
        // layer_->weights->data[i] = sqrt(1.0 / (num_inputs + num_neurons)) * ((double)rand() / RAND_MAX * 2.0 - 1.0);

    }

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

    // Init lambda for regularization
    layer_->lambda_l1 = 0.001;
    layer_->lambda_l2 = 0.01;

    // Initialize velocity for weights
    layer_->w_velocity = allocate_matrix(layer_->weights->dim1, layer_->weights->dim2);

    // Initialize velocity for biases
    layer_->b_velocity = allocate_matrix(layer_->biases->dim1, layer_->biases->dim2);
 
    // Initialize cache weights and biases for the layer
    layer_->cache_bias = allocate_matrix(layer_->biases->dim1, layer_->biases->dim2);

    layer_->cache_weights = allocate_matrix(layer_->weights->dim1, layer_->weights->dim2);

    // return layer dense object
    return layer_;
}

void free_layer(layer_dense* layer) {
    // Free weights
    free(layer->weights->data);
    free(layer->weights);
    layer->weights = NULL;

    // Free biases
    free(layer->biases->data);
    free(layer->biases);
    layer->biases = NULL;

    // Free dweights
    if (layer->dweights != NULL) {
        free(layer->dweights->data);
        free(layer->dweights); 
        layer->dweights = NULL; 
    }

    // Free dbiases
    if (layer->dbiases != NULL) {
        free(layer->dbiases->data);
        free(layer->dbiases);     
        layer->dbiases = NULL;
    }

    // If inputs != NULL so do these other variables
    if (layer->inputs != NULL) {
        free(layer->inputs->data);
        free(layer->inputs);
        layer->inputs = NULL;
    }
    if (layer->dinputs != NULL) {
        free(layer->dinputs->data);
        free(layer->dinputs);
        layer->dinputs = NULL;
    }
    if (layer->pre_activation_output != NULL) {
        free(layer->pre_activation_output->data);
        free(layer->pre_activation_output);
        layer->pre_activation_output = NULL;
    }
    if (layer->post_activation_output != NULL) {
        free(layer->post_activation_output->data);
        free(layer->post_activation_output);    
        layer->post_activation_output = NULL;
    }

    free(layer->w_velocity->data);
    free(layer->w_velocity);
    layer->w_velocity = NULL;

    free(layer->b_velocity->data);
    free(layer->b_velocity);
    layer->b_velocity = NULL;
  
    free(layer->cache_weights->data);
    free(layer->cache_weights);
    layer->cache_weights = NULL;
  
    free(layer->cache_bias->data);
    free(layer->cache_bias);
    layer->cache_bias = NULL;

    free(layer);
    layer = NULL;
}

void free_memory(layer_dense* layer) {
    
    // Free dweights
    if (layer->dweights != NULL) {
        free(layer->dweights->data);
        free(layer->dweights); 
        layer->dweights = NULL; 
    }
    // Free dbiases
    if (layer->dbiases != NULL) {
        free(layer->dbiases->data);
        free(layer->dbiases);     
        layer->dbiases = NULL;
    }

    // If inputs != NULL so do these other variables
    if (layer->inputs != NULL) {
        free(layer->inputs->data);
        free(layer->inputs);
        layer->inputs = NULL;
    }
    if (layer->dinputs != NULL) {
        free(layer->dinputs->data);
        free(layer->dinputs);
        layer->dinputs = NULL;
    }
    if (layer->pre_activation_output != NULL) {
        free(layer->pre_activation_output->data);
        free(layer->pre_activation_output);
        layer->pre_activation_output = NULL;
    }
    if (layer->post_activation_output != NULL) {
        free(layer->post_activation_output->data);
        free(layer->post_activation_output);    
        layer->post_activation_output = NULL;
    }

    // Free binary mask for dropout
    if (layer->binary_mask != NULL) {
        free(layer->binary_mask->data);
        free(layer->binary_mask);
        layer->binary_mask = NULL;
    }

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

//////////////////////////////////////////////////// LOSS METHODS ///////////////////////////////////////////////////////////////////////////

matrix* loss_categorical_cross_entropy(matrix* true_pred, layer_dense* last_layer, ClassLabelEncoding encoding) {

    // check if predictions and true values dim1 match in size
    if(last_layer->post_activation_output->dim1 != true_pred->dim1) {
        fprintf(stderr, "Mismatch in prediction batch size and true value size. \n");
        exit(1);
    }

    // initialize losses data.
    matrix* losses = allocate_matrix(last_layer->post_activation_output->dim1, 1);

    // one hot encoded assumption
    if(encoding == ONE_HOT) {
        
        // check if one hot is the correct size
        if (last_layer->post_activation_output->dim2 != true_pred->dim2) {
            fprintf(stderr, "Error: Dimension 2 for one hot vectors and predictions do not match.\n");
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
        exit(1);
    }

    // return losses
    return(losses);
}

matrix* loss_binary_cross_entropy(layer_dense* layer, matrix* Y) {
    // Check for dimension compatibility
    if (layer->post_activation_output->dim1 != Y->dim1 || Y->dim2 != 1) {
        fprintf(stderr, "Error: Mismatch between prediction and true label dimensions.\n");
        exit(1);
    }

    // Initialize loss matrix
    matrix* losses = malloc(sizeof(matrix));
    losses->dim1 = Y->dim1;
    losses->dim2 = 1;
    losses->data = (double*) calloc(Y->dim1, sizeof(double));

    if (losses->data == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for losses.\n");
        exit(1);
    }

    // Iterate over each sample
    for (int i = 0; i < Y->dim1; i++) {
        double true_label = Y->data[i];
        double predicted_sample = layer->post_activation_output->data[i];

        // Clip predictions to avoid log(0) errors
        if (predicted_sample < 1e-15) predicted_sample = 1e-15;
        if (predicted_sample > 1 - 1e-15) predicted_sample = 1 - 1e-15;

        // Binary cross-entropy loss calculation
        double loss = -(
            true_label * log(predicted_sample) + 
            (1 - true_label) * log(1 - predicted_sample)
        );

        // Store loss
        losses->data[i] = loss;
    }

    return losses;
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
    for (int i = 0; i < layer->weights->dim1 * layer->weights->dim2; i++) {
        l1_w += fabs(layer->weights->data[i]);
        l2_w += layer->weights->data[i] * layer->weights->data[i];
    }


    // Bias regularization L1 and L2
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

    // Allocate memory for layer input 
    if (layer->inputs == NULL) {
        layer->inputs = allocate_matrix(inputs->dim1, inputs->dim2);
    } 

    // Allocate memory for derivative of inputs
    if (layer->dinputs == NULL) {
        layer->dinputs = allocate_matrix(inputs->dim1, inputs->dim2);
    }

    // Copy inputs into layer structure
    memcpy(layer->inputs->data, inputs->data, layer->inputs->dim1 * layer->inputs->dim2 * sizeof(double));

    // Allocate memory for pre activation outputs
    if (layer->pre_activation_output == NULL) {
        layer->pre_activation_output = allocate_matrix(inputs->dim1, layer->num_neurons);
    }
    
    // Calculate Z
    matrix* mult_matrix = matrix_mult(inputs, layer->weights); // supports parallel

    // Add biases for the layer to the batch output data
    #pragma omp for collapse(2) schedule(static)
    for (int i = 0; i < layer->pre_activation_output->dim1; i++) {
        // output dim2-> num neurons
        for (int j = 0; j < layer->pre_activation_output->dim2; j++) {
            layer->pre_activation_output->data[i * layer->pre_activation_output->dim2 + j] = mult_matrix->data[i * layer->pre_activation_output->dim2 + j] + layer->biases->data[j];
        }
    }

    // relu activation
    if (layer->activation == RELU) {
        // Handles memory allocation for post activation outputs
        layer->post_activation_output = forward_reLu(layer->pre_activation_output);
    } 

    // softmax activation
    else if(layer->activation == SOFTMAX) {
        // Handles memory allocation for post activation outputs
        layer->post_activation_output = forward_softMax(layer->pre_activation_output);
    }

    // sigmoid activation
    else if(layer->activation == SIGMOID) {
        // Handles memory allocation for post activation outputs
        layer->post_activation_output = forward_sigmoid(layer->pre_activation_output);
    }

    // linear activation
    else if (layer->activation == LINEAR) {
        layer->post_activation_output = forward_linear(layer->pre_activation_output);
    }

    // Check memory allocation
    if (layer->post_activation_output->data == NULL) {
        fprintf(stderr, "Error in memory allocation for post_activation_output in forward pass.\n");
        exit(1);
    }

    // Apply dropout
    if ((layer->drop_out_rate > 0.0 && layer->is_training == true) && (layer->activation != SOFTMAX && layer->activation != SIGMOID)) {
        apply_drop_out(layer, layer->drop_out_rate);
    }

    // Free uneeded memory
    free_matrix(mult_matrix);
}

matrix* forward_reLu(matrix* Z) {

    // Allocate memory for return matrix
    matrix* outputs = allocate_matrix(Z->dim1, Z->dim2);

#ifdef ENABLE_PARALLEL // Parallel Approach

    #pragma omp for schedule(static)
    for (int i = 0; i < Z->dim1 * Z->dim2; i++){

        // if the input value is <= 0, rectify it to 0 (otherwise, leave it unchanged)
        if(Z->data[i] <= 0) {
            outputs->data[i] = 0;
        }
        else {
            outputs->data[i] = Z->data[i];
        }
    }

#else // Sequential Approach

    // iterate through every point in the batch input
    for (int i = 0; i < Z->dim1 * Z->dim2; i++){

        // if the input value is <= 0, rectify it to 0 (otherwise, leave it unchanged)
        if(Z->data[i] <= 0) {
            outputs->data[i] = 0;
        }
        else {
            outputs->data[i] = Z->data[i];
        }
    }
#endif

    // Return post activation outputs.
    return outputs;
}

matrix* forward_softMax(matrix* Z) {

    // Allocate memory for outputs
    matrix* outputs = allocate_matrix(Z->dim1, Z->dim2);

#ifdef ENABLE_PARALLEL // Parallel approach (Not implemented)

    // iterate over the batch
#pragma omp parallel for schedule(static)
for(int i = 0; i < outputs->dim1; i++) {

    // step 1: Subtract maximum value from each value in the input batch for numerical stability
    double max = -DBL_MAX;
    for(int j = 0; j < Z->dim2; j++) {
        if (Z->data[i * Z->dim2 + j] > max) {
            max = Z->data[i * Z->dim2 + j];
        }
    }

    // step 2: calculate exponentials and sum them
    double* exp_values = (double*) calloc(Z->dim2, sizeof(double));
    double sum = 0.0;

    // Parallelize the summation loop using reduction on sum
    #pragma omp parallel for reduction(+:sum)
    for(int j = 0; j < Z->dim2; j++) {
        exp_values[j] = exp(Z->data[i * Z->dim2 + j] - max);
        sum += exp_values[j];
    }

    // step 3: normalize exponentials by dividing by the sum to get probabilities
    for(int j = 0; j < outputs->dim2; j++) {
        outputs->data[i * outputs->dim2 + j] = exp_values[j] / sum;
    }

    // step 4: free temp exp values
    free(exp_values);
}


# else // Sequential Approach

    // iterate over the batch
    for(int i = 0; i < outputs -> dim1; i++) {

        //step 1: Subtract maximum value from each value in the input batch to ensure numerical stability (no large exponentiations)
        double max = -DBL_MAX;
        for(int j = 0; j < Z->dim2; j++){
            if (Z->data[i*Z->dim2 + j] > max) {
                max = Z->data[i*Z->dim2 + j];
            }
        }

        // step 2: calculate exponentials and sum them
        double* exp_values = (double*) calloc(Z->dim2, sizeof(double));
        double sum = 0.0;
        for(int j = 0; j < Z -> dim2; j++) {
            exp_values[j] = exp(Z->data[i * Z->dim2 + j] - max);
            sum += exp_values[j];
        }

        // step 3: normalize exponentials by dividing by the sum to get probabilities
        for(int j = 0; j < outputs->dim2; j++) {
            outputs->data[i * outputs->dim2 + j] = exp_values[j] / sum;
        }

        // step 4: free temp exp values 
        free(exp_values);
    }

#endif 

    // Return post activation outputs
    return outputs;
}

matrix* forward_sigmoid(matrix* inputs) {
    // Allocate Memory for outputs
    matrix* outputs = allocate_matrix(inputs->dim1, inputs->dim2);
   int row_outputs = outputs->dim1;
   int cols_outputs = outputs->dim2;

#ifdef ENABLE_PARALLEL // Parallel Approach
#pragma omp parallel 
{
    int thread_id = omp_get_thread_num(); // Get current thread id
    int total_threads = omp_get_num_threads(); // Get total num threads
    int rows_per_thread = (row_outputs + total_threads - 1) / total_threads; // Get num rows to calc per each thread
    int start_row = rows_per_thread * thread_id; // Get start row for unique thread
    int end_row = rows_per_thread * thread_id + rows_per_thread; // Get end row for unique thread

    // check bounds
    if (end_row > row_outputs) {
        end_row = row_outputs;
    }

    // Each thread gets a region to compute the softmax
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < cols_outputs; j++) {
            outputs->data[i * cols_outputs + j] = 1.0  / (1 + exp(-1 * inputs->data[i * cols_outputs + j]));
        }
    }

}

#else // Sequential Approach

    // Inputs corespond to the outputs (z of the current layer)
    for (int i = 0; i < outputs->dim1; i++) {
        for (int j = 0; j < outputs->dim2; j++) {
            outputs->data[i * outputs->dim2 + j] = 1.0  / (1 + exp(-1 * inputs->data[i * inputs->dim2 + j]));
        }
    }

#endif
    // return outputs
    return outputs;
}

matrix* forward_linear(matrix* inputs) {
    matrix* outputs = allocate_matrix(inputs->dim1, inputs->dim2);
    memcpy(outputs->data, inputs->data, inputs->dim1 * inputs->dim2 * sizeof(double));
    return outputs;
}

////////////////////////////////////////////////// BACKWARD METHODS ///////////////////////////////////////////////////////////////////////////

void backward_reLu(matrix* input_gradients, layer_dense* layer) {
    /*
    Find Gradient of a ReLu layer.
    STEPS:
        > Gradients for DROPOUT
        > Gradients for ACTIVATION
        > Gradients for WEIGHTS
        > Gradients for BIASES
        > Gradients for INPUTS
        > Pass Gradients for INPUTS to the layer previous.
    */

    // Allocate memory for dweights and dbiases if not already done
    if (layer->dweights == NULL) {
        layer->dweights = allocate_matrix(layer->num_inputs, layer->num_neurons);   
    }

    if (layer->dbiases == NULL) {
        layer->dbiases = allocate_matrix(1, layer->num_neurons);     
    }
    
    // If using dropout, apply
    if (layer->drop_out_rate > 0.0) {
        apply_dropout_gradients(input_gradients, layer); // supports parallel
    }

    // Allocate memory for ReLU gradient
    matrix* relu_gradients = allocate_matrix(layer->pre_activation_output->dim1,
                                        layer->pre_activation_output->dim2);

    // Iterate through every value in layer post activation output to get relu gradients
    calculate_relu_gradients(relu_gradients, layer); // supports parallel
    #ifdef DEBUG
    printf("Layer %d \n", layer->id);
    printf("Input Gradients with dropout \n");
    print_matrix(input_gradients);
    printf("Relu Gradients \n");
    print_matrix(relu_gradients);
    #endif

    // Check dimensions for element by element multiplication of input gradients and relu gradients.
    if (input_gradients->dim1 != relu_gradients->dim1 || input_gradients->dim2 != relu_gradients->dim2) {
        fprintf(stderr,"Error: Dimensionality mismatch between relu gradients and input gradients in backwards relu.\n");
        exit(1);
    }

    // Element by element mult of input gradients and relu gradients, free relu gradients after 
    matrix* relu_and_input_grads = element_matrix_mult(relu_gradients, input_gradients); // supports parallel

    #ifdef DEBUG
    printf("Input gradients with dropout * relu gradients\n");
    print_matrix(relu_and_input_grads);
    #endif

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
    if(inputs_transposed->dim2 != relu_and_input_grads-> dim1) {
        fprintf(stderr, "Error: Dimensionality mismatch between inputs_transposed and relu_gradients in weight calculation.\n");
        exit(1);
    }

    // Perform the dot product
    layer->dweights = matrix_mult(inputs_transposed, relu_and_input_grads); // supports parallel
    #ifdef DEBUG
    printf("Inputs\n");
    print_matrix(layer->inputs);
    printf("Inputs transposed \n");
    print_matrix(inputs_transposed);
    printf("Weight gradients\n");
    print_matrix(layer->dweights);
    #endif

    // If clipping gradients, apply
    if (layer->clip_value > 0) {
        clip_gradients(layer->dweights, layer->clip_value); // supports parallel
    }

    // Sum the relu gradients for each example in the batch of inputs
    calculate_bias_gradients(relu_and_input_grads, layer); // supports parallel

    #ifdef debug
    printf("Bias Gradients\n");
    print_matrix(layer->dbiases);
    #endif
    // If clipping gradients, apply
    if (layer->clip_value > 0) {
        clip_gradients(layer->dbiases, layer->clip_value); // supports parallel
    }

    // If using l1 and l2 regularization, apply
    if (layer->useRegularization) {
        apply_regularization_gradients(layer); // supports parallel
    }

    #ifdef DEBUG
    printf("Weights after regularization gradients\n");
    print_matrix(layer->dweights);
    printf("Biases after regularization gradients\n");
    print_matrix(layer->dbiases);
    #endif 

    // Calculate gradients for the input

    // Transpose weights
    matrix* weights_transposed = transpose_matrix(layer->weights);

    // Check dimensions
    if (relu_and_input_grads->dim2 != weights_transposed->dim1) {
        fprintf(stderr, "Error: Dimensionality mismatch between relu gradients and weights transposed in backwards RELU\n");
        exit(1);
    }

    // Dot product of relu_gradients and weights transposed
    layer->dinputs = matrix_mult(relu_and_input_grads, weights_transposed); // supports parallel

    #ifdef DEBUG
    printf("Input gradients\n");
    print_matrix(layer->dinputs);
    #endif

    // If clipping gradients, apply
    if (layer->clip_value > 0) {
        clip_gradients(layer->dinputs, layer->clip_value); // supports parallel
    }

    // free memory
    free_matrix(relu_gradients);
    free_matrix(relu_and_input_grads);
    free_matrix(inputs_transposed);
    free_matrix(weights_transposed);

}

void backward_linear(matrix* input_gradients, layer_dense* layer) {
    /*
    Find Gradients of a Linear Activation layer.
    STEPS:
        > Gradients for DROPOUT
        > Gradients for ACTIVATION
        > Gradients for WEIGHTS
        > Gradients for BIASES
        > Gradients for INPUTS
        > Pass Gradients for INPUTS to the layer previous.
    */

  // Allocate memory for dweights and dbiases if not already done
    if (layer->dweights == NULL) {
        layer->dweights = allocate_matrix(layer->num_inputs, layer->num_neurons);   
    }

    if (layer->dbiases == NULL) {
        layer->dbiases = allocate_matrix(1, layer->num_neurons);     
    }

    // Calculate linear gradients -> Do nothing, gradient of linear activation = 1 * dvalues

    // If using dropout, apply
    if (layer->drop_out_rate > 0.0) {
        apply_dropout_gradients(input_gradients, layer); // supports parallel
    }

// Calculate weight gradients

    // Transpose inputs
    matrix* inputs_transposed = transpose_matrix(layer->inputs);

    // Perform the dot product
    layer->dweights = matrix_mult(inputs_transposed, input_gradients); // supports parallel

    // If clipping gradients, apply
    if (layer->clip_value > 0) {
        clip_gradients(layer->dweights, layer->clip_value); // supports parallel
    }

    // Sum the relu gradients for each example in the batch of inputs
    calculate_bias_gradients(input_gradients, layer); // supports parallel

    // If clipping gradients, apply
    if (layer->clip_value > 0) {
        clip_gradients(layer->dbiases, layer->clip_value); // supports parallel
    }

    // If using l1 and l2 regularization, apply
    if (layer->useRegularization) {
        apply_regularization_gradients(layer); // supports parallel
    }

    // Calculate gradients for the input

    // Transpose weights
    matrix* weights_transposed = transpose_matrix(layer->weights);

    // Dot product of relu_gradients and weights transposed
    layer->dinputs = matrix_mult(input_gradients, weights_transposed); // supports parallel

    // If clipping gradients, apply
    if (layer->clip_value > 0) {
        clip_gradients(layer->dinputs, layer->clip_value); // supports parallel
    }

    // free memory
    free_matrix(inputs_transposed);
    free_matrix(weights_transposed);
}

void backwards_softmax_and_loss(matrix* true_labels, layer_dense* layer) {
    // Check dimensionality
    if (layer->post_activation_output->dim1 != true_labels->dim1 || layer->post_activation_output->dim2 != true_labels->dim2) {
        fprintf(stderr, "Error: Dimensionality mismatch between true labels and predictions in backwards softmax.\n");
        exit(1);
    }

    // Allocate memory for dweights and dbiases if not already done
    if (layer->dweights == NULL) {
        layer->dweights = allocate_matrix(layer->num_inputs, layer->num_neurons); 
    }

    if (layer->dbiases == NULL) {
        layer->dbiases = allocate_matrix(1, layer->num_neurons);   
    }

    // Calculate softmax loss partial derivatives

    // Allocate memory for loss gradients
    matrix* loss_gradients = allocate_matrix(layer->post_activation_output->dim1,
                                             layer->post_activation_output->dim2);

    // For each example in the input batch calculate softmax loss gradients
    calculate_softmax_gradients(loss_gradients, layer, true_labels); // modifies loss gradients

    // Calculate layer weight derivatives
    // dot product of inputs for the layer and loss_gradients calculated above.

    // Transpose layer inputs
    matrix* inputs_T = transpose_matrix(layer->inputs);

    // Check dimensions
    if (inputs_T->dim2 != loss_gradients->dim1) {
        fprintf(stderr, "Error: Dimensionality mismatch for inputs transposed in backwards softmax.\n");
        exit(1);
    }

    /*
    dweights and dbiases not used in backprop here, but later in optimization.
    */

    // Calculate dweights -> dont need to allocate memory as matrix_mult does that.
    layer->dweights = matrix_mult(inputs_T, loss_gradients); // supports parallel

    // If clipping gradients, apply
    if (layer->clip_value > 0) {
        clip_gradients(layer->dweights, layer->clip_value); //supports parallel
    }

    // Calculate layer bias derivatives

    // Sum the loss gradients for each example in the batch of inputs
    calculate_bias_gradients(loss_gradients, layer); // supports parallel, modifies dbiases through layer object

    // If clipping gradients, apply
    if (layer->clip_value > 0) {
        clip_gradients(layer->dbiases, layer->clip_value); // supports parallel
    }

    // Add regularization derivatives to dweights and dbiases
    // Check if using regularization
    if (layer->useRegularization) {
        apply_regularization_gradients(layer); // supports parallel
    }

    // Backpropogate derivatives for previous layer
    // Transpose weights for layer
    matrix* weights_transposed = transpose_matrix(layer->weights); 

    // Check dimensions
    if(loss_gradients->dim2 != weights_transposed->dim1) {
        fprintf(stderr, "Error: Dimensionality mismatch for weights transposed in backprop softmax function.\n");
        exit(1);    
    }

    // Calculate backprop derivative to pass to layer previous
    layer->dinputs = matrix_mult(loss_gradients, weights_transposed); // supports parallel

    // If clipping gradients, apply
    if (layer->clip_value > 0) {
        clip_gradients(layer->dinputs, layer->clip_value); // supports parallel
    }

    // free memory
    free_matrix(loss_gradients);
    free_matrix(inputs_T);
    free_matrix(weights_transposed);
}

void backwards_sigmoid_and_loss(matrix* true_labels, layer_dense* layer) {
    /*
    Calculate derivative of loss with respect to inputs
    */

    // Check dims
    // Both should be size (num batch inputs x 1 )
    if (true_labels->dim2 != 1 || layer->post_activation_output->dim2 != 1) {
        fprintf(stderr, "Error, mismatching dimensions between true labels and post act outputs in backward sigmoid and loss.\n");
        exit(1);
    }
    if (true_labels->dim1 != layer->post_activation_output->dim1) {
        fprintf(stderr, "Error, mismatching dimensions between true labels and post act outputs in backward sigmoid and loss.\n");
        exit(1);
    }

    // Allocate and check memory for loss_gradients
    matrix* loss_gradients = allocate_matrix(layer->post_activation_output->dim1, 
                                            layer->post_activation_output->dim2);


    // y_hat - y_pred
    for (int i = 0; i < layer->post_activation_output->dim1 * layer->post_activation_output->dim2; i++) {
        loss_gradients->data[i] = layer->post_activation_output->data[i] - true_labels->data[i];
    }

    /*
    Step 1: Calculate Sigmoid Gradients
    The element by element product of the sigmoid outputs and 1 - the sigmoid outputs and the input
    gradients from loss.
    */   

    matrix* sigmoid_gradients = allocate_matrix(layer->post_activation_output->dim1, 
                                            layer->post_activation_output->dim2);

    // Calculate derivative of sigmoid function
    calculate_sigmoid_gradients(sigmoid_gradients, layer); // supports parallel

    // Calculate derivative with respect to input gradients
    sigmoid_gradients = element_matrix_mult(sigmoid_gradients, loss_gradients); // supports parallel


    /*
    Step 2: Calculate Weight Gradients
    The dot product of the layers Inputs Transposed and the sigmoid gradients calculated above.
    */
    matrix* inputs_transposed = transpose_matrix(layer->inputs);

    layer->dweights = matrix_mult(inputs_transposed, sigmoid_gradients);

    /*
    Step 3: Calculate the Bias Gradients
    The sum of the input gradients across each sample in the batch.
    */

    calculate_bias_gradients(sigmoid_gradients, layer); // directly accesses the layer biases

    /*
    Step 4: Calculate the Input Gradients
    The dot product of the actvation functions gradients and weights transposed.
    */

    matrix* weights_transposed = transpose_matrix(layer->weights);

    layer->dinputs = matrix_mult(sigmoid_gradients, weights_transposed);

    // Free memory
    free_matrix(weights_transposed);
    free_matrix(loss_gradients);
    free_matrix(inputs_transposed);
    free_matrix(sigmoid_gradients);
    exit(1);
}

void calculate_relu_gradients(matrix* relu_gradients, layer_dense* layer) {

    // Check if correct dimensions 
    if (relu_gradients->dim1 != layer->pre_activation_output->dim1 || relu_gradients->dim2 != layer->pre_activation_output->dim2) {
        fprintf(stderr, "Error, dimension mismatch between relu gradients and pre act outputs in calc_relu_gradients\n");
        exit(1);
    }


#ifdef ENABLE_PARALLEL 
    #pragma omp parallel
    {

    #pragma omp for 
    for (int i = 0; i < layer->pre_activation_output->dim1 * layer->pre_activation_output->dim2; i++) {
        if (layer->pre_activation_output->data[i] >= 0) {
            relu_gradients->data[i] = 1;
        }
        else {
            relu_gradients->data[i] = 0;
        }
    }

    }
 
#else
    // Iterate through every value in layer post activation output to get relu gradients
    for (int i = 0; i < layer->pre_activation_output->dim1 * layer->pre_activation_output->dim2; i++) {
        if (layer->pre_activation_output->data[i] >= 0) {
            relu_gradients->data[i] = 1;
        }
        else {
            relu_gradients->data[i] = 0;
        }
    }

#endif

}

void calculate_softmax_gradients(matrix* softmax_gradients, layer_dense* layer, matrix* true_labels) {
    // Check if correct dimensions 
    if (softmax_gradients->dim1 != layer->post_activation_output->dim1 || softmax_gradients->dim2 != layer->post_activation_output->dim2) {
        fprintf(stderr, "Error, dimension mismatch between softmax gradients and post act outputs in calc_softmax_gradients\n");
        exit(1);
    }

    if (softmax_gradients->dim1 != true_labels->dim1 || softmax_gradients->dim2 != true_labels->dim2) {
        fprintf(stderr, "Error, dimension mismatch between softmax gradients and true_labels in calc_softmax_gradients\n");
        exit(1);
    } 

#ifdef ENABLE_PARALLEL
    int row_gradients = softmax_gradients->dim1;
    int col_gradients = softmax_gradients->dim2;

#pragma omp parallel
{
    int thread_id = omp_get_thread_num(); // Get current thread id
    int total_threads = omp_get_num_threads(); // Get total num threads
    int rows_per_thread = (row_gradients + total_threads - 1) / total_threads; // Get num rows to calc per each thread
    int start_row = rows_per_thread * thread_id; // Get start row for unique thread
    int end_row = rows_per_thread * thread_id + rows_per_thread; // Get end row for unique thread

    // Check to see if in bounds of thread calculations
    if (end_row > row_gradients) {
        end_row = row_gradients;
    }

    for (int i = start_row; i < end_row; i++){
        // For each neuron in the input vector
        for(int j = 0; j < col_gradients; j++) {
            softmax_gradients->data[i * col_gradients + j] = layer->post_activation_output->data[i * col_gradients + j] - 
                                true_labels->data[i * col_gradients + j];
        }
    }   
}

#else
    for (int i = 0; i < softmax_gradients->dim1; i++){
        // For each neuron in the input vector
        for(int j = 0; j < softmax_gradients->dim2; j++) {
            softmax_gradients->data[i * softmax_gradients->dim2 + j] = layer->post_activation_output->data[i * softmax_gradients->dim2 + j] - 
                                true_labels->data[i * softmax_gradients->dim2 + j];
        }
    }   

#endif


}

void calculate_sigmoid_gradients(matrix* sigmoid_gradients, layer_dense* layer) {
    // Check dimensions
    if (sigmoid_gradients->dim1 != layer->post_activation_output->dim1 || 
        sigmoid_gradients->dim1 != layer->post_activation_output->dim2) {
            fprintf(stderr, "Error: Dimensionality mismatch in calculate sigmoid gradients.\n");
            exit(1);
    }

    // Calculate 1 - sigmoid outputs
    // (1 - sig output, use true to indicate abs val of the scalar sums)
    matrix* sigmoid_diff = matrix_scalar_sum(layer->post_activation_output, -1, true); // supports parallel, true indicates use abs 
    
    // Element wise multiplication 
    sigmoid_gradients = element_matrix_mult(layer->post_activation_output, sigmoid_diff); // supports parallel

}   

void calculate_bias_gradients(matrix* input_gradients, layer_dense* layer) {

    // Check dimensions
    if (layer->dbiases->dim2 != input_gradients->dim2) {
        fprintf(stderr, "Dimensionality mismatch in calculate bias gradients.\n");
        exit(1);
    }

#ifdef ENABLE_PARALLEL
    int num_biases = layer->dbiases->dim2;
    int row_gradients = input_gradients->dim1;
    int col_gradients = input_gradients->dim2;

#pragma omp parallel 
{
    int thread_id = omp_get_thread_num(); // Get current thread id
    int total_threads = omp_get_num_threads(); // Get total num threads
    int rows_per_thread = (num_biases + total_threads - 1) / total_threads; // Get num rows to calc per each thread
    int start_row = rows_per_thread * thread_id; // Get start row for unique thread
    int end_row = rows_per_thread * thread_id + rows_per_thread; // Get end row for unique thread

    if (end_row > num_biases) {
        end_row = num_biases;
    }

    // Calculate bias gradients
    for (int j = start_row; j < end_row; j++) {
        for(int i = 0; i < row_gradients; i++) {
            // sum across rows
            layer->dbiases->data[j] += input_gradients->data[i * col_gradients + j];
        }
    }
}
#else

    // Calculate bias gradients
    for (int j = 0; j < layer->dbiases->dim2; j++) {
        for(int i = 0; i < input_gradients->dim1; i++) {
            // sum across rows
            layer->dbiases->data[j] += input_gradients->data[i * input_gradients->dim2 + j];
        }
    }

#endif
}

void apply_regularization_gradients(layer_dense* layer) {  
#ifdef ENABLE_PARALLEL

    // weights
    #pragma omp for
    for (int i = 0; i < layer->dweights->dim1 * layer->dweights->dim2; i++) {
        // L2 gradients
        layer->dweights->data[i] += 2 * layer->lambda_l2 * layer->weights->data[i];

        // L1 gradients (1 if > 0, -1 if < 0)
        layer->dweights->data[i] += layer->lambda_l1 * (layer->weights->data[i] >= 0.0 ? 1.0 : -1.0);
    }
    // biases
    #pragma omp for
    for (int i = 0; i < layer->dbiases->dim1 * layer->dbiases->dim2; i++) {
        // L2 gradients
        layer->dbiases->data[i] += 2 * layer->lambda_l2 * (layer->biases->data[i]);

        // L1 gradients (1 if > 0, -1 if < 0)
        layer->dbiases->data[i] += layer->lambda_l1 * (layer->biases->data[i] >= 0 ? 1.0: -1.0);
    }

#else
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

#endif
}

void apply_dropout_gradients(matrix* input_gradients, layer_dense* layer) {
    // Check dims
    if (input_gradients->dim1 != layer->binary_mask->dim1 || input_gradients->dim2 != layer->binary_mask->dim2) {
        fprintf(stderr, "Error: Mismatching dimensions between input gradients and binary mask in apply dropout gradients.\n");
        exit(1);
    }

#ifdef ENABLE_PARALLEL // Parallel approach

    #pragma omp for schedule(static)
    for (int i = 0; i < input_gradients->dim1 * input_gradients->dim2; i++) {
        input_gradients->data[i] *= layer->binary_mask->data[i];
    }

#else // Sequential approach
    for (int i = 0; i < input_gradients->dim1 * input_gradients->dim2; i++) {
        input_gradients->data[i] *= layer->binary_mask->data[i];
    }


#endif

}

////////////////////////////////////////////////// OPTIMIZER METHODS ///////////////////////////////////////////////////////////////////////////

void update_params_sgd(layer_dense* layer, double* learning_rate, int current_epoch, double decay_rate) {

    // Decay the learning rate after each epoch
    *learning_rate = fmax(*learning_rate * exp(-decay_rate * current_epoch), 0.000001);

#ifdef ENABLE_PARALLEL // Parallel Approach

    // Update weights
    #pragma omp for collapse(2) schedule(static) 
    for (int i = 0; i < layer->num_neurons; i++) {
        for (int j = 0; j < layer->num_inputs; j++) {
            // W = W - learning_rate * dL/dW
            layer->weights->data[i * layer->num_inputs + j] -= *learning_rate * layer->dweights->data[i * layer->num_inputs + j];
            
        }
    }

    // Update biases
    #pragma omp for schedule(static) 
    for(int i = 0; i < layer->num_neurons; i++) {
        // b = b - learning_rate * dL/dB
        layer->biases->data[i] -= *learning_rate * layer->dbiases->data[i];
    }
#else // Sequential Approach

    // Update weights
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
#endif
}

void update_params_sgd_momentum(layer_dense* layer, double* learning_rate, int current_epoch, double decay_rate, double beta) {

    // Decay the learning rate after each epoch
    *learning_rate = fmax(*learning_rate * exp(-decay_rate * current_epoch), 0.001);
    
#ifdef ENABLE_PARALLEL // Parallel Approach

    // Update weights
    #pragma omp for collapse(2) schedule(static)
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
    #pragma omp for schedule(static)
    for(int i = 0; i < layer->num_neurons; i++) {
        // v_t = beta * v_(t-1) + (1 - beta) * dL/dB
        layer->b_velocity->data[i] = beta * layer->b_velocity->data[i] + (1 - beta) * layer->dbiases->data[i];
        // b = b - learning_rate * v_t
        layer->biases->data[i] -= *learning_rate * layer->b_velocity->data[i];
    }

#else // Sequential Approach
    // Update weights
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
    for(int i = 0; i < layer->num_neurons; i++) {
        // v_t = beta * v_(t-1) + (1 - beta) * dL/dB
        layer->b_velocity->data[i] = beta * layer->b_velocity->data[i] + (1 - beta) * layer->dbiases->data[i];
        // b = b - learning_rate * v_t
        layer->biases->data[i] -= *learning_rate * layer->b_velocity->data[i];
    }
#endif
}

void update_params_adagrad(layer_dense* layer, double* learning_rate, int current_epoch, double decay_rate, double epsilon) {

#ifdef ENABLE_PARALLEL // Parallel Approach
    // WEIGHTS
    // Square every element in dweights, add to cache_weights
    #pragma omp for schedule(static)
    for (int i = 0; i < layer->cache_weights->dim1 * layer->cache_weights->dim2; i++) {
        // Calculate cache
        layer->cache_weights->data[i] += layer->dweights->data[i] * layer->dweights->data[i];

        layer->weights->data[i] -= *learning_rate * layer->dweights->data[i] / (sqrt(layer->cache_weights->data[i]) + epsilon);
    }


    // BIASES

    // Square every element in dbiases, add to cache_biases
    #pragma omp for schedule(static) 
    for (int i = 0; i < layer->cache_bias->dim1 * layer->cache_bias->dim2; i++) {
        // Calculate cache
        layer->cache_bias->data[i] += layer->dbiases->data[i] * layer->dbiases->data[i];

        layer->biases->data[i] -= *learning_rate * layer->dbiases->data[i] / (sqrt(layer->cache_bias->data[i]) + epsilon);
    }
#else // Sequential Approach

    // WEIGHTS
    // Square every element in dweights, add to cache_weights
    for (int i = 0; i < layer->cache_weights->dim1 * layer->cache_weights->dim2; i++) {
        // Calculate cache
        layer->cache_weights->data[i] += layer->dweights->data[i] * layer->dweights->data[i];

        layer->weights->data[i] -= *learning_rate * layer->dweights->data[i] / (sqrt(layer->cache_weights->data[i]) + epsilon);
    }

    // BIASES
    // Square every element in dbiases, add to cache_biases
    for (int i = 0; i < layer->cache_bias->dim1 * layer->cache_bias->dim2; i++) {
        // Calculate cache
        layer->cache_bias->data[i] += layer->dbiases->data[i] * layer->dbiases->data[i];

        layer->biases->data[i] -= *learning_rate * layer->dbiases->data[i] / (sqrt(layer->cache_bias->data[i]) + epsilon);
    }

#endif
}

void update_params_rmsprop(layer_dense* layer, double* learning_rate, double decay_rate, double epsilon) {

#ifdef ENABLE_PARALLEL // Parallel Approach
    // WEIGHTS
    #pragma omp for schedule(static) 
    for (int i = 0; i < layer->weights->dim1 * layer->weights->dim2; i++) {
        // Update cache for weights
        layer->cache_weights->data[i] = decay_rate * layer->cache_weights->data[i] +
                                                (1.0 - decay_rate) * layer->dweights->data[i] * layer->dweights->data[i];
        // Update weights
        layer->weights->data[i] -= *learning_rate * layer->dweights->data[i] /
                                           (sqrt(layer->cache_weights->data[i]) + epsilon);
    }

    // BIASES
    #pragma omp for schedule(static) 
    for (int i = 0; i < layer->biases->dim1 * layer->biases->dim2; i++) {
        // Update cache for biases
        layer->cache_bias->data[i] = decay_rate * layer->cache_bias->data[i] +
                                                (1.0 - decay_rate) * layer->dbiases->data[i] * layer->dbiases->data[i];
        // Update biases
        layer->biases->data[i] -= *learning_rate * layer->dbiases->data[i] /
                                           (sqrt(layer->cache_bias->data[i]) + epsilon);
    }

#else // Sequential Approach
    // WEIGHTS
    for (int i = 0; i < layer->weights->dim1 * layer->weights->dim2; i++) {
        // Update cache for weights
        layer->cache_weights->data[i] = decay_rate * layer->cache_weights->data[i] +
                                                (1.0 - decay_rate) * layer->dweights->data[i] * layer->dweights->data[i];
        // Update weights
        layer->weights->data[i] -= *learning_rate * layer->dweights->data[i] /
                                           (sqrt(layer->cache_weights->data[i]) + epsilon);
    }

    // BIASES
    for (int i = 0; i < layer->biases->dim1 * layer->biases->dim2; i++) {
        // Update cache for biases
        layer->cache_bias->data[i] = decay_rate * layer->cache_bias->data[i] +
                                                (1.0 - decay_rate) * layer->dbiases->data[i] * layer->dbiases->data[i];
        // Update biases
        layer->biases->data[i] -= *learning_rate * layer->dbiases->data[i] /
                                           (sqrt(layer->cache_bias->data[i]) + epsilon);
    }

#endif
}

void update_params_adam (layer_dense* layer, double* learning_rate, double decay_rate, double beta_1, 
                double beta_2, double epsilon, int current_epoch, int total_epochs, bool correctBias) {

    // Apply learning rate decay (if decay factor is specified)
    if (decay_rate > 0.0) {
        *learning_rate = fmax(*learning_rate * (1 / (1 + decay_rate * current_epoch)), 1e-6);
    }

#ifdef ENABLE_PARALLEL // Parallel Approach

    // Weights
    #pragma omp for schedule(static)
    for (int i = 0; i < layer->dweights->dim1 * layer->dweights->dim2; i++) {

        // Update Momentum
        layer->w_velocity->data[i] = beta_1 * layer->w_velocity->data[i] + (1.0 - beta_1) * layer->dweights->data[i];
        
        // Correct Momentum
        if (correctBias) {
            long double log_beta_1 = log(beta_1);
            layer->w_velocity->data[i] = (double) layer->w_velocity->data[i] / (1.0 - exp((current_epoch + 1) * log_beta_1)); // Bias correction for weights momentum
        }

        // Update cache 
        layer->cache_weights->data[i] = beta_2 * layer->cache_weights->data[i] + (1.0 - beta_2) * layer->dweights->data[i] * layer->dweights->data[i];
        
        // Correct cache
        if (correctBias) {
            long double log_beta_2 = log(beta_2);
            layer->cache_weights->data[i] = (double) layer->cache_weights->data[i] / (1.0 - exp((current_epoch + 1) * log_beta_2)); // Bias correction for weight cache
        }

        // Update Weights using corrected moments and cache
        layer->weights->data[i] -= (*learning_rate) * layer->w_velocity->data[i] / (sqrt(layer->cache_weights->data[i]) + epsilon);
    }
    

    // Biases
    #pragma omp for schedule(static)
    for (int i = 0; i < layer->dbiases->dim1 * layer->dbiases->dim2; i++) {
        // Update Momentum
        layer->b_velocity->data[i] = beta_1 * layer->b_velocity->data[i] + (1.0 - beta_1) * layer->dbiases->data[i];
        
        // Correct Momentum
        if (correctBias) {
            long double log_beta_1 = log(beta_1);
            layer->b_velocity->data[i] = (double) layer->b_velocity->data[i] / (1.0 - exp((current_epoch + 1) * log_beta_1)); // Bias correction for bias momentum
        }
        
        // Update cache 
        layer->cache_bias->data[i] = beta_2 * layer->cache_bias->data[i] + (1.0 - beta_2) * layer->dbiases->data[i] * layer->dbiases->data[i];
        
        // Correct cache
        if (correctBias) {
            long double log_beta_2 = log(beta_2);
            layer->cache_bias->data[i] = (double) layer->cache_bias->data[i] / (1.0 - exp((current_epoch + 1) * log_beta_2)); // Bias correction for bias cache
        }

        // Update Bias using corrected moments and cache
        layer->biases->data[i] -= (*learning_rate) * layer->b_velocity->data[i] / (sqrt(layer->cache_bias->data[i]) + epsilon);
    }

# else // Sequential Approach
    // Weights
    for (int i = 0; i < layer->dweights->dim1 * layer->dweights->dim2; i++) {

        // Update Momentum
        layer->w_velocity->data[i] = beta_1 * layer->w_velocity->data[i] + (1.0 - beta_1) * layer->dweights->data[i];
        
        // Correct Momentum
        if (correctBias) {
            double log_beta_1 = log(beta_1);
            layer->w_velocity->data[i] = layer->w_velocity->data[i] / (1.0 - exp((current_epoch + 1) * log_beta_1)); // Bias correction for weights momentum
        }

        // Update cache 
        layer->cache_weights->data[i] = beta_2 * layer->cache_weights->data[i] + (1.0 - beta_2) * layer->dweights->data[i] * layer->dweights->data[i];
        
        // Correct cache
        if (correctBias) {
            double log_beta_2 = log(beta_2);
            layer->cache_weights->data[i] = layer->cache_weights->data[i] / (1.0 - exp((current_epoch + 1) * log_beta_2)); // Bias correction for weight cache
        }

        // Update Weights using corrected moments and cache
        layer->weights->data[i] -= (*learning_rate) * layer->w_velocity->data[i] / (sqrt(layer->cache_weights->data[i]) + epsilon);
    }
    

    // Biases
    for (int i = 0; i < layer->dbiases->dim1 * layer->dbiases->dim2; i++) {
        // Update Momentum
        layer->b_velocity->data[i] = beta_1 * layer->b_velocity->data[i] + (1.0 - beta_1) * layer->dbiases->data[i];
        
        // Correct Momentum
        if (correctBias) {
            double log_beta_1 = log(beta_1);
            layer->b_velocity->data[i] = layer->b_velocity->data[i] / (1.0 - exp((current_epoch + 1) * log_beta_1)); // Bias correction for bias momentum
        }
        
        // Update cache 
        layer->cache_bias->data[i] = beta_2 * layer->cache_bias->data[i] + (1.0 - beta_2) * layer->dbiases->data[i] * layer->dbiases->data[i];
        
        // Correct cache
        if (correctBias) {
            double log_beta_2 = log(beta_2);
            layer->cache_bias->data[i] = layer->cache_bias->data[i] / (1.0 - exp((current_epoch + 1) * log_beta_2)); // Bias correction for bias cache
        }

        // Update Bias using corrected moments and cache
        layer->biases->data[i] -= (*learning_rate) * layer->b_velocity->data[i] / (sqrt(layer->cache_bias->data[i]) + epsilon);
    }

#endif 
}

void optimization_dense(layer_dense* layer, double* lr, double lr_decay, int current_epoch, int total_epochs, double beta1, double beta2,
                        double epsilon, bool useBiasCorrection) {

    if (layer->optimization == SGD){
        update_params_sgd(layer, lr, current_epoch, lr_decay);
    }
    else if (layer->optimization == SGD_MOMENTUM) {
        update_params_sgd_momentum(layer, lr, current_epoch, lr_decay, beta1);
    }

    else if(layer->optimization == ADA_GRAD) {
        update_params_adagrad(layer, lr, current_epoch, lr_decay, epsilon);
    }

    else if(layer->optimization == RMS_PROP) {
        update_params_rmsprop(layer, lr, lr_decay, epsilon);
    }

    else if (layer->optimization == ADAM) {
        update_params_adam(layer, lr, lr_decay, beta1, beta2, epsilon, current_epoch, total_epochs, useBiasCorrection);
    }

}


