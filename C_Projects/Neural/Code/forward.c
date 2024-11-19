#include "forward.h"

//////////////////////////////////////////////////// Forward Pass Methods //////////////////////////////////////////////////////////////

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

    // check if matrix mult worked.
    if (mult_matrix->data == NULL) {
        fprintf(stderr, "Error in matrix multiplication for forward pass.\n");
        free(output->data);
        exit(1);
    }

    // Add biases for the layer to the batch output data
    // batch_size x num_neurons, where output dim1 -> batch size
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

    // Free unused memory
    free(output->data);
    free(output);
    free(mult_matrix->data);
    free(mult_matrix);
}

void forward_reLu(matrix* batch_input) {
    // iterate through every point in the batch input
    for (int i = 0; i < batch_input->dim1 * batch_input->dim2; i++){

        // if the input value is <= 0, rectify it to 0 (otherwise, leave it unchanged)
        if(batch_input->data[i] <= 0) {
            batch_input->data[i] = 0;
        }
    }
}

void forward_softMax(matrix* batch_input) {
    // iterate over the batch
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

