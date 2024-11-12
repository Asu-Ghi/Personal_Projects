#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "linalg.h"
#include <time.h>
#include <float.h> 


/*
layer data structure 
holds # neurons and # inputs
holds weights and biases as pointers to 1d double arrays
*/
typedef struct {
    int num_inputs, num_neurons;
    double* weights;
    double* biases;
} layer;

/*
matrix data structure 
holds dimensional data
holds data as a pointer to a 1d double array
used to store outputs and inputs.
*/
typedef struct {
    double* data;
    int dim1, dim2;
} matrix;

// ACTIVATION FUNCTIONS // 
/*
Apply non linearity to forward passes using ReLu Activation
Steps: If a value is less than 0, rectify it to 0.
*/
void reLu(matrix* batch_input) {
    for (int i = 0; i < batch_input->dim1 * batch_input->dim2; i++){
        if(batch_input->data[i] < 0) {
            batch_input->data[i] = 0;
        }
    }
}

/*
Calculate the probabilities for classification problems using SoftMax activation.
Step 1: Subtract maximum value from each value in the input batch to ensure numerical stability (no large exponentiations)
Step 2: Calculate exponentials and sum them
Step 3: Normalize exponentials by dividing by the sum to get probabilities
*/
void softMax(matrix* batch_input) {
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

/*
Initialize a layer with number of inputs and neurons. 
Returns a layer object to use later.
*/
layer init_layer(int num_inputs, int num_neurons) {

    // initialize a layer object
    layer layer_;

    // init data for biases and weights in row major order
    layer_.weights = (double*)calloc(num_neurons * num_inputs, sizeof(double));
    layer_.biases = (double*)calloc(num_neurons, sizeof(double));
 

    if (layer_.weights == NULL || layer_.biases == NULL) {
        fprintf(stderr, "Memory Allocation Error in Init Layer.\n");
        exit(1);
    }
    

    // randomize weights
    srand(time(NULL));  // Seed the random number generator with the current time

    double min = -1, max = 1;
    //  n_inputs x n_neurons matrix
    for (int i = 0; i < num_neurons * num_inputs; i++){
        // random double between -1 and 1 scaled down by 0.05
        layer_.weights[i] = 1.5 * (min + (max - min) * ((double)rand() / RAND_MAX));
    }

    // adjust matrix objects
    layer_.num_inputs  = num_inputs; // Input dimension
    layer_.num_neurons = num_neurons; // Neuron Dimension

    // return layer object
    return(layer_);
}


/*
Performs a forward pass using a batch of inputs and a given layer and its activation function.asinhf
Returns a matrix object with output data and dimensional data for the outputs to be used in later passes.
*/
matrix forward_pass(matrix* inputs, layer* layer, char* a_func) {

    // create output struct
    matrix output;
    output.dim1 = inputs->dim1;
    output.dim2 = layer->num_neurons;
    output.data = (double*) calloc(output.dim1  * output.dim2, sizeof(double));
    if (output.data == NULL) {
        fprintf(stderr, "Error in matrix mult for forward pass.\n");
        exit(1); 
    }

    // iterate through every input row
    for (int i = 0; i < inputs->dim1; i++) {

        // matrix mult each row in inputs by weights
        double* output_data = matrix_mult(&inputs->data[i * inputs->dim2], layer->weights,
         layer->num_inputs, layer->num_neurons);

        if (output_data == NULL) {
            fprintf(stderr, "Error in matrix mult for forward pass.\n");
            exit(1);
        }

        // vector add bias to dot product of weights and input row
        for (int j = 0; j < layer->num_neurons; j++) {
            output_data[j] += layer->biases[j];
        } 

        // add output to output struct
        for (int j = 0; j < layer->num_neurons; j++) {
            output.data[i * layer->num_neurons + j] = output_data[j];
        }  

        // free temp data
        free(output_data);
    }

    // relu activation
    if (strcmp(a_func, "relu") == 0) {
        reLu(&output);
    } 

    // softmax activation
    else if(strcmp(a_func, "softmax") == 0) {
        softMax(&output);
    }

    else {
        fprintf(stderr, "Error, unsupported activation function. \n");
        exit(1);
    }

    // returns input batch after forward pass 
    return(output);
}



int main(int argc, char** argv) {
    
    // check if inputs exist
    if (argc < 2) {
        printf("Command usage %s num_inputs, num_neurons", argv[0]);
        exit(1);
    }
    // define inputs
    int num_inputs = atoi(argv[1]);
    int num_neurons = atoi(argv[2]);

    // init layers
    layer layer_1 = init_layer(3, 5);
    layer layer_2 = init_layer(5, 10);
    layer layer_3 = init_layer(10, 5);

    layer layer_4 = init_layer(5, 3);


    // Define a batch of inputs (example with 3 inputs and 4 examples in the batch)
    matrix batch;
    batch.dim1 = 4;  // Batch size
    batch.dim2 = 3;  // Number of input features (e.g., 3 inputs per example)
    batch.data = (double*)calloc(batch.dim1 * batch.dim2, sizeof(double));

    // Example input data for a batch of 4 examples, each with 3 features
    batch.data[0] = 1.0; batch.data[1] = 0.5; batch.data[2] = -0.3;  // First input
    batch.data[3] = 0.7; batch.data[4] = -0.4; batch.data[5] = 0.2;  // Second input
    batch.data[6] = -0.1; batch.data[7] = 0.3; batch.data[8] = 0.9;  // Third input
    batch.data[9] = 0.5; batch.data[10] = -0.6; batch.data[11] = 0.8; // Fourth input

    // batch after first forward pass
    matrix output_1 = forward_pass(&batch, &layer_1, "relu");
    print_matrix(output_1.data, output_1.dim1, output_1.dim2);
    printf("------------------------------------------------\n");

    matrix output_2 = forward_pass(&output_1, &layer_2, "relu");
    print_matrix(output_2.data, output_2.dim1, output_2.dim2);
    printf("------------------------------------------------\n");

    matrix output_3 = forward_pass(&output_2, &layer_3, "relu");
    print_matrix(output_3.data, output_3.dim1, output_3.dim2);
    printf("------------------------------------------------\n");

    matrix softmax_output = forward_pass(&output_3, &layer_4, "softmax");
    print_matrix(softmax_output.data, softmax_output.dim1, softmax_output.dim2);
    printf("------------------------------------------------\n");

}