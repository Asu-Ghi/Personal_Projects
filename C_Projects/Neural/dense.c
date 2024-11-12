#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "linalg.h"
#include <time.h>



typedef struct {
    double* data;
    int dim1, dim2;
} matrix;

// initialze a layer

void init_layer(int num_inputs, int num_neurons, matrix* weights, matrix* biases) {

    // row major order
    double* weight_data = (double*)calloc(num_neurons * num_inputs, sizeof(double));
    double* bias_data = (double*)calloc(num_neurons, sizeof(double));

    if (weight_data == NULL || bias_data == NULL) {
        fprintf(stderr, "Memory Allocation Error in Init Layer.\n");
        exit(1);
    }
    

    // randomize weights
    srand(time(NULL));  // Seed the random number generator with the current time

    double min = -1, max = 1;
    for (int i = 0; i < num_neurons * num_inputs; i++){
        // random double between -1 and 1 scaled down by 0.05
        weight_data[i] = 0.05 * (min + (max - min) * ((double)rand() / RAND_MAX));
    }

    // adjust matrix objects
    weights -> data = weight_data;
    weights -> dim1 = num_inputs; // Input dimension
    weights -> dim2 = num_neurons; // Neuron Dimension

    biases -> data = bias_data;
    biases -> dim1 = 1; // Biases have a single row
    biases -> dim2 = num_neurons; // One bias per neuron in layer

}

void forward_pass(matrix* inputs, matrix* weights, matrix* biases) {

    int dim1[] = {inputs -> dim1, inputs -> dim2};
    int dim2[]  = {weights -> dim1, weights -> dim2};

    // returns inputs dim1 x weights dim2 matrix
    double* output_data = matrix_mult(inputs->data, weights->data, dim1, dim2);

    // returns inputs dot weights summed with biases
    if (biases -> dim2 != weights -> dim2) {
        fprintf(stderr, "Error in dimensions for vector addition. \n");
        exit(1);    
    }
    double* output_data_w_bias = vector_sum(output_data, biases->data, biases->dim2);
    // free inputs, and allocate new memory
    free(inputs -> data);
    inputs -> data = NULL;
    // update inputs
    inputs -> data = output_data_w_bias;
    inputs -> dim1 = dim1[0];
    inputs -> dim2 = dim2[1];

    // free old memory
    free(output_data);
    free(output_data_w_bias);

    }

int main(int argc, char** argv) {
    // define 
    int num_inputs = 3;
    int num_neurons = 4;

    // init layer
    matrix weights, biases;
    init_layer(num_inputs, num_neurons, &weights, &biases);
    // define inputs and run a single forward pass.

    double* input_data = (double*) calloc(3, sizeof(double));
    input_data[0] = 1.0;
    input_data[1] = 2.0;
    input_data[2] = 3.0;

    matrix inputs = {input_data, 1, 3};

    print_matrix(inputs.data, 1, 3);

    forward_pass(&inputs, &weights, &biases);

    print_matrix(inputs.data, 1, 4);
}