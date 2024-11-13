#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "linalg.h"
#include <time.h>
#include <float.h> 

/*
matrix data structure 
holds dimensional data
holds data as a pointer to a 1d double array
used to store outputs and inputs.
*/
typedef struct {
    double* data;
    int dim1;
    int dim2;
} matrix;

/*
Activation function enum structure
Enum to store what activation function is being used
*/
typedef enum {
    RELU,
    SOFTMAX
} ActivationType;


/*
layer_dense data structure 
Has matrix data structures that hold layer info
Includes information for backpropogation
*/
typedef struct {
    matrix* weights;
    matrix* biases;
    matrix* inputs;
    matrix* output;
    matrix* dweights;
    matrix* dbiases;
    matrix* dinputs;
    ActivationType activation;
} layer_dense;


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
layer_dense init_layer(int num_inputs, int num_neurons, ActivationType activation) {

    // initialize a layer dense object
    layer_dense layer_;

    // init data for weights and biases in row major order
    layer_.weights = (matrix*) malloc(sizeof(matrix));
    layer_.weights->data = (double*) calloc(num_inputs * num_neurons, sizeof(double));
    layer_.weights->dim1 = num_inputs; 
    layer_.weights->dim2 = num_neurons; 

    layer_.biases = (matrix*) malloc(sizeof(matrix));
    layer_.biases->data = (double*)calloc(num_neurons, sizeof(double));
    layer_.biases->dim1 = 1;
    layer_.biases->dim2 = num_neurons;

    // check if memory for matrix object was allocated.
    if (layer_.weights == NULL || layer_.biases == NULL) {
        fprintf(stderr, "Memory Allocation Error in Init Layer.\n");
        exit(1);
    }

    // check if memory for data in matrix objects were allocated.
    if (layer_.weights->data == NULL || layer_.biases->data == NULL) {
        fprintf(stderr, "Memory Allocation Error in Init Layer.\n");
        exit(1);
    }
    

    // randomize weights
    srand(42);  // Seed the random number generator with the fixed val 

    double min = -1, max = 1;
    //  n_inputs x n_neurons matrix
    for (int i = 0; i < num_neurons * num_inputs; i++){
        // random double between -1 and 1 scaled down by 0.05
        layer_.weights->data[i] = 1.5 * (min + (max - min) * ((double)rand() / RAND_MAX));
    }

    // initialize other matrix objects

    // derivative of weights
    layer_.dweights = (matrix*) malloc(sizeof(matrix));
    layer_.dweights->data = (double*) calloc(num_inputs * num_neurons, sizeof(double));
    layer_.dweights->dim1 = num_inputs;
    layer_.dweights->dim2 = num_neurons;

    // derivative of biases
    layer_.dbiases = (matrix*) malloc(sizeof(matrix));
    layer_.dbiases->data = (double*) calloc(num_neurons, sizeof(double));
    layer_.dbiases->dim1 = 1;
    layer_.dbiases->dim2 = num_neurons;

    // derivative of inputs
    layer_.dinputs = (matrix*) malloc(sizeof(matrix));
    layer_.dinputs->data = (double*) calloc(num_inputs, sizeof(double));
    layer_.dinputs->dim1 = 1;
    layer_.dinputs->dim2 = num_inputs;

    // initialize activation function for the layer
    layer_.activation = activation;

    // return layer dense object
    return layer_;
}

// Loss Functions
/*
Categorical Cross Entropy Loss Function
Takes in either one hot encoded vectors as y_pred (true values), and a set of softmax outputs (predictions)
The higher the confidence output in softmax, the lower the loss value -> log(1) < log(.01)
Returns loss related to the predictions, determines deviation from the true values.
Sparse vector is size inputs in the batch (dim1) and contains the index of the true class per input in batch.asinhf
One hot encoded vector is size inputs in the batch x size of the input vectors ie dim1 x dim2
*/
matrix loss_categorical_cross_entropy(matrix* pred, matrix* true, char* type_encoding) {

    // check if predictions and true values dim1 match in size
    if(pred->dim1 != true->dim1) {
        fprintf(stderr, "Mismatch in prediction batch size and true value size. \n");
        exit(1);
    }

    // initialize losses data.
    matrix losses;
    losses.data = (double*) calloc(pred->dim1, sizeof(double));
    losses.dim1 = pred->dim1;
    losses.dim2 = 1;

    // one hot encoded assumption
    if(strcmp(type_encoding, "one_hot") == 0) {
        
        // check if one hot is the correct size
        if (pred->dim2 != true->dim2) {
            fprintf(stderr, "Error: Dimension 2 for one hot vectors and predictions do not match.\n");
            free(losses.data);
            exit(1);
        }

        // iterate over every vector in the prediction batch
        for (int i = 0; i < pred->dim1; i++) {

            // find true class in one hot vector
            int true_class = -1;
            for (int j = 0; j < true->dim2; j++) {
                if (true->data[i * true->dim2 + j] == 1.0) {
                    true_class = j;
                    break;
                }
            }

            // error handling if no true class is found
            if(true_class == -1) {
                fprintf(stderr, "Error: No true class found in one hot vectors. \n");
                free(losses.data);
                exit(1);
            }

            // get predicted sample in question with relation to true class
            double predicted_sample = pred->data[i * pred->dim2 + true_class];

            // clip value so we never calculate log(0)
            if(predicted_sample < 1e-15) {
                predicted_sample = 1e-15;
            }
            
            // calcuale -log loss for the sample in question and append to loss matrix
            double loss = -log(predicted_sample);
            losses.data[i] = loss;
        }
    }

    // sparse encoded assumption
    else if (strcmp(type_encoding, "sparse") == 0) {

        // iterate through every true classification in the sparse vector
        for (int i = 0; i < true->dim1; i++) {

            // get true class value from 1d sparse vector.
            int true_class = (int)true->data[i];

            // Error handling, check if true class is in bounds of prediction vectors
            if (true_class < 0 || true_class >= pred->dim2) {
                fprintf(stderr,"Error: True class dimensions out of bounds. \n");
                free(losses.data);
                exit(1);
            }  

            // get predicted sample from batch data 
            double predicted_sample = pred->data[i * pred->dim2 + true_class];
            
            // clip value so we never calculate log(0)
            if (predicted_sample < 1e-15) {
                predicted_sample = 1e-15;
            }

            // calcuale -log loss for the sample in question and append to loss matrix
            double loss = -log(predicted_sample);
            
            losses.data[i] = loss;
        }
    }

    // error handling
    else {
        fprintf(stderr, "Error: Incorrect type encoding provided. \n");
        free(losses.data);
        exit(1);
    }

    // return losses
    return(losses);
}

/*
Returns the accuracy of the model based on comparing softmax outputs to true values. Works similar to 
categorical cross entropy loss, but returns the average of how many times each prediction is correct.
*/
double calculate_accuracy(matrix* pred, matrix* class_targets, char* type_encoding) {

    // handles mismatching first dimensions 
    if (class_targets->dim1 != pred->dim1) {
        fprintf(stderr, "Error: Mismatching dimensions in calculate accuracy, dim1 for class targets and predictions.\n");
        exit(1);
    } 

    // stores correct prediction count
    int correct_count = 0;

    // stores number of samples
    int num_samples = pred->dim1;

    // handles one hot encoded vectors
    if (strcmp(type_encoding, "one_hot") == 0) {

         // handles mismatching second dimensions 
        if (class_targets->dim2 != pred->dim2) {
            fprintf(stderr, "Error: Mismatching dimensions in calculate accuracy, dim2 for class targets and predictions.\n");
            exit(1);
        } 

        // iter through every prediction
        for (int i = 0; i < pred->dim1; i++) {

            // find max value, ie the prediction in each input in the batch
            int max_indx = -1;
            double max = -DBL_MAX;
            for (int j = 0; j < pred->dim2; j++) {
                if (pred->data[i * pred->dim2 + j] > max) {
                    max = pred->data[i * pred->dim2 + j];
                    max_indx = j;
                }
            }

            // incriment correct count if the predictions match the one hot encoded vector
            if (class_targets->data[i * class_targets->dim2 + max_indx] == 1) {
                correct_count += 1;
            }
        }    
    }

    else if (strcmp(type_encoding, "sparse") == 0) {

        // iter through every prediction
        for (int i = 0; i < pred->dim1; i++) {
            int max_indx = -1;
            double max = -DBL_MAX;
            for (int j = 0; j < pred->dim2; j++) {
                if (pred->data[i * pred->dim2 + j] > max) {
                    max = pred->data[i * pred->dim2 + j];
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

/*
Performs a forward pass using a batch of inputs and a given layer and its activation function.asinhf
Returns a matrix object with output data and dimensional data for the outputs to be used in later passes.
*/
matrix forward_pass(matrix* inputs, layer_dense* layer) {

    // create output struct
    matrix output;
    output.dim1 = inputs->dim1; // number of vectors in batch
    output.dim2 = layer->weights->dim2; // number of neurons in weights
    output.data = (double*) calloc(output.dim1  * output.dim2, sizeof(double));
    if (output.data == NULL) {
        fprintf(stderr, "Error in matrix mult for forward pass.\n");
        exit(1); 
    }

    // iterate through every input row
    for (int i = 0; i < inputs->dim1; i++) {

        // matrix mult each row in inputs by weights
        double* output_data = matrix_mult(&inputs->data[i * inputs->dim2], layer->weights->data,
         layer->weights->dim1, layer->weights->dim2);

        if (output_data == NULL) {
            fprintf(stderr, "Error in matrix mult for forward pass.\n");
            exit(1);
        }

        // vector add bias to dot product of weights and input row
        for (int j = 0; j < layer->weights->dim2; j++) {
            output_data[j] += layer->biases->data[j];
        } 

        // add output to output struct
        for (int j = 0; j < layer->weights->dim2; j++) {
            output.data[i * layer->weights->dim2 + j] = output_data[j];
        }  

        // free temp data
        free(output_data);
    }

    // relu activation
    if (layer->activation == RELU) {
        reLu(&output);
    } 

    // softmax activation
    else if(layer->activation == SOFTMAX) {
        softMax(&output);
    }

    else {
        fprintf(stderr, "Error, unsupported activation function. \n");
        exit(1);
    }

    // returns input batch after forward pass 
    return(output);
}

/*
Used to free memory for each dense layer
*/
void free_layer(layer_dense* layer) {
    free(layer->weights->data);
    free(layer->weights);
    free(layer->biases->data);
    free(layer->biases);
    free(layer->dweights->data);
    free(layer->dweights);
    free(layer->dbiases->data);
    free(layer->dbiases);
    free(layer->dinputs->data);
    free(layer->dinputs);
}


/*
Main Method
*/
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
    layer_dense layer_1 = init_layer(3, 5,  RELU);
    layer_dense layer_2 = init_layer(5, 10, RELU);
    layer_dense layer_3 = init_layer(10, 5, RELU);
    layer_dense layer_4 = init_layer(5, 3, SOFTMAX);


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
    matrix output_1 = forward_pass(&batch, &layer_1);
    print_matrix(output_1.data, output_1.dim1, output_1.dim2);
    printf("------------------------------------------------\n");

    matrix output_2 = forward_pass(&output_1, &layer_2);
    print_matrix(output_2.data, output_2.dim1, output_2.dim2);
    printf("------------------------------------------------\n");

    matrix output_3 = forward_pass(&output_2, &layer_3);
    print_matrix(output_3.data, output_3.dim1, output_3.dim2);
    printf("------------------------------------------------\n");

    matrix softmax_output = forward_pass(&output_3, &layer_4);
    print_matrix(softmax_output.data, softmax_output.dim1, softmax_output.dim2);
    printf("------------------------------------------------\n");

    // create true label vectors

    // size 4 x 3 -> matches input batch  
    matrix one_hot_vector;
    double data1[] = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0};
    one_hot_vector.data = data1;
    one_hot_vector.dim1 = 4;
    one_hot_vector.dim2 = 3;

    // size 4 x 1 -> first dim matches input batch
    matrix sparse_vector;
    double data2[] = {0.0, 2.0, 1.0, 1.0};
    sparse_vector.data = data2;
    sparse_vector.dim1 = 4;
    sparse_vector.dim2 = 1;

    // calculate loss from softmax
    matrix losses_one_hot = loss_categorical_cross_entropy(&softmax_output, &one_hot_vector, "one_hot");
    matrix losses_sparse = loss_categorical_cross_entropy(&softmax_output, &sparse_vector, "sparse");

    // calculate  and print accuracy of the batch
    double accuracy_one_hot = calculate_accuracy(&softmax_output, &one_hot_vector, "one_hot");
    double accuracy_sparse = calculate_accuracy(&softmax_output, &sparse_vector, "sparse");

    printf("--------------------------------------\n");
    printf("Accuracy using one hot encoded method: %f\n", accuracy_one_hot);
    printf("Accuracy using sparse encoded method: %f\n", accuracy_sparse);
    printf("--------------------------------------\n");

    // print losses
    printf("------ One_Hot Losses --------\n");
    print_matrix(losses_one_hot.data, losses_one_hot.dim1, losses_one_hot.dim2);

    printf("------ Sparse Losses --------\n");
    print_matrix(losses_sparse.data, losses_sparse.dim1, losses_sparse.dim2);
    
    // free memory
    free(batch.data);
    free(output_1.data);
    free(output_2.data);
    free(output_3.data);
    free(softmax_output.data);
    free(losses_one_hot.data);
    free(losses_sparse.data);

    // free layers last
    free_layer(&layer_1);
    free_layer(&layer_2);
    free_layer(&layer_3);
    free_layer(&layer_4);
}