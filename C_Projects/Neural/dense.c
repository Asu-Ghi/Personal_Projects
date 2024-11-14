#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "linalg.h"
#include <time.h>
#include <float.h> 

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
    char* id;
    matrix* weights;
    matrix* biases;
    matrix* inputs;
    matrix* pre_activation_output;
    matrix* post_activation_output;
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
void forward_reLu(matrix* batch_input) {
    // iterate through every point in the batch input
    for (int i = 0; i < batch_input->dim1 * batch_input->dim2; i++){

        // if the input value is <= 0, rectify it to 0 (otherwise, leave it unchanged)
        if(batch_input->data[i] <= 0) {
            batch_input->data[i] = 0;
        }
    }
}

/*
Takes in gradients from layer ahead and inputs from the layer before to calculate 
gradients to send backwards to step before.
*/
void backward_reLu(matrix* inputs, matrix* input_gradients, matrix* output) {

    // check dimensions
    if(inputs->dim2 != input_gradients->dim1) {
        fprintf(stderr, "Error, mismatch in dimensions in backwards ReLu.\n");
        exit(1);
    }

    // Loop over the input_transposed matrix 
    // calculate the derivatives of each output of ReLu
    // 1 if > 0, 0 otherwise.
    for (int i = 0; i < inputs->dim1 * inputs->dim2; i++) {
        // Use the transposed input matrix to apply the ReLU derivative
        if (inputs->data[i] > 0) {
            inputs->data[i] = 1;
        } 
        else {
            inputs->data[i] = 0;
        }
    }

    // matrix multiply inputs and input_gradients
    output->data = matrix_mult(inputs->data, input_gradients->data, inputs->dim1, inputs->dim2, input_gradients->dim2);
}


/*
Calculate the probabilities for classification problems using SoftMax activation.
Step 1: Subtract maximum value from each value in the input batch to ensure numerical stability (no large exponentiations)
Step 2: Calculate exponentials and sum them
Step 3: Normalize exponentials by dividing by the sum to get probabilities
*/
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


/*
Backwards softmax and categorical loss entropy function
returns computed gradients to propogate backwards
input is the already calculated softmax outputs and true labels (one hot or sparse)
*/
void backwards_softmax_and_loss(matrix* input_gradients, matrix* inputs, matrix* true_labels, char* type_encoding) {
    // for each example in the batch
    for (int i = 0; i < inputs->dim1; i++) {

        // for each class in the output layer
        for (int j = 0; j < inputs->dim2; j++) {
            double softmax_output = inputs->data[i * inputs->dim2 + j];

            if (strcmp(type_encoding, "one_hot") == 0) {
                if (true_labels->dim1 != inputs->dim1 || true_labels->dim2 != inputs->dim2) {
                    fprintf(stderr,"Error: Mismatching dimensions in one hot encoded label.\n");
                    exit(1);
                }
                // if one-hot encoding, compute the gradient with respect to each class
                double true_label = true_labels->data[i * true_labels->dim2 + j]; // 1 for correct class, 0 otherwise
                input_gradients->data[i * inputs->dim2 + j] = softmax_output - true_label;
            }
             else if (strcmp(type_encoding, "sparse") == 0) {
                if (true_labels->dim1 != inputs->dim1) {
                    fprintf(stderr,"Error: Mismatching dimensions in one hot encoded label.\n");
                    exit(1);
                }
                // if sparse encoding, use only the correct class index
                int true_class = (int)true_labels->data[i];  // single index for true class
                if (j == true_class) {
                    input_gradients->data[i * inputs->dim2 + j] = softmax_output - 1.0;
                } 

                else {
                    input_gradients->data[i * inputs->dim2 + j] = softmax_output;
                }
            }
        }
    }
    printf("-------SOFTMAX GRADIENTS------\n");
    print_matrix(input_gradients->data, input_gradients->dim1, input_gradients->dim2);
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
Performs a forward pass using a batch of inputs and a given layer and its activation function
Returns a matrix object with output data and dimensional data for the outputs to be used in later passes.
*/
matrix forward_pass(matrix* inputs, layer_dense* layer) {

    // create output struct
    matrix* output = malloc(sizeof(matrix));
    output->dim1 = inputs->dim1; // number of vectors in batch
    output->dim2 = layer->weights->dim2; // number of neurons in weights
    output->data = (double*) calloc(output->dim1  * output->dim2, sizeof(double));
    if (output->data == NULL) {
        fprintf(stderr, "Error in matrix mult for forward pass.\n");
        exit(1); 
    }
    // num_inputs x num_neurons
    // Perform matrix multiplication between inputs (dim1 x dim2) and weights (dim2 x dim3)
    // returns batch_size x num_neuron matrix for the layer
    double* multiplied_data = matrix_mult(inputs->data, layer->weights->data,
                                          inputs->dim1, inputs->dim2, layer->weights->dim2);

    // check if matrix mult worked.
    if (multiplied_data == NULL) {
        fprintf(stderr, "Error in matrix multiplication for forward pass.\n");
        free(output->data);
        exit(1);
    }

    // Add biases for the layer to the batch output data
    // batch_size x num_neurons, where output dim1 -> batch size
    for (int i = 0; i < output->dim1; i++) {
        // output dim2-> num neurons
        for (int j = 0; j < output->dim2; j++) {
            output->data[i * output->dim2 + j] = multiplied_data[i * output->dim2 + j] + layer->biases->data[j];
        }
    }

    // Allocate memory for pre activation outputs
    layer->pre_activation_output = malloc(sizeof(matrix));
    layer->pre_activation_output->dim1 = output->dim1;
    layer->pre_activation_output->dim2 = output->dim2;
    layer->pre_activation_output->data = (double*) calloc(output->dim1  * output->dim2, sizeof(double));
    // Update pre activation outputs
    memcpy(layer->pre_activation_output->data,  output->data, output->dim1 * output->dim2 * sizeof(double));

    // modifies output 
    // relu activation
    if (layer->activation == RELU) {
        forward_reLu(output);
    } 

    // softmax activation
    else if(layer->activation == SOFTMAX) {
        forward_softMax(output);
    }

    else {
        fprintf(stderr, "Error, unsupported activation function. \n");
        exit(1);
    }

    // Allocate memory for pre activation outputs
    layer->post_activation_output = malloc(sizeof(matrix));
    layer->post_activation_output->dim1 = output->dim1;
    layer->post_activation_output->dim2 = output->dim2;
    layer->post_activation_output->data = (double*) calloc(output->dim1  * output->dim2, sizeof(double));

    // Update post activation outputs
    memcpy(layer->post_activation_output->data,  output->data, output->dim1 * output->dim2 * sizeof(double));
    return *output;
}


/*
Performs a backward pass of the data for each dense layer provided
Takes in an input of gradients from layer ahead, and inputs (outputs from layer before)
Returns a matrix of gradients to be used in the layer before, and later for optimization.
Stores this matrix of gradients in the layer dense data structure to be used later
*/
matrix backward_pass(matrix* input_gradients, layer_dense* layer, matrix* inputs) {

    // Print debug info
    printf("---------LAYER %s---------\n", layer->id);
    printf("----Inputs---\n");
    print_matrix(inputs->data, inputs->dim1, inputs->dim2);
    printf("----Input Gradients---\n");
    print_matrix(input_gradients->data, input_gradients->dim1, input_gradients->dim2);


    // Check if activation function is softmax
    if (layer->activation == SOFTMAX) {
        // Softmax gradient already computed separately
        // We just pass the computed input_gradients to the next layer
        return *input_gradients;
    }

    // Initialize gradients for weights, biases, and inputs
    matrix dweights, dbiases, dinputs;

    // Create a transposed object of inputs.
    matrix* inputs_transposed = transpose_matrix(inputs);

    // Apply activation function gradient (backprop through ReLU or others)

   // Initialize output matrix for activation function backwards pass.
    matrix activation_outputs;
    activation_outputs.data = (double*) calloc(inputs_transposed->dim1 * input_gradients->dim2, sizeof(double));
    activation_outputs.dim1 = inputs_transposed->dim1;
    activation_outputs.dim2 = input_gradients->dim2;
    if (activation_outputs.data == NULL) {
        fprintf(stderr, "Error: Memory allocation failure in backwards ReLu function. \n");
        exit(1);
    }

    if (layer->activation == RELU) {
        // returns batch_size x neurons matrix
         backward_reLu(inputs_transposed, input_gradients, &activation_outputs);  // ReLU backward pass
        
        // debugging
        printf("--------Derivative ReLU------------\n");
        print_matrix(activation_outputs.data, activation_outputs.dim1, activation_outputs.dim2);
    }

    // Continue with weight and bias gradient calculations...

    // Example for weight gradient computation (dW = X^T * delta):
    dweights.dim1 = inputs_transposed->dim1;  // Number of input features
    dweights.dim2 = activation_outputs.dim2;  // Number of neurons in the current layer
    
    printf("%d x %d\n", inputs_transposed->dim1, inputs_transposed->dim2);
    printf("%d x %d\n", activation_outputs.dim1, activation_outputs.dim2);

    dweights.data = matrix_mult(inputs->data, activation_outputs.data, 
                                 inputs->dim1, inputs->dim2, activation_outputs.dim2);  
                                       
    printf("--------Derivative Weights------------\n");
    print_matrix(dweights.data, dweights.dim1, dweights.dim2);



    // Sum gradients for biases (db = sum(delta)):
    dbiases.dim1 = 1;  // 1 row, sum of biases across all examples
    dbiases.dim2 = activation_outputs.dim2;  // Number of neurons
    dbiases.data = (double*)calloc(dbiases.dim2, sizeof(double));
    
    for (int i = 0; i < activation_outputs.dim2; i++) {  // Loop over the neurons
        for (int j = 0; j < input_gradients->dim1; j++) {  // Loop over the examples
            dbiases.data[i] += input_gradients->data[j * input_gradients->dim2 + i];
        }
    }

    // Gradient for inputs (dX = delta * W^T):
    matrix* weights_transposed = transpose_matrix(layer->weights);
    dinputs.data = matrix_mult(input_gradients->data, weights_transposed->data, 
                               input_gradients->dim1, input_gradients->dim2, weights_transposed->dim2);
    dinputs.dim1 = input_gradients->dim1;  // Number of examples
    dinputs.dim2 = layer->weights->dim1;  // Number of features in the previous layer

    // Store gradients in layer for later use:
    layer->dweights = (matrix*)malloc(sizeof(matrix));
    layer->dbiases = (matrix*)malloc(sizeof(matrix));
    layer->dinputs = (matrix*)malloc(sizeof(matrix));

    *layer->dweights = dweights;
    *layer->dbiases = dbiases;
    *layer->dinputs = dinputs;

    // Return the gradient for the inputs to the previous layer
    free(inputs_transposed->data);
    free(inputs_transposed);
    return dinputs;
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
Create input gradients to be used in first step of backprop.
*/
matrix create_input_gradients(matrix* softmax_outputs) {
    matrix input_gradients;
    // Allocate input_gradients to match dimensions of inputs (softmax outputs)
    input_gradients.dim1 = softmax_outputs->dim1;
    input_gradients.dim2 = softmax_outputs->dim2;
    input_gradients.data = (double*)calloc(input_gradients.dim1 * input_gradients.dim2, sizeof(double));

    if (input_gradients.data == NULL) {
        fprintf(stderr, "Error: Memory allocation failure for input_gradients.\n");
        exit(1);
    }
    return input_gradients;
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
    layer_1.id = "1";
    layer_dense layer_2 = init_layer(5, 10, RELU);
    layer_2.id = "2";
    layer_dense layer_3 = init_layer(10, 5, RELU);
    layer_3.id = "3";
    layer_dense layer_4 = init_layer(5, 3, SOFTMAX);
    layer_4.id = "4";



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
    // matrix losses_one_hot = loss_categorical_cross_entropy(&softmax_output, &one_hot_vector, "one_hot");
    // matrix losses_sparse = loss_categorical_cross_entropy(&softmax_output, &sparse_vector, "sparse");

    // // calculate  and print accuracy of the batch
    // double accuracy_one_hot = calculate_accuracy(&softmax_output, &one_hot_vector, "one_hot");
    // double accuracy_sparse = calculate_accuracy(&softmax_output, &sparse_vector, "sparse");

    /*
    
    Insert Backprop work here
    
    */

    // Step 1: Initialize `input_gradients` for the last layer (softmax layer)
    // Assuming 'softmax_output' is the output of the softmax layer and the loss function is applied
    matrix input_gradients;
    input_gradients.dim1 = softmax_output.dim1;
    input_gradients.dim2 = softmax_output.dim2;
    input_gradients.data = (double*)calloc(input_gradients.dim1 * input_gradients.dim2, sizeof(double));

    if (input_gradients.data == NULL) {
        fprintf(stderr, "Error: Memory allocation failure for input_gradients.\n");
        exit(1);
    }

    // Step 2: Compute gradients for the softmax/loss layer
    backwards_softmax_and_loss(&input_gradients, &softmax_output, &one_hot_vector, "one_hot");
    printf("Arrive1\n");

    // Step 3: Compute gradients for the softmax layer (already done with backwards_softmax_and_loss)

    matrix grad_layer3 = backward_pass(&input_gradients, &layer_4, &output_3);  // Backpropagate from softmax to layer 3
    printf("grad_layer3 dim: %d x %d\n", grad_layer3.dim1, grad_layer3.dim2);

    matrix grad_layer2 = backward_pass(&grad_layer3, &layer_3, &output_2);  // Backpropagate from layer 3 to layer 2
    printf("grad_layer2 dim: %d x %d\n", grad_layer2.dim1, grad_layer2.dim2);

    matrix grad_layer1 = backward_pass(&grad_layer2, &layer_2, &output_1);  // Backpropagate from layer 2 to layer 1
    printf("grad_layer1 dim: %d x %d\n", grad_layer1.dim1, grad_layer1.dim2);

    matrix grad_input = backward_pass(&grad_layer1, &layer_1, &batch);  // Backpropagate from layer 1 to the input
    printf("grad_input dim: %d x %d\n", grad_input.dim1, grad_input.dim2);

 

    // Print layer gradients to verify
    printf("Layer 1 Weight Gradients:\n");
    print_matrix(layer_1.dweights->data, layer_1.dweights->dim1, layer_1.dweights->dim2);
    
    printf("Layer 2 Weight Gradients:\n");
    print_matrix(layer_2.dweights->data, layer_2.dweights->dim1, layer_2.dweights->dim2);
    
    printf("Layer 3 Weight Gradients:\n");
    print_matrix(layer_3.dweights->data, layer_3.dweights->dim1, layer_3.dweights->dim2);
    
    printf("Layer 4 Weight Gradients:\n");
    print_matrix(layer_4.dweights->data, layer_4.dweights->dim1, layer_4.dweights->dim2);


    // free memory
    free(batch.data);
    free(output_1.data);
    free(output_2.data);
    free(output_3.data);
    free(softmax_output.data);
    // free(losses_one_hot.data);
    // free(losses_sparse.data);
    free(input_gradients.data);
    free(grad_layer3.data);
    free(grad_layer2.data);
    free(grad_layer1.data);
    free(grad_input.data);

    // free layers last
    free_layer(&layer_1);
    free_layer(&layer_2);
    free_layer(&layer_3);
    free_layer(&layer_4);
}