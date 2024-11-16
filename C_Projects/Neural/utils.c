#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

//////////////////////////////////////////////////// Linear Algebra Methods //////////////////////////////////////////////////////////////
/*
Transposes a give matrix. Assumes row major order.
Includes dimensionality checks, allocates memory on the heap for the return pointer.
(row_w x col_w) => (col_w x row_w)
*/
matrix* transpose_matrix(matrix* w){

    // Check if w has data
    if (w->data == NULL) {
        fprintf(stderr, "Error: Input Matrix has no data (NULL).\n");
        exit(1);
    }

    // Create a new matrix object to hold the transposed matrix
    matrix* transposed_matrix = (matrix*) malloc(sizeof(matrix));

    // Check memory allocation for the matrix struct
    if (transposed_matrix == NULL) {
        fprintf(stderr, "Error: Memory allocation failure for transposed_matrix struct.\n");
        exit(1);
    }

    // Allocate memory for the transposed data
    transposed_matrix->dim1 = w->dim2;  // Transposed matrix rows = original matrix cols
    transposed_matrix->dim2 = w->dim1;  // Transposed matrix cols = original matrix rows
    transposed_matrix->data = (double*) calloc(transposed_matrix->dim1 * transposed_matrix->dim2, sizeof(double));

    // Check memory allocation for the transposed data
    if (transposed_matrix->data == NULL) {
        fprintf(stderr, "Error: Memory allocation failure for transposed matrix data.\n");
        exit(1);
    }

    // Iterate through the original matrix and fill the transposed matrix
    for (int i = 0; i < w->dim1; i++) {
        for (int j = 0; j < w->dim2; j++) {
            // Swap row and column indices to transpose the matrix
            transposed_matrix->data[j * w->dim1 + i] = w->data[i * w->dim2 + j];
        }
    }

    // Return the pointer to the transposed matrix
    return transposed_matrix;
}

/*
Prints a give matrix. Assumes row major order.
*/
void print_matrix(matrix* M) {
    int m = M->dim1;  // Number of rows
    int n = M->dim2;  // Number of columns

    // Loop through the rows and columns of the matrix
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", M->data[i * n + j]);  // Print element at [i, j]
        }
        printf("\n");  // New line after each row
    }
}

/*
Multiplies two matrices. Assumes row major order.
Includes dimensionality checks, allocates memory on the heap for the return pointer.
(rows_w x cols_v) = (rows_w x cols_w) * (rows_v x cols_v)
*/
matrix* matrix_mult(matrix* w, matrix* v) {
    // Get dimensionality info
    int rows_w = w->dim1;
    int cols_w = w->dim2;
    int cols_v = v->dim2;

    // Check dimensions
    if (w->dim2 != v->dim1) {
        fprintf(stderr, "Error in matrix mult, dimensionality mismatch.\n");
        exit(1);
    }

    // Allocate result matrix with dimensions rows_w x cols_v
    matrix* result = malloc(sizeof(matrix));
    result->dim1 = rows_w;
    result->dim2 = cols_v;
    result->data = (double*) calloc(rows_w * cols_v, sizeof(double));
    
    // Check memory allocation
    if (result->data == NULL) {
        fprintf(stderr, "Error: Memory allocation failure in matrix_mult.\n");
        exit(1);
    }

    // Perform the matrix multiplication
    for (int i = 0; i < rows_w; i++) { // For each row in the result
        for (int j = 0; j < cols_v; j++) { // For each column in the result
            for (int k = 0; k < cols_w; k++) { // Shared dimension
                result->data[i * cols_v + j] += w->data[i * cols_w + k] * v->data[k * cols_v + j];
            }
        }
    }

    return result;
}



//////////////////////////////////////////////////// Network Methods //////////////////////////////////////////////////////////////////

/*
Initialize a layer with number of inputs and neurons. 
Returns a pointer to a layer object to use later.
*/
layer_dense* init_layer(int num_inputs, int num_neurons, ActivationType activation, int batch_size) {

    // initialize a layer dense object
    layer_dense* layer_ = malloc(sizeof(layer_dense));
    layer_->num_inputs = num_inputs;
    layer_->num_neurons = num_neurons;
    
    // Allocate memory for layer input and dinput data
    layer_->inputs = malloc(sizeof(matrix));
    layer_->inputs->dim1 = batch_size;
    layer_->inputs->dim2 = num_inputs;
    layer_->inputs->data = (double*) calloc(layer_->inputs->dim1 * layer_->inputs->dim2, sizeof(double));
    
    // Check memory allocation
    if (layer_->inputs->data == NULL) {
        fprintf(stderr, "Error in memory allocation for inputs in forward pass.\n");
        exit(1);
    }

    // derivative of inputs
    layer_->dinputs = (matrix*) malloc(sizeof(matrix));
    layer_->dinputs->dim1 = batch_size;
    layer_->dinputs->dim2 = num_inputs;
    layer_->dinputs->data = (double*) calloc(layer_->dinputs->dim1 * layer_->dinputs->dim2, sizeof(double));

    // Check memory allocation
    if (layer_->dinputs->data == NULL) {
        fprintf(stderr, "Error in memory allocation for dinputs in forward pass.\n");
        exit(1);
    }

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

    // Allocate memory for pre activation outputs
    layer_->pre_activation_output = malloc(sizeof(matrix));
    layer_->pre_activation_output->dim1 = batch_size;
    layer_->pre_activation_output->dim2 = num_neurons;
    layer_->pre_activation_output->data = (double*) calloc(layer_->pre_activation_output->dim1*
                                                    layer_->pre_activation_output->dim2, sizeof(double));

    // Check memory allocation
    if (layer_->pre_activation_output->data == NULL) {
        fprintf(stderr, "Error in memory allocation for pre_activation_outputs in forward pass.\n");
        exit(1);
    }

    // Allocate memory for post activation outputs
    layer_->post_activation_output = malloc(sizeof(matrix));
    layer_->post_activation_output->dim1 = batch_size;
    layer_->post_activation_output->dim2 = num_neurons;
    layer_->post_activation_output->data = (double*) calloc(layer_->post_activation_output->dim1*
                                                    layer_->post_activation_output->dim2, sizeof(double));

    // Check memory allocation
    if (layer_->post_activation_output->data == NULL) {
        fprintf(stderr, "Error in memory allocation for post_activation_output in forward pass.\n");
        exit(1);
    }

    // randomize weights
    srand(42);  // Seed the random number generator with the fixed val 

    double min = -1, max = 1;
    //  n_inputs x n_neurons matrix
    for (int i = 0; i < num_neurons * num_inputs; i++){
        // random double between -1 and 1 scaled down by 0.05
        layer_->weights->data[i] = 1.5 * (min + (max - min) * ((double)rand() / RAND_MAX));

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

    // return layer dense object
    return layer_;
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
    free(layer->pre_activation_output->data);
    free(layer->pre_activation_output);
    free(layer->post_activation_output->data);
    free(layer->post_activation_output);
    free(layer);

}

/*
Calculates the accuracy of the network.
*/
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

/*
Calculates the losses for each example in the batch of the network.
*/
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

/*
SGD OPTIMIZATION
ADD INFO HERE
*/
void update_params_sgd(layer_dense* layer, double learning_rate) {
    // Update weights
    for (int i = 0; i < layer->num_neurons; i++) {
        for (int j = 0; j < layer->num_inputs; j++) {
            // W = W - learning_rate * dL/dW
            layer->weights->data[i * layer->num_inputs + j] -= learning_rate * layer->dweights->data[i * layer->num_inputs + j];
        }
    }

    // Update biases
    for(int i = 0; i < layer->num_neurons; i++) {
        // b = b - learning_rate * dL/dB
        layer->biases->data[i] -= learning_rate * layer->dbiases->data[i];
    }
}

/*
ADAGRAD OPTIMIZATION
ADD INFO HERE
*/


/*
Loads IRIS Data
*/

#define NUM_FEATURES 4
#define NUM_CLASSES 3
void load_iris_data(const char* file_path, matrix* X_train, matrix* Y_train, matrix* X_test, matrix* Y_test, int num_batches, float train_ratio) {
    // Allocate memory for temporary X and Y
    matrix X_temp, Y_temp;
    X_temp.dim1 = num_batches;
    X_temp.dim2 = NUM_FEATURES;
    Y_temp.dim1 = num_batches;
    Y_temp.dim2 = NUM_CLASSES;

    X_temp.data = (double*)calloc(X_temp.dim1 * X_temp.dim2, sizeof(double));
    Y_temp.data = (double*)calloc(Y_temp.dim1 * Y_temp.dim2, sizeof(double));

    if(X_temp.data == NULL || Y_temp.data == NULL) {
        fprintf(stderr, "Error: Memory Allocation failed in load data.\n");
        exit(1);
    }

    // Open file
    FILE* file = fopen(file_path, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file.\n");
        exit(1);
    }

    // Initialize character array for lines
    char line[1024];
    int row = 0;

    // Load data from the file
    while(fgets(line, sizeof(line), file) && row < num_batches) {
        // Tokenize the line by comma
        char* token = strtok(line, ",");
        int col = 0;

        // Process the features (first 4 tokens)
        while (token != NULL && col < NUM_FEATURES) {
            X_temp.data[row * NUM_FEATURES + col] = atof(token);
            token = strtok(NULL, ",");
            col++;
        }

        // Process the label (the last token)
        if (token != NULL) {
            token = strtok(token, "\n");  // Trim newline character
            // One-hot encode the label
            if (strcmp(token, "Iris-setosa") == 0) {
                Y_temp.data[row * NUM_CLASSES] = 1.0;
            } else if (strcmp(token, "Iris-versicolor") == 0) {
                Y_temp.data[row * NUM_CLASSES + 1] = 1.0;
            } else if (strcmp(token, "Iris-virginica") == 0) {
                Y_temp.data[row * NUM_CLASSES + 2] = 1.0;
            }
        }

        row++;
        if (row >= num_batches) {
            fprintf(stderr, "Error: Too many rows in the dataset\n");
            break;
        }
    }

    // Close the file
    fclose(file);

    // Shuffle the data to randomize the training/test split
    srand(time(NULL));
    for (int i = 0; i < num_batches; i++) {
        int j = rand() % num_batches;
        // Swap rows in X_temp and Y_temp
        for (int k = 0; k < NUM_FEATURES; k++) {
            double temp = X_temp.data[i * NUM_FEATURES + k];
            X_temp.data[i * NUM_FEATURES + k] = X_temp.data[j * NUM_FEATURES + k];
            X_temp.data[j * NUM_FEATURES + k] = temp;
        }

        for (int k = 0; k < NUM_CLASSES; k++) {
            double temp = Y_temp.data[i * NUM_CLASSES + k];
            Y_temp.data[i * NUM_CLASSES + k] = Y_temp.data[j * NUM_CLASSES + k];
            Y_temp.data[j * NUM_CLASSES + k] = temp;
        }
    }

    // Calculate the split index
    int train_size = (int)(train_ratio * num_batches);
    int test_size = num_batches - train_size;

    // Allocate memory for training and testing sets
    X_train->dim1 = train_size;
    X_train->dim2 = NUM_FEATURES;
    Y_train->dim1 = train_size;
    Y_train->dim2 = NUM_CLASSES;
    X_test->dim1 = test_size;
    X_test->dim2 = NUM_FEATURES;
    Y_test->dim1 = test_size;
    Y_test->dim2 = NUM_CLASSES;

    X_train->data = (double*)calloc(X_train->dim1 * X_train->dim2, sizeof(double));
    Y_train->data = (double*)calloc(Y_train->dim1 * Y_train->dim2, sizeof(double));
    X_test->data = (double*)calloc(X_test->dim1 * X_test->dim2, sizeof(double));
    Y_test->data = (double*)calloc(Y_test->dim1 * Y_test->dim2, sizeof(double));

    if (X_train->data == NULL || Y_train->data == NULL || X_test->data == NULL || Y_test->data == NULL) {
        fprintf(stderr, "Error: Memory Allocation failed for training or testing data.\n");
        exit(1);
    }

    // Copy data to training and testing sets
    for (int i = 0; i < train_size; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            X_train->data[i * NUM_FEATURES + j] = X_temp.data[i * NUM_FEATURES + j];
        }
        for (int j = 0; j < NUM_CLASSES; j++) {
            Y_train->data[i * NUM_CLASSES + j] = Y_temp.data[i * NUM_CLASSES + j];
        }
    }

    for (int i = 0; i < test_size; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            X_test->data[i * NUM_FEATURES + j] = X_temp.data[(train_size + i) * NUM_FEATURES + j];
        }
        for (int j = 0; j < NUM_CLASSES; j++) {
            Y_test->data[i * NUM_CLASSES + j] = Y_temp.data[(train_size + i) * NUM_CLASSES + j];
        }
    }

    // Free temporary arrays
    free(X_temp.data);
    free(Y_temp.data);
}