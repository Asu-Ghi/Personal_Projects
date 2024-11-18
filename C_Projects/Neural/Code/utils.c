

#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>
#define IRIS_NUM_FEATURES 4
#define IRIS_NUM_CLASSES 3

/////////////////////////////////////////////////////// Misc. Methods /////////////////////////////////////////////////////////////////

void load_iris_data(char* file_path, matrix* X_train, matrix* Y_train, matrix* X_test, matrix* Y_test, int num_batches, double train_ratio) {
    // Allocate memory for temporary X and Y
    matrix X_temp, Y_temp;
    X_temp.dim1 = num_batches;
    X_temp.dim2 = IRIS_NUM_FEATURES;
    Y_temp.dim1 = num_batches;
    Y_temp.dim2 = IRIS_NUM_CLASSES;

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
        while (token != NULL && col < IRIS_NUM_FEATURES) {
            X_temp.data[row * IRIS_NUM_FEATURES + col] = atof(token);
            token = strtok(NULL, ",");
            col++;
        }

        // Process the label (the last token)
        if (token != NULL) {
            token = strtok(token, "\n");  // Trim newline character
            // One-hot encode the label
            if (strcmp(token, "Iris-setosa") == 0) {
                Y_temp.data[row * IRIS_NUM_CLASSES] = 1.0;
            } else if (strcmp(token, "Iris-versicolor") == 0) {
                Y_temp.data[row * IRIS_NUM_CLASSES + 1] = 1.0;
            } else if (strcmp(token, "Iris-virginica") == 0) {
                Y_temp.data[row * IRIS_NUM_CLASSES + 2] = 1.0;
            }
        }

        row++;
        if (row > num_batches) {
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
        for (int k = 0; k < IRIS_NUM_FEATURES; k++) {
            double temp = X_temp.data[i * IRIS_NUM_FEATURES + k];
            X_temp.data[i * IRIS_NUM_FEATURES + k] = X_temp.data[j * IRIS_NUM_FEATURES + k];
            X_temp.data[j * IRIS_NUM_FEATURES + k] = temp;
        }

        for (int k = 0; k < IRIS_NUM_CLASSES; k++) {
            double temp = Y_temp.data[i * IRIS_NUM_CLASSES + k];
            Y_temp.data[i * IRIS_NUM_CLASSES + k] = Y_temp.data[j * IRIS_NUM_CLASSES + k];
            Y_temp.data[j * IRIS_NUM_CLASSES + k] = temp;
        }
    }

    // Calculate the split index
    int train_size = (int)(train_ratio * num_batches);
    int test_size = num_batches - train_size;

    // Allocate memory for training and testing sets
    X_train->dim1 = train_size;
    X_train->dim2 = IRIS_NUM_FEATURES;

    Y_train->dim1 = train_size;
    Y_train->dim2 = IRIS_NUM_CLASSES;
    X_test->dim1 = test_size;
    X_test->dim2 = IRIS_NUM_FEATURES;
    Y_test->dim1 = test_size;
    Y_test->dim2 = IRIS_NUM_CLASSES;

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
        for (int j = 0; j < IRIS_NUM_FEATURES; j++) {
            X_train->data[i * IRIS_NUM_FEATURES + j] = X_temp.data[i * IRIS_NUM_FEATURES + j];
        }
        for (int j = 0; j < IRIS_NUM_CLASSES; j++) {
            Y_train->data[i * IRIS_NUM_CLASSES + j] = Y_temp.data[i * IRIS_NUM_CLASSES + j];
        }
    }

    for (int i = 0; i < test_size; i++) {
        for (int j = 0; j < IRIS_NUM_FEATURES; j++) {
            X_test->data[i * IRIS_NUM_FEATURES + j] = X_temp.data[(train_size + i) * IRIS_NUM_FEATURES + j];
        }
        for (int j = 0; j < IRIS_NUM_CLASSES; j++) {
            Y_test->data[i * IRIS_NUM_CLASSES + j] = Y_temp.data[(train_size + i) * IRIS_NUM_CLASSES + j];
        }
    }

    // Free temporary arrays
    free(X_temp.data);
    free(Y_temp.data);
}

void load_data(const char* filename, double* data, int rows, int cols) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        exit(1);
    }

    // Read data from CSV and store it in the array
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fscanf(file, "%lf,", &data[i * cols + j]);  // Read each value
        }
        fscanf(file, "\n");  // Move to the next line
    }

    fclose(file);
}

void load_labels(const char *filename, int *labels, int size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        fscanf(file, "%d,", &labels[i]);
    }
    fclose(file);
}

char* optimization_type_to_string(OptimizationType type) {
    switch (type) {
        case SGD: return "SGD";
        case SGD_MOMENTUM: return "SGD_MOMENTUM";
        case ADA_GRAD: return "ADA_GRAD";
        case RMS_PROP: return "RMS_PROP";
        case ADAM: return "ADAM";
        default: return "UNKNOWN";
    }
}

char* activation_type_to_string(ActivationType type) {
    switch (type) {
        case RELU: return "RELU";
        case SOFTMAX: return "SOFTMAX";
        case SIGMOID: return "SIGMOID";
        case TANH: return "TANH";
        default: return "UNKNOWN";
    }
}


//////////////////////////////////////////////////// Linear Algebra Methods //////////////////////////////////////////////////////////////

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

layer_dense* init_layer(int num_inputs, int num_neurons, ActivationType activation, OptimizationType optimization, int batch_size) {

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
    // srand(time(NULL));  // Seed random number with current time
    srand(42);
    //  n_inputs x n_neurons matrix
    for (int i = 0; i < num_neurons * num_inputs; i++){
        // Random between -1 and 1 scaled by sqrt(1/n)
        // He initialization
        // layer_->weights->data[i] = sqrt(1.0 / num_inputs) * ((double)rand() / RAND_MAX * 2.0 - 1.0);  

        // Xavier init
        layer_->weights->data[i] = sqrt(1.0 / (num_inputs + num_neurons)) * ((double)rand() / RAND_MAX * 2.0 - 1.0);

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
    free(layer->dinputs->data);
    free(layer->dinputs);
    free(layer->pre_activation_output->data);
    free(layer->pre_activation_output);
    free(layer->post_activation_output->data);
    free(layer->post_activation_output);
    free(layer->w_velocity->data);
    free(layer->w_velocity);
    free(layer->b_velocity->data);
    free(layer->b_velocity);
    free(layer->cache_weights->data);
    free(layer->cache_weights);
    free(layer->cache_bias->data);
    free(layer->cache_bias);
    free(layer);
}

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

void update_params_sgd(layer_dense* layer, double* learning_rate, int current_epoch, double decay_rate) {

    // Decay the learning rate after each epoch
    // *learning_rate = *learning_rate / (1 + decay_rate * current_epoch);
    // fmax ensures min learning rate of 0.000001
    *learning_rate = fmax(*learning_rate * exp(-decay_rate * current_epoch), 0.000001);

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
}

void update_params_sgd_momentum(layer_dense* layer, double* learning_rate, int current_epoch, double decay_rate, double beta) {

    // Decay the learning rate after each epoch
    // *learning_rate = *learning_rate / (1 + decay_rate * current_epoch);
    // fmax ensures min learning rate of 0.000001
    *learning_rate = fmax(*learning_rate * exp(-decay_rate * current_epoch), 0.000001);
    
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
}

void update_params_adagrad(layer_dense* layer, double* learning_rate, double decay_rate, double epsilon) {
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
}

void update_params_rmsprop(layer_dense* layer, double* learning_rate, double decay_rate, double epsilon) {
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
}

void update_params_adam (layer_dense* layer, double* learning_rate, double decay_rate, double beta_1, double beta_2, double epsilon, int t) {
    
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
    if (decay_rate > 0.0) {
        *learning_rate = *learning_rate / (1.0 + decay_rate * t);
    }

    // Update momentum (first moment) with current gradients
    for (int i = 0; i < layer->weights->dim1 * layer->weights->dim2; i++) {
        layer->w_velocity->data[i] = beta_1 * layer->w_velocity->data[i] + (1.0 - beta_1) * layer->dweights->data[i];
    }

    for (int i = 0; i < layer->dbiases->dim1 * layer->dbiases->dim2; i++) {
        layer->b_velocity->data[i] = beta_1 * layer->b_velocity->data[i] + (1.0 - beta_1) * layer->dbiases->data[i];
    }

    // Correct momentum 
    for (int i = 0; i < layer->weights->dim1 * layer->weights->dim2; i++) {
        layer->w_velocity->data[i] = layer->w_velocity->data[i] / (1.0 - pow(beta_1, t + 1)); // Bias correction for momentum

        // Update cache 
        layer->cache_weights->data[i] = beta_2 * layer->cache_weights->data[i] + (1.0 - beta_2) * layer->dweights->data[i] * layer->dweights->data[i];
    }

    for (int i = 0; i < layer->biases->dim2; i++) {
        layer->b_velocity->data[i] = layer->b_velocity->data[i] / (1.0 - pow(beta_1, t + 1)); // Bias correction for bias momentum

        // Update cache 
        layer->cache_bias->data[i] = beta_2 * layer->cache_bias->data[i] + (1.0 - beta_2) * layer->dbiases->data[i] * layer->dbiases->data[i];
    }

    // Bias correction for cache
    for (int i = 0; i < layer->weights->dim1 * layer->weights->dim2; i++) {
        layer->cache_weights->data[i] = layer->cache_weights->data[i] / (1.0 - pow(beta_2, t + 1)); // Bias correction for weight cache
    }

    for (int i = 0; i < layer->biases->dim2; i++) {
        layer->cache_bias->data[i] = layer->cache_bias->data[i] / (1.0 - pow(beta_2, t + 1)); // Bias correction for bias cache
    }

    // Update weights and biases using corrected momenta and cache
    for (int i = 0; i < layer->weights->dim1 * layer->weights->dim2; i++) {
        layer->weights->data[i] -= (*learning_rate) * layer->w_velocity->data[i] / (sqrt(layer->cache_weights->data[i]) + epsilon);
    }

    for (int i = 0; i < layer->biases->dim2; i++) {
        layer->biases->data[i] -= (*learning_rate) * layer->b_velocity->data[i] / (sqrt(layer->cache_bias->data[i]) + epsilon);
    }
}
