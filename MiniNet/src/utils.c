#include "utils.h"

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

void load_data(const char* filename, double* data, int start_row, int end_row, int cols) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        exit(1);
    }

    // Temporary buffer to read lines from the CSV file
    char line[1024];  // Adjust size depending on expected row size

    // Skip rows before start_row
    for (int i = 0; i < start_row; i++) {
        if (!fgets(line, sizeof(line), file)) {
            fprintf(stderr, "Error: Reached end of file before start_row\n");
            exit(1);
        }
        // Discard the entire line since we are skipping the row
    }

    // Read data from file into the data array
    for (int i = start_row; i < end_row; i++) {
        if (!fgets(line, sizeof(line), file)) {
            fprintf(stderr, "Error: Reached end of file unexpectedly\n");
            exit(1);
        }

        // Tokenize the row to read each column
        char* token = strtok(line, ",");
        for (int j = 0; j < cols && token != NULL; j++) {
            data[i * cols + j] = atof(token);  // Convert token to double and store in the array
            token = strtok(NULL, ",");  // Get next token
        }
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

    // Print dim
    printf("(%d x %d)\n", m, n);    
    // Loop through the rows and columns of the matrix
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", M->data[i * n + j]);  // Print element at [i, j]
        }
        printf("\n");  // New line after each row
    }
}

// Includes ifdef for parallelization
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

#ifdef ENABLE_PARALLEL 
    // double start_time = omp_get_wtime();
    // Perform the matrix multiplication
    #pragma omp parallel 
    {   
        int thread_id = omp_get_thread_num(); // Get current thread id
        int total_threads = omp_get_num_threads(); // Get total num threads
        int rows_per_thread = (rows_w + total_threads - 1) / total_threads; // Get num rows to calc per each thread
        int start_row = rows_per_thread * thread_id; // Get start row for unique thread
        int end_row = rows_per_thread * thread_id + rows_per_thread; // Get end row for unique thread

        // Check to see if in bounds of thread calculations
        if (end_row > rows_w) {
            end_row = rows_w;
        }

        // Calculate matrix mult for each row
        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < cols_v; j++) {
                for (int k = 0; k < cols_w; k++) {
                    result->data[i * cols_v + j] += w->data[i * cols_w + k] * v->data[k * cols_v + j];
                }
            }
        }
    }
    // double end_time = omp_get_wtime();
    // printf("Matrix multiplication completed in %.6f seconds.\n", end_time - start_time);


#else 
        for (int i = 0; i < rows_w; i++) {
            for (int j = 0; j < cols_v; j++) {
                for (int k = 0; k < cols_w; k++) {
                    result->data[i * cols_v + j] += w->data[i * cols_w + k] * v->data[k * cols_v + j];
                }
            }
        }


#endif

    return result;

}

// Includes ifdef for parallelization
matrix* element_matrix_mult(matrix* w, matrix* v){
    // Check dimensions
    if(w->dim1 != v->dim1 || w->dim2 != v->dim2) {
        fprintf(stderr, "Error, mismatching dimensions in element matrix mult.\n");
    }
    int row_w = w->dim1;
    int col_w = w->dim2;

    // Allocate and check memory for result
    matrix* result = malloc(sizeof(matrix));
    result->dim1 = row_w;
    result->dim2 = col_w;
    result->data = (double*) calloc(row_w * col_w, sizeof(double));

    if (result->data == NULL) {
        fprintf(stderr, "Memory allocation failure for result in element matrix mult.\n");
        exit(1);
    }



#ifdef ENABLE_PARALLEL 
    // Parallel Code
    #pragma omp parallel 
    {
        int thread_id = omp_get_thread_num(); // Get current thread id
        int total_threads = omp_get_num_threads(); // Get total num threads
        int rows_per_thread = (row_w + total_threads - 1) / total_threads; // Get num rows to calc per each thread
        int start_row = rows_per_thread * thread_id; // Get start row for unique thread
        int end_row = rows_per_thread * thread_id + rows_per_thread; // Get end row for unique thread

        if (end_row > row_w) {
            end_row = row_w;
        }

        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < col_w; j++) {
                result->data[i * col_w + j] = w->data[i * col_w + j] * v->data[i * col_w + j];
            }
        }
    }

#else

    // Sequential Code
    for (int i = 0; i < row_w; i++) {
        for (int j = 0; j < col_w; j++) {
            result->data[i * col_w + j] = w->data[i * col_w + j] * v->data[i * col_w + j];
        }
    }

#endif
    return result;
}

// Includes ifdef for parallelization
double vector_dot_product(matrix* w, matrix* v) {

    // Check dimensions
    if (w->dim1 != v->dim1 || w->dim2 != v->dim2) {
        fprintf(stderr, "Error, Dimensionality mismatch in vector dot product.\n");
        exit(1);
    }

    double sum = 0.0;
    int dim = w->dim1 * w->dim2;

#ifdef ENABLE_PARALLEL
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num(); // Get current thread id
        int total_threads = omp_get_num_threads(); // Get total num threads
        int rows_per_thread = (dim + total_threads - 1) / total_threads; // Get num rows to calc per each thread
        int start_row = rows_per_thread * thread_id; // Get start row for unique thread
        int end_row = rows_per_thread * thread_id + rows_per_thread; // Get end row for unique thread
        double thread_sum = 0.0;

        // check bounds
        if (end_row > dim) {
            end_row = dim;
        }
        for (int i = start_row; i < end_row; i++) {
            thread_sum += w->data[i] * v->data[i];
        }

        // update sum
        #pragma omp atomic
        sum += thread_sum;
    }

#else
    for (int i = 0; i < dim; i++) {
        sum += w->data[i] * v->data[i];
    }

#endif
    return sum;
}

// Includes ifdef for parallelization
void matrix_scalar_mult(matrix* w, double s) {

    int rows = w->dim1;
    int cols = w->dim2;
    
#ifdef ENABLE_PARALLEL
#pragma omp parallel
{
    int thread_id = omp_get_thread_num(); // Get current thread id
    int total_threads = omp_get_num_threads(); // Get total num threads
    int rows_per_thread = (rows + total_threads - 1) / total_threads; // Get num rows to calc per each thread
    int start_row = rows_per_thread * thread_id; // Get start row for unique thread
    int end_row = rows_per_thread * thread_id + rows_per_thread; // Get end row for unique thread
    
    // check bounds
    if(end_row > rows) {
        end_row = rows;
    }

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < cols; j++) {
            w->data[i * cols + j] = s * w->data[i * cols + j];
        }
    }

}

#else
    for (int i = 0; i < rows * cols; i++) {
        w->data[i] = s * w->data[i];
    }

#endif

}



