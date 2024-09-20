#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

// Create vector data structure
typedef struct {
    float x;
    float y;
} Vector2D;

// create matrix data structure
typedef struct {
    int rows;
    int cols;
    float **data;
} Matrix;

// create generic Vector structure
typedef struct {
    int length;
    float* data;
} Vector;


// Vector addition
Vector2D* vector2d_addition (Vector2D* v1, Vector2D* v2) {
    Vector2D *vec = (Vector2D*) malloc(sizeof(Vector2D));
    vec -> x = v1->x + v2->x;
    vec -> y = v1->y + v2->y;
    return vec;
}

// Vector subtraction
Vector2D* vector2d_subtraction (Vector2D* v1, Vector2D* v2) {
    Vector2D *vec = (Vector2D*) malloc(sizeof(Vector2D));
    vec -> x = v1->x - v2->x;
    vec -> y = v1->y - v2->y;
    return vec;
}

// Vector multiplication
Vector2D* vector2d_multiplication (Vector2D* v1, float scalar) {
     Vector2D *vec = (Vector2D*) malloc(sizeof(Vector2D));
     vec -> x = v1->x * scalar;
     vec -> y = v1->y * scalar;
     return vec;
}

// Vector dot product
float dot2d_product(Vector2D* v1, Vector2D* v2) {
    float sum = v1->x * v2->x + v1->y * v2->y;
    return sum;
}

// Create Matrix
Matrix* create_matrix(int num_row, int num_col) {

    Matrix *matrix = (Matrix *) malloc(sizeof(Matrix));
    if (!matrix) {
        return NULL; // Allocation failed
    }

    matrix -> rows = num_row;
    matrix -> cols = num_col;

    // Allocate Memory for Rows
    // Creates space for n row integer pointers
    matrix -> data = (float**) malloc(num_row * sizeof(float *));

    if (!matrix -> data) {
        free(matrix);
        return NULL; // Allocation failed.
    }

    // Allocate memory for each column
    for (int i=0; i < num_row; i++) {
        matrix -> data[i] = (float*) malloc(num_col * sizeof(float));
        // If allocation fails.
        if (matrix -> data[i] == NULL) {
            // Remove all previously allocated rows.
            for (int j=0; j<i; j++) {
                free(matrix->data[j]);
            }
            // Remove matrix data and matrix memory.
            free(matrix->data);
            free(matrix);
            return NULL;
        }
    }
    return matrix;
}

// create a Vector
Vector* create_vector(int size) {
    Vector *vector = (Vector *) malloc(sizeof(Vector));
    if (!vector) {
        return NULL; // Allocation failed
    }

    vector -> length = size;

    vector -> data = (float*) malloc(size * sizeof(float *));
    if (!(vector -> data)) {
        free(vector);
        return NULL; // Allocation failed
    }

    // return vector
    return vector;
}


// Free a matrix from Memory
void free_matrix(Matrix *matrix) {
    for (int i=0; i < matrix->rows; i++) {
        free(matrix -> data[i]); // Free each row in matrix
    }
    free(matrix -> data); // Free array of row pointers
    free(matrix); // Free matrix structure
}

// Add Matrices
Matrix * add_matrices(Matrix* m1, Matrix* m2) {
    if (m1 -> rows != m2 -> rows || m1 -> cols != m2 -> cols) {
        printf("Matrices m1 and m2 do not have equal dimensions. \n");
        return NULL;
    }

    Matrix* result = create_matrix(m1 -> rows, m1 -> cols);
    if (!result) {
        return NULL; // Allocation Failed
    }

    // Sum coresponding indicies.
    for (int i=0; i < result->rows; i++) {
        for (int j=0; j < result->cols; j++) {
            result->data[i][j] = m1->data[i][j] + m2->data[i][j];
        }
    }

    return result;
}

// Subtract Matrices
Matrix * subtract_matrices(Matrix* m1, Matrix* m2) {
    if (m1 -> rows != m2 -> rows || m1 -> cols != m2 -> cols) {
        printf("Matrices m1 and m2 do not have equal dimensions. \n");
        return NULL;
    }

    Matrix* result = create_matrix(m1 -> rows, m1 -> cols);
    if (!result) {
        return NULL; // Allocation Failed
    }

    // Sum coresponding indicies.
    for (int i=0; i < result->rows; i++) {
        for (int j=0; j < result->cols; j++) {
            result->data[i][j] = m1->data[i][j] - m2->data[i][j];
        }
    }

    return result;
}

// multiply matrices
Matrix* multiply_matrices(Matrix* m1, Matrix* m2) {
    if (m1 -> cols != m2 -> rows) {
        return NULL; // Dimensions not matching, cannot multiply matrices
    }

    Matrix * result = create_matrix(m1 -> rows, m2-> cols);
    if (!result) {
        return NULL; // Allocation failed
    }

    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m2->cols; j++) {
            // Initialize row col sum to be 0.
            result->data[i][j] = 0;
            for (int k=0; k < m1->cols; k++) {
                // Sum over columns of m1 and rows of m2
                result -> data[i][j] += m1->data[i][k] * m2->data[k][j];
            }
        }
    }

    return result;
}

// Display matrix function
void display_matrix(Matrix* m) {
    if (m == NULL || m->data == NULL) {
        printf("Matrix is NULL or not initialized.\n");
        return;
    }

    printf("Matrix dimensions: %d x %d\n", m->rows, m->cols);
    
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%.2f ", m->data[i][j]); 
        }
        printf("\n");
    }
}

// row reduce matrix
void row_reduce_matrix(Matrix *m) {
    float ** data = m->data;
    Vector* leading_rows = create_vector(m->rows);

    for (int i = 0; i < m->rows; i++) {
        if (fabsf(*data[i]) > 0) {
            leading_rows -> data[i] = data[i][0];
            printf("leading_rows [%d]:%f\n", i, leading_rows->data[i]);
        }
    }
    // sort leading rows
    for (int i=0; i < leading_rows->length; i++) {
        if (leading_rows -> data[i] < leading_rows -> data[i+1])
    }
    
}


// main method
int main() {
    // define matrices
    Matrix* a = create_matrix(3, 3);
    Matrix* b = create_matrix(3, 3);

    // define dim
    int rows_a = a -> rows;
    int cols_a = a -> cols;

    int rows_b = b -> rows;
    int cols_b = b -> cols;

    // define data
    float dat[3][3] = {{-1, 1, 2}, {3, 4, 5}, {6, 7, 8}};
    float dat_2[3][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};

    // fill matrix data
    for (int i = 0; i < rows_a; i++) {
        for (int j = 0; j < cols_a; j++) {
            a -> data[i][j] = dat[i][j];
            b -> data[i][j] = dat_2[i][j];
        }
    }

    display_matrix(a);
    display_matrix(b);

    row_reduce_matrix(a);

    return 0;
}



