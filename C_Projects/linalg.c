#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

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

// Vector addition
Vector2D* vector_addition (Vector2D* v1, Vector2D* v2) {
    Vector2D *vec = (Vector2D*) malloc(sizeof(Vector2D));
    vec -> x = v1->x + v2->x;
    vec -> y = v1->y + v2->y;
    return vec;
}

// Vector subtraction
Vector2D* vector_subtraction (Vector2D* v1, Vector2D* v2) {
    Vector2D *vec = (Vector2D*) malloc(sizeof(Vector2D));
    vec -> x = v1->x - v2->x;
    vec -> y = v1->y - v2->y;
    return vec;
}

// Vector multiplication
Vector2D* vector_multiplication (Vector2D* v1, float scalar) {
     Vector2D *vec = (Vector2D*) malloc(sizeof(Vector2D));
     vec -> x = v1->x * scalar;
     vec -> y = v1->y * scalar;
     return vec;
}

// Vector dot product
float dot_product(Vector2D* v1, Vector2D* v2) {
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
            printf("%f ", m->data[i][j]); 
        }
        printf("\n");
    }
}


// main method
int main() {
    Matrix* a = create_matrix(4, 3);
    Matrix* b = create_matrix(3, 4);

    display_matrix(a);

    return 0;
}



