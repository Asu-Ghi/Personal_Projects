#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

typedef struct {
    int* data;
    int num_rows, num_cols, stride;    
} Matrix;

Matrix* create_matrix(int rows, int cols) {
    Matrix* matrix = (Matrix*) malloc(sizeof(Matrix));
    if (matrix == NULL) {
        printf("Allocation error \n");
        return 1;
    }
    return matrix
}

int main() {
    Matrix* m = create_matrix(2, 2);
    return 0;
}