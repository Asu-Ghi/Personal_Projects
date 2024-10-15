int main() {

}

// row major order
void matrix_multiply( double * A , double * B , double * C , int m , int p , int n ) {
    // Returning a m x n matrix -> iter over m and then n
    C = (double*) calloc(m*n, sizeof(double));
    // Rows A
    for (int i = 0; i < m; i++) {
        // Cols B
        for (int j = 0; j < n; j++) {
            // Row B or Col A
            for (int k = 0; k < p; k++) {
                C[i * n + j] += A[i * n + k] * B[k * p + j]
            }
        } 
    }

}