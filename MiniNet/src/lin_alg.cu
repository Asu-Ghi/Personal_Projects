#include <iostream>
#include <cuda_runtime.h>

// Simple CUDA kernel to add two arrays
__global__ void add_arrays(int* A, int* B, int* C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1000;  // Number of elements in the arrays
    int size = N * sizeof(int);

    // Host arrays
    int* h_A = (int*)malloc(size);
    int* h_B = (int*)malloc(size);
    int* h_C = (int*)malloc(size);

    // Initialize arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Device arrays
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel with 1 block and N threads
    add_arrays<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            std::cerr << "Error at index " << i << ": " << h_C[i] << " != " << h_A[i] + h_B[i] << std::endl;
            return -1;
        }
    }

    std::cout << "Test passed!" << std::endl;

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
