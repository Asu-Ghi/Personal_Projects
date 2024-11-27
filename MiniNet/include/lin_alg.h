#ifndef LIN_ALG_H
#define LIN_ALG_H

#include <stdio.h>
#include <cuda_runtime.h>

/*
Header file for Cuda implemented linear algebra methods.
All CUDA methods refrenced and initiated from utils.c
*/

__global__ void matrix_mult_kernel(double* d_w, double* d_v, double* d_result, int rows_w, int cols_w, int cols_v);


#endif