#ifndef TEST_FUNCTIONS_H
#define TEST_FUNCTIONS_H

#include "network.h"

///////////////////////////////////////////////////LINEAR ALGEBRA FUNCTIONS////////////////////////////////////////////////////////////////

/*
Test Matrix Transpose
*/
void test_matrix_transpose();

/*
Test Matrix Multiplication.
Compares M1(4x1) * M2(1x4) 
M1- >[1, 2, 3, 4]
M2 -> 1, 2, 3, 4]
*/
void test_matrix_mult();


///////////////////////////////////////////////////LAYER FUNCTIONS////////////////////////////////////////////////////////////////


/*
Test Init Layer
*/
void test_init_layer();

/*
Test Free Layer
*/
void test_free_layer();

/*
Test Forward Pass.
Includes test for SOFTMAX and RELU
*/
void test_forward_pass();

/*
Test Loss (Categorical Cross Entropy)
*/
void test_loss_categorical();

/*
Test Accuracy
*/
void test_accuracy();

/*
Test Backwards softmax
*/
void test_backward_softmax();

/*
Test Backwards ReLu
*/
void test_backward_ReLu();



/*
Test Linear Algebra Functions
*/



///////////////////////////////////////////////////NETWORK FUNCTIONS////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////TEST ALL METHODS/////////////////////////////////////////////////////////////////

/*
Tests Every Method defined above.
*/
void test_all_methods();


#endif
