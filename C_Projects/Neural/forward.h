#ifndef FORWARD_H
#define FORWARD_H

#include "utils.h"

//////////////////////////////////////////////////// Forward Pass Methods //////////////////////////////////////////////////////////////


/*
Performs a forward pass using a batch of inputs and a given layer. 
Can take in either a batch of inputs, or a single sample. Works with the layer dense object.
Weights are randomly assigned, biases are initialized to 0. Can handle operations for all activation types.
Stores outputs in the layer dense object.
*/
void forward_pass(matrix* inputs, layer_dense* layer);

/*
Takes in a set of outputs from multiplying inputs and weights and rectifies them if less than 0.
Rectifies the same pointer "batch_input"
*/
void forward_reLu(matrix* batch_input);

/*
Calculate the probabilities for classification problems using SoftMax activation.
Step 1: Subtract maximum value from each value in the input batch to ensure numerical stability (no large exponentiations)
Step 2: Calculate exponentials and sum them
Step 3: Normalize exponentials by dividing by the sum to get probabilities

Output is size of "batch_input"
*/
void forward_softMax(matrix* batch_input);







#endif