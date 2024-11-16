#ifndef BACKWARD_H
#define BACKWARD_H

#include "utils.h"

//////////////////////////////////////////////////// Backward Pass Methods //////////////////////////////////////////////////////////////

/*
Takes in gradients from layer above, and the current layer.
Calculates the derivatives of 
    >ReLU activation
    >Weights
    >Biases
    >Inputs
Used to send the derivatives of input (gradients) backwards to the layer before.
*/
void backward_reLu(matrix* input_gradients, layer_dense* layer);

/*
Takes in true labels used for classification and the current layer.
Calculates the derivatives of 
    >SoftMax activation
    >Weights
    >Biases
    >Inputs
Used to send the derivatives of input (gradients) backwards to the layer before.
*/
void backwards_softmax_and_loss(matrix* true_labels, layer_dense* layer);


#endif