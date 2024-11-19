#include "backward.h"

//////////////////////////////////////////////////// Backward Pass Methods //////////////////////////////////////////////////////////////

void backward_reLu(matrix* input_gradients, layer_dense* layer) {

    // Print debug info
    // printf("---------LAYER ---------\n");
    // printf("----Inputs---\n");
    // print_matrix(layer->inputs);
    // printf("----Input Gradients---\n");
    // print_matrix(input_gradients);
    // printf("----Layer_Outputs(After Activation)---\n");
    // print_matrix(layer->post_activation_output);

    /*
    Find Gradient of the ReLu of the layer.
    STEP 1: Find Gradient of ReLu based off of pre activation output of the layer (ie. data that hasnt had ReLU applied yet.)
    STEP 2: Perform an element by element multiplication of the input gradients from the layer before to the gradients found in step 1.
    */

    // Allocate memory for ReLU gradient
    matrix* relu_gradients = malloc(sizeof(matrix));
    relu_gradients->dim1 = layer->pre_activation_output->dim1;
    relu_gradients->dim2 = layer->pre_activation_output->dim2;
    relu_gradients->data = (double*) calloc(relu_gradients->dim1 * relu_gradients->dim2, sizeof(double));

    // Check memory allocation
    if (relu_gradients->data == NULL) {
        fprintf(stderr, "Error in memory allocation for relu gradient in backward relu.\n");
        exit(1);
    }

    // Iterate through every value in layer post activation output to get relu gradients
    for (int i = 0; i < layer->pre_activation_output->dim1 * layer->pre_activation_output->dim2; i++) {
        if (layer->pre_activation_output->data[i] >= 0) {
            relu_gradients->data[i] = 1;
        }
        else {
            relu_gradients->data[i] = 0;
        }
    }

    // Check dimensions for element by element multiplication of input gradients and relu gradients.
    if (input_gradients->dim1 != relu_gradients->dim1 || input_gradients->dim2 != relu_gradients->dim2) {
        fprintf(stderr,"Error: Dimensionality mismatch between relu gradients and input gradients in backwards relu.\n");
        printf("input_gradients(%d x %d) != relu_gradients(%d x %d)\n", input_gradients->dim1, input_gradients->dim2,
        relu_gradients->dim1, relu_gradients->dim2);
        free(relu_gradients->data);
        exit(1);
    }

    // Element by element mult of input gradients and relu gradients
    for(int i = 0; i < relu_gradients->dim1; i++) {
        for (int j = 0; j < relu_gradients->dim2; j++) {
            relu_gradients->data[i * relu_gradients->dim2 + j] = relu_gradients->data[i * relu_gradients->dim2 + j] * 
            input_gradients->data[i * relu_gradients->dim2 + j];
        }
    }

    /*
    Find Gradient of the Weights and Biases of the layer.
    STEP 1: Find the gradient of weights
        -> inputs coming from the layer before transposed dot the ReLu gradients calculated earlier. 
    STEP 2: Find the gradients of biases
        -> summation of the ReLu gradients across eatch example in the batch: returns dimensions 1 x 5
    */

    // Calculate weight gradients

    // Transpose inputs
    matrix* inputs_transposed = transpose_matrix(layer->inputs);

    // Check dimensions
    if(inputs_transposed->dim2 != relu_gradients-> dim1) {
        fprintf(stderr, "Error: Dimensionality mismatch between inputs_transposed and relu_gradients in weight calculation.\n");
        free(inputs_transposed);
        free(relu_gradients);
        exit(1);
    }
    // Define max gradient for exploding gradients
    double max_gradient = 100.0;

    // Perform the dot product
    layer->dweights = matrix_mult(inputs_transposed, relu_gradients);
    // clip_gradients(layer->dweights->data, layer->dweights->dim1 * layer->dweights->dim2, -2, 2);



    // Calculate bias gradients
    // Sum the relu gradients for each example in the batch of inputs
    for (int j = 0; j < layer->dbiases->dim2; j++) {
        for(int i = 0; i < relu_gradients->dim1; i++) {
            // sum across rows
            layer->dbiases->data[j] += relu_gradients->data[i * relu_gradients->dim2 + j];
            if (layer->dbiases->data[j] > max_gradient) {
                layer->dbiases->data[j] = max_gradient;
            }
        }
    }

    // clip_gradients(layer->dbiases->data, layer->dbiases->dim1 * layer->dbiases->dim2, -2, 2);

    if (layer->useRegularization) {
        // weights
        for (int i = 0; i < layer->dweights->dim1 * layer->dweights->dim2; i++) {
            // L2 gradients
            layer->dweights->data[i] += 2 * layer->lambda_l2 * layer->weights->data[i];

            // L1 gradients (1 if > 0, -1 if < 0)
            layer->dweights->data[i] += layer->lambda_l1 * (layer->weights->data[i] >= 0.0 ? 1.0 : -1.0);
        }
        // biases
        for (int i = 0; i < layer->dbiases->dim1 * layer->dbiases->dim2; i++) {
            // L2 gradients
            layer->dbiases->data[i] += 2 * layer->lambda_l2 * (layer->biases->data[i]);

            // L1 gradients (1 if > 0, -1 if < 0)
            layer->dbiases->data[i] += layer->lambda_l1 * (layer->biases->data[i] >= 0 ? 1.0: -1.0);
        }
    }

    // Calculate gradients for the input

    // Transpose weights
    matrix* weights_transposed = transpose_matrix(layer->weights);

    // Check dimensions
    if (relu_gradients->dim2 != weights_transposed->dim1) {
        fprintf(stderr, "Error: Dimensionality mismatch between relu gradients and weights transposed in backwards RELU\n");
        free(weights_transposed->data);
        free(weights_transposed);
        free(relu_gradients->data);
        free(relu_gradients);
        free(inputs_transposed->data);
        free(inputs_transposed);
        exit(1);
    }

    // Dot product of relu_gradients and weights transposed
    matrix* output_gradients= matrix_mult(relu_gradients, weights_transposed);

    // Copy to dinputs

    memcpy(layer->dinputs->data, output_gradients->data, layer->dinputs->dim1 * layer->dinputs->dim2 * sizeof(double));

    // clip_gradients(layer->dinputs->data, layer->dinputs->dim1 * layer->dinputs->dim2, -2, 2);

    // printf("GradientsRELU: %f\n", layer->dinputs->data[0]);

    // Final dimensionality check
    if (layer->weights->dim1 != layer->dweights->dim1 || layer->weights->dim2 != layer->dweights->dim2) {
        fprintf(stderr, "Error. Dimensionality mismatch between dweights and weights in backwards ReLu.\n");
        free(output_gradients->data);
        free(output_gradients);
        free(weights_transposed->data);
        free(weights_transposed);
        free(relu_gradients->data);
        free(relu_gradients);
        free(inputs_transposed->data);
        free(inputs_transposed);
    }

    if (layer->biases->dim1 != layer->dbiases->dim1 || layer->biases->dim2 != layer->dbiases->dim2) {
        fprintf(stderr, "Error. Dimensionality mismatch between dbiases and biases in backwards ReLu.\n");
        free(output_gradients->data);
        free(output_gradients);
        free(weights_transposed->data);
        free(weights_transposed);
        free(relu_gradients->data);
        free(relu_gradients);
        free(inputs_transposed->data);
        free(inputs_transposed);
    }

    if (layer->inputs->dim1 != layer->dinputs->dim1 || layer->inputs->dim2 != layer->dinputs->dim2) {
        fprintf(stderr, "Error. Dimensionality mismatch between dinputs and inputs in backwards ReLu.\n");
        free(output_gradients->data);
        free(output_gradients);
        free(weights_transposed->data);
        free(weights_transposed);
        free(relu_gradients->data);
        free(relu_gradients);
        free(inputs_transposed->data);
        free(inputs_transposed);
    }

    // debugging
    // printf("-------- WEIGHT GRADIENTS------------\n");
    // print_matrix(layer->dweights);

    // printf("--------BIAS GRADIENTS------------\n");
    // print_matrix(layer->dbiases);

    // printf("--------INPUT GRADIENTS ------------\n");
    // print_matrix(layer->dinputs);

    // free unused memory
    free(output_gradients->data);
    free(output_gradients);
    free(weights_transposed->data);
    free(weights_transposed);
    free(relu_gradients->data);
    free(relu_gradients);
    free(inputs_transposed->data);
    free(inputs_transposed);
}

void backwards_softmax_and_loss(matrix* true_labels, layer_dense* layer) {

    // // Print debug info
    // printf("---------LAYER ---------\n");
    // printf("----Layer_Inputs---\n");
    // print_matrix(layer->inputs);
    // printf("----Layer_Outputs(After Activation)---\n");
    // print_matrix(layer->post_activation_output);


    // Check dimensionality
    if (layer->post_activation_output->dim1 != true_labels->dim1 || layer->post_activation_output->dim2 != true_labels->dim2) {
        fprintf(stderr, "Error: Dimensionality mismatch between true labels and predictions in backwards softmax.\n");
        printf("Inputs:(%d x %d) != True:(%d x %d)\n", layer->post_activation_output->dim1, layer->post_activation_output->dim2, 
        true_labels->dim1, true_labels->dim2);
        exit(1);
    }

    // Calculate softmax loss partial derivatives

    // Allocate memory for loss gradients
    matrix* loss_gradients = malloc(sizeof(matrix));

    // same dimensions as the input (batch_size x neurons)
    // same dimensions as the true_labels
    loss_gradients->dim1 = layer->post_activation_output->dim1;
    loss_gradients->dim2 = layer->post_activation_output->dim2;

    loss_gradients->data = (double*) calloc(loss_gradients->dim1 * loss_gradients->dim2, sizeof(double));

    // Check memory allocation
    if(loss_gradients->data == NULL) {
        fprintf(stderr, "Error in memory allocation for loss gradients in softmax backprop.\n");
        free(loss_gradients);
        exit(1);
    }

    // For each example in the input batch
    for (int i = 0; i < loss_gradients->dim1; i++){
        // For each neuron in the input vector
        for(int j = 0; j < loss_gradients->dim2; j++) {
            loss_gradients->data[i * loss_gradients->dim2 + j] = layer->post_activation_output->data[i * loss_gradients->dim2 + j] - 
            true_labels->data[i * loss_gradients->dim2 + j];
        }
    }

    // Calculate layer weight derivatives
    // dot product of inputs for the layer and loss_gradients calculated above.

    // Transpose layer inputs
    matrix* inputs_T = transpose_matrix(layer->inputs);

    // Check dimensions
    if(inputs_T->dim2 != loss_gradients->dim1) {
        fprintf(stderr, "Error: Dimensionality mismatch for inputs transposed in backwards softmax.\n");
        free(loss_gradients->data);
        free(loss_gradients);
        free(inputs_T->data);
        exit(1);
    }

    /*
    dweights and dbiases not used in backprop here, but later in optimization.
    */

    // Calculate dweights -> dont need to allocate memory as matrix_mult does that.
    layer->dweights = matrix_mult(inputs_T, loss_gradients);

    // clip_gradients(layer->dweights->data, layer->dweights->dim1 * layer->dweights->dim2, -2, 2);

    // Calculate layer bias derivatives

    // Check memory allocation
    if (layer->dbiases->data == NULL) {
        fprintf(stderr, "Error in memory allocation for dbiases in softmax backprop. \n");
        free(loss_gradients->data);
        free(loss_gradients);
        free(inputs_T->data);
        exit(1);    
    }

    // Sum the loss gradients for each example in the batch of inputs
    for (int j = 0; j < layer->dbiases->dim2; j++) {
        for(int i = 0; i < layer->post_activation_output->dim1; i++) {
            // sum across rows
            layer->dbiases->data[j] += loss_gradients->data[i * loss_gradients->dim2 + j];
        }
    }
    // clip_gradients(layer->dbiases->data, layer->dbiases->dim1 * layer->dbiases->dim2, -2, 2);

    // Add regularization derivatives to dweights and dbiases

    // Check if using regularization
    if (layer->useRegularization) {
        // weights
        for (int i = 0; i < layer->dweights->dim1 * layer->dweights->dim2; i++) {
            // L2 gradients
            layer->dweights->data[i] += 2 * layer->lambda_l2 * layer->weights->data[i];

            // L1 gradients (1 if > 0, -1 if < 0)
            layer->dweights->data[i] += layer->lambda_l1 * (layer->weights->data[i] >= 0.0 ? 1.0 : -1.0);
        }
        // biases
        for (int i = 0; i < layer->dbiases->dim1 * layer->dbiases->dim2; i++) {
            // L2 gradients
            layer->dbiases->data[i] += 2 * layer->lambda_l2 * (layer->biases->data[i]);

            // L1 gradients (1 if > 0, -1 if < 0)
            layer->dbiases->data[i] += layer->lambda_l1 * (layer->biases->data[i] >= 0 ? 1.0: -1.0);
        }
    }

    // Backpropogate derivatives for previous layer

    // Transpose weights for layer
    matrix* weights_transposed = transpose_matrix(layer->weights);

    // Check dimensions
    if(loss_gradients->dim2 != weights_transposed->dim1) {
        fprintf(stderr, "Error: Dimensionality mismatch for weights transposed in backprop softmax function.\n");
        free(loss_gradients->data);
        free(loss_gradients);
        free(inputs_T->data);
        free(layer->dbiases->data);
        free(layer->dbiases);
        free(layer->dweights->data);
        free(layer->dweights);
        exit(1);    
    }

    // Calculate backprop derivative to pass to layer previous
    matrix* output_gradients = matrix_mult(loss_gradients, weights_transposed);

    // Ensure dimensions and sizes match
    if (output_gradients->dim1 != layer->dinputs->dim1 || output_gradients->dim2 != layer->dinputs->dim2) {
        fprintf(stderr, "Error, dinputs does not match calculated gradient dimensions in backwards softmax.\n");
        free(loss_gradients->data);
        free(loss_gradients);
        free(inputs_T->data);
        free(layer->dbiases->data);
        free(layer->dbiases);
        free(layer->dweights->data);
        free(layer->dweights);
        exit(1);    
    }

    // Save to layer data structure
    memcpy(layer->dinputs->data, output_gradients->data, layer->dinputs->dim1 * layer->dinputs->dim2 * sizeof(double));

    // clip_gradients(layer->dinputs->data, layer->dinputs->dim1 * layer->dinputs->dim2, -2, 2);

    // Final dimensionality check
    if (layer->weights->dim1 != layer->dweights->dim1 || layer->weights->dim2 != layer->dweights->dim2) {
        fprintf(stderr, "Error. Dimensionality mismatch between dweights and weights in backwards ReLu.\n");
        free(output_gradients->data);
        free(output_gradients);
        free(weights_transposed->data);
        free(weights_transposed);
        free(loss_gradients->data);
        free(loss_gradients);
    }

    if (layer->biases->dim1 != layer->dbiases->dim1 || layer->biases->dim2 != layer->dbiases->dim2) {
        fprintf(stderr, "Error. Dimensionality mismatch between dbiases and biases in backwards ReLu.\n");
        free(output_gradients->data);
        free(output_gradients);
        free(weights_transposed->data);
        free(weights_transposed);
        free(loss_gradients->data);
        free(loss_gradients);
    }

    if (layer->inputs->dim1 != layer->dinputs->dim1 || layer->inputs->dim2 != layer->dinputs->dim2) {
        fprintf(stderr, "Error. Dimensionality mismatch between dinputs and inputs in backwards ReLu.\n");
        free(output_gradients->data);
        free(output_gradients);
        free(weights_transposed->data);
        free(weights_transposed);
        free(loss_gradients->data);
        free(loss_gradients);
    }

    // Debug printing
    // printf("-------WEIGHT GRADIENTS------\n");
    // print_matrix(layer->dweights);

    // printf("-------BIAS GRADIENTS------\n");
    // print_matrix(layer->dbiases);

    // printf("-------INPUT(SOFTMAX) GRADIENTS------\n");
    // print_matrix(layer->dinputs);

    // free unused memory
    free(output_gradients->data);
    free(output_gradients);
    free(weights_transposed->data);
    free(weights_transposed);
    free(loss_gradients->data);
    free(loss_gradients);
}