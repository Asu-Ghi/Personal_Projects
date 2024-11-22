#include "test_functions.h"

///////////////////////////////////////////////////LINEAR ALGEBRA FUNCTIONS////////////////////////////////////////////////////////////////

/*
Test Matrix Transpose.
*/
void test_matrix_transpose() {
    printf("//////////////////////////////////////////////////////\n");
    printf("////////////// START TEST MATRIX TRANSPOSE ///////////\n");
    printf("//////////////////////////////////////////////////////\n");
    matrix Matrix;
    Matrix.dim1 = 1;
    Matrix.dim2 = 4;
    Matrix.data = (double*) calloc(Matrix.dim1 * Matrix.dim2, sizeof(double));

    matrix* transposed_matrix = transpose_matrix(&Matrix);
    
    // Check dimensions
    if (transposed_matrix->dim1 != Matrix.dim2 || transposed_matrix->dim2 != Matrix.dim1) {
        fprintf(stderr, "Error: Test_Matrix_Transpose has incorrect dimensions after transpose.\n");
        free(transposed_matrix->data);
        free(transposed_matrix);
        exit(1);
    }


    // Print test result
    printf("Transpose Matrix performed as expected.\n");
    printf("------------------------------\n");

    // Free data
    free(transposed_matrix->data);
    free(transposed_matrix);
    printf("//////////////////////////////////////////////////////\n");
    printf("////////////// END TEST MATRIX TRANSPOSE /////////////\n");
    printf("//////////////////////////////////////////////////////\n");
}

void test_matrix_mult() {
    printf("//////////////////////////////////////////////////////\n");
    printf("////////////// START TEST MATRIX MULT ////////////////\n");
    printf("//////////////////////////////////////////////////////\n");
    // Init M1
    matrix M1;
    M1.dim1 = 4;
    M1.dim2 = 1;
    double m1_data[4] = {1.0, 2.0, 3.0, 4.0};
    M1.data = m1_data;

    // Init M2
    matrix M2;
    M2.dim1 = 1;
    M2.dim2 = 4;
    double m2_data[4] = {1.0, 2.0, 3.0, 4.0};
    M2.data = m2_data;

    // Init valid results
    matrix valid_result;
    valid_result.dim1 = 4;
    valid_result.dim2 = 4;
    double valid_result_data[16] = {1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12, 4, 8, 12, 16};
    valid_result.data = valid_result_data;

    // Perform Matrix Mult
    matrix* test_result = matrix_mult(&M1, &M2);

    // Check to see if initialized properly
    if (test_result == NULL || test_result->data == NULL) {
        fprintf(stderr, "Error: Matrix Mult failed in test_matrix_mult.\n");
        exit(1);
    }

    // Compare output dimensions
    if (test_result->dim1 != valid_result.dim1 || test_result->dim2 != valid_result.dim2) {
        fprintf(stderr, "Error: Mismatching Dimensions between test and valid matrices in test_matrix_mult.\n");
        free(test_result->data);
        free(test_result);
        exit(1);
    }

    // Compare data
    for (int i = 0; i < valid_result.dim1 * valid_result.dim2; i++) {
        if (valid_result.data[i] != test_result->data[i]) {
            fprintf(stderr, "Error: Data does not corespond to true output in test_matrix_mult\n");
            print_matrix(test_result);
            printf("------------------------------\n");
            print_matrix(&valid_result);
            free(test_result->data);
            free(test_result);
            exit(1);
        }
    }

    // Print test status
    printf("matrix_mult performed as expected.\n");
    printf("------------------------------\n");

    // free test_result 
    free(test_result);
    printf("//////////////////////////////////////////////////////\n");
    printf("//////////////// END TEST MATRIX MULT ////////////////\n");
    printf("//////////////////////////////////////////////////////\n");
}

///////////////////////////////////////////////////LAYER FUNCTIONS////////////////////////////////////////////////////////////////

void test_init_layer() {
    printf("//////////////////////////////////////////////////////\n");
    printf("/////////////// START TEST INIT LAYER ////////////////\n");
    printf("//////////////////////////////////////////////////////\n");
    int num_input_features = 3;
    int num_neurons = 4;
    ActivationType test_type = RELU;
    int batch_size = 10;

    layer_dense* test_layer = init_layer(num_input_features, num_neurons, test_type, SGD_MOMENTUM);

    // Check Weights Dimensions
    if (test_layer->weights->dim1 != 3 || test_layer->weights->dim2 != 4) {
        fprintf(stderr, "Error: Weight Dimensions are not accurate in test_init_layer.\n");
        free_layer(test_layer);
        exit(1);
    }

    // Check dWeights Dimensions
    if (test_layer->dweights->dim1 != 3 || test_layer->dweights->dim2 != 4) {
        fprintf(stderr, "Error: dWeight Dimensions are not accurate in test_init_layer.\n");
        free_layer(test_layer);
        exit(1);
    }

    // Check Bias Dimensions
    if (test_layer->biases->dim1 != 1 || test_layer->biases->dim2 != 4) {
        fprintf(stderr, "Error: Bias Dimensions are not accurate in test_init_layer.\n");
        free_layer(test_layer);
        exit(1);
    }

    // Check dBias Dimensions
    if (test_layer->dbiases->dim1 != 1 || test_layer->dbiases->dim2 != 4) {
        fprintf(stderr, "Error: dBias Dimensions are not accurate in test_init_layer.\n");
        free_layer(test_layer);
        exit(1);
    }

    // Check Inputs Dimensions
    if (test_layer->inputs->dim1 != 10 || test_layer->inputs->dim2 != 3) {
        fprintf(stderr, "Error: Inputs Dimensions are not accurate in test_init_layer.\n");
        free_layer(test_layer);
        exit(1);
    }

    // Check dInputs Dimensions
    if (test_layer->dinputs->dim1 != 10 || test_layer->dinputs->dim2 != 3) {
        fprintf(stderr, "Error: dInputs Dimensions are not accurate in test_init_layer.\n");
        free_layer(test_layer);
        exit(1);
    }

    // Check PreActivation Output Dimensions
    if (test_layer->pre_activation_output->dim1 != 10 || test_layer->pre_activation_output->dim2 != 4) {
        fprintf(stderr, "Error: PreActivation Dimensions are not accurate in test_init_layer.\n");
        free_layer(test_layer);
        exit(1);
    }

    // Check PostActivation Output Dimensions
    if (test_layer->post_activation_output->dim1 != 10 || test_layer->post_activation_output->dim2 != 4) {
        fprintf(stderr, "Error: PostActivation Dimensions are not accurate in test_init_layer.\n");
        free_layer(test_layer);
        exit(1);
    }

    // Print Weights and Bias Matrices

    printf("---------Test Weights--------\n");
    printf("(%d x %d)\n", test_layer->weights->dim1, test_layer->weights->dim2);
    print_matrix(test_layer->weights);

    printf("---------Test Biases--------\n");
    print_matrix(test_layer->biases);
    printf("(%d x %d)\n", test_layer->biases->dim1, test_layer->biases->dim2);
    printf("------------------------------\n");

    // Print Test Status
    printf("Init_Layer works as expected.\n");
    printf("------------------------------\n");

    // Free layer
    free_layer(test_layer);

    printf("//////////////////////////////////////////////////////\n");
    printf("///////////////// END TEST INIT LAYER ////////////////\n");
    printf("//////////////////////////////////////////////////////\n");
}

void test_forward_pass() {
    // Initialize input matrix
    matrix inputs;
    double input_data[9] = {1, 2, 3, 1, 2, 3, 1, 2, 3};
    inputs.data = input_data;

    inputs.dim1 = BATCH_SIZE;
    inputs.dim2 = NUM_BATCH_FEATURES;
    
    printf("//////////////////////////////////////////////////////\n");
    printf("/////////////// START TEST FORWARD PASS///////////////\n");
    printf("//////////////////////////////////////////////////////\n");
    /*
    Test First Layer Forward Pass.
    */
    printf("---------LAYER 1 (FIRST)------------\n");
    layer_dense* first_layer = init_layer(NUM_BATCH_FEATURES, NUM_NEURONS_1, RELU, SGD_MOMENTUM);
    forward_pass(&inputs, first_layer);

    // Check Dimensions for PreActivation Outputs
    if (first_layer->pre_activation_output->dim1 != BATCH_SIZE || first_layer->pre_activation_output->dim2 != first_layer->num_neurons) {
        fprintf(stderr,"Error: Output matrix dimensions incorrect for first layer in test_forward_pass.\n");
        printf("-----PRE ACTIVATION OUTPUT------\n");
        print_matrix(first_layer->pre_activation_output);
        free_layer(first_layer);
        exit(1);
    }

    // Check Dimensions for PostActivation Outputs
    if (first_layer->post_activation_output->dim1 != BATCH_SIZE || first_layer->post_activation_output->dim2 != first_layer->num_neurons) {
        fprintf(stderr,"Error: Output matrix dimensions incorrect for first layer in test_forward_pass.\n");
        printf("-----POST ACTIVATION OUTPUT (RELU)------\n");
        print_matrix(first_layer->post_activation_output);
        free_layer(first_layer);
        exit(1);
    }

    // Print Outputs for the first layer.
    printf("-----PRE ACTIVATION OUTPUT------\n");
    print_matrix(first_layer->pre_activation_output);
    printf("-----POST ACTIVATION OUTPUT (RELU)------\n");
    print_matrix(first_layer->post_activation_output);

    /*
    Test Hidden Layer Forward Pass.
    */
    printf("---------LAYER 2 (HIDDEN)------------\n");
    layer_dense* second_layer = init_layer(NUM_INPUT_FEATURES_2, NUM_NEURONS_2, RELU, SGD_MOMENTUM);
    forward_pass(first_layer->post_activation_output, second_layer);

    // Check Dimensions for PreActivation Outputs
    if (second_layer->pre_activation_output->dim1 != BATCH_SIZE || second_layer->pre_activation_output->dim2 != second_layer->num_neurons) {
        fprintf(stderr,"Error: Output matrix dimensions incorrect for first layer in test_forward_pass.\n");
        printf("-----PRE ACTIVATION OUTPUT (RELU)------\n");
        print_matrix(second_layer->pre_activation_output);
        free_layer(second_layer);
        exit(1);
    }

    // Check Dimensions for PostActivation Outputs
    if (second_layer->post_activation_output->dim1 != BATCH_SIZE || second_layer->post_activation_output->dim2 != second_layer->num_neurons) {
        fprintf(stderr,"Error: Output matrix dimensions incorrect for first layer in test_forward_pass.\n");
        printf("-----POST ACTIVATION OUTPUT (RELU)------\n");
        print_matrix(second_layer->post_activation_output);
        free_layer(second_layer);
        exit(1);
    }

    // Print Outputs for the first layer.
    printf("-----PRE ACTIVATION OUTPUT------\n");
    print_matrix(second_layer->pre_activation_output);
    printf("-----POST ACTIVATION OUTPUT (RELU)------\n");
    print_matrix(second_layer->post_activation_output);


    /*
    Test Final Layer Forward Pass.
    */
    printf("---------LAYER 3 (LAST)------------\n");
    layer_dense* third_layer = init_layer(NUM_INPUT_FEATURES_3, NUM_NEURONS_3, SOFTMAX, SGD_MOMENTUM);
    forward_pass(second_layer->post_activation_output, third_layer);
    
    // Check Dimensions for PreActivation Outputs
    if (third_layer->pre_activation_output->dim1 != BATCH_SIZE || third_layer->pre_activation_output->dim2 != third_layer->num_neurons) {
        fprintf(stderr,"Error: Output matrix dimensions incorrect for first layer in test_forward_pass.\n");
        printf("-----PRE ACTIVATION OUTPUT (SOFTMAX)------\n");
        print_matrix(third_layer->pre_activation_output);
        free_layer(third_layer);
        exit(1);
    }

    // Check Dimensions for PostActivation Outputs
    if (third_layer->post_activation_output->dim1 != BATCH_SIZE || third_layer->post_activation_output->dim2 != third_layer->num_neurons) {
        fprintf(stderr,"Error: Output matrix dimensions incorrect for first layer in test_forward_pass.\n");
        printf("-----POST ACTIVATION OUTPUT (SOFTMAX)------\n");
        print_matrix(third_layer->post_activation_output);
        free_layer(third_layer);
        exit(1);
    }

    // Print Outputs for the first layer.
    printf("-----PRE ACTIVATION OUTPUT------\n");
    print_matrix(third_layer->pre_activation_output);
    printf("-----POST ACTIVATION OUTPUT (SOFTMAX)------\n");
    print_matrix(third_layer->post_activation_output);

    // Free Memory
    free_layer(first_layer);
    free_layer(second_layer);
    free_layer(third_layer);

    printf("//////////////////////////////////////////////////////\n");
    printf("///////////////// END TEST FORWARD PASS///////////////\n");
    printf("//////////////////////////////////////////////////////\n");
}

void test_accuracy() {
    printf("//////////////////////////////////////////////////////\n");
    printf("///////////////// START TEST ACCURACY ////////////////\n");
    printf("//////////////////////////////////////////////////////\n");
    // Y_Pred (One Hot) (BATCH SIZE x BATCH FEATURES)
    matrix class_targets;
    class_targets.dim1 = BATCH_SIZE;
    class_targets.dim2 = NUM_BATCH_FEATURES;
    double class_target_data[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    class_targets.data = class_target_data;

    // Initialize input matrix
    matrix inputs;
    double input_data[9] = {1, 2, 3, 1, 2, 3, 1, 2, 3};
    inputs.data = input_data;

    inputs.dim1 = BATCH_SIZE;
    inputs.dim2 = NUM_BATCH_FEATURES;

    // Create Layers for Forward Pass
    layer_dense* layer_one = init_layer(NUM_BATCH_FEATURES, NUM_NEURONS_1, RELU, SGD_MOMENTUM);
    layer_dense* layer_two = init_layer(NUM_INPUT_FEATURES_2, NUM_NEURONS_2, RELU, SGD_MOMENTUM);
    layer_dense* layer_three = init_layer(NUM_INPUT_FEATURES_3, NUM_NEURONS_3, SOFTMAX, SGD_MOMENTUM);

    // Perform forward pass
    forward_pass(&inputs, layer_one);
    forward_pass(layer_one->post_activation_output, layer_two);
    forward_pass(layer_two->post_activation_output, layer_three);
    
    // Calculate accuracy
    double accuracy = calculate_accuracy(&class_targets, layer_three, ONE_HOT);
    printf("Layer 3 Outputs (Softmax Outputs)\n");
    print_matrix(layer_three->post_activation_output);
    printf("Y_Pred (One Hot Labels)\n");
    print_matrix(&class_targets);
    printf("Expected Accuracy = %f\n", accuracy);
    printf("Calculated Accuracy = %f\n", 0.333333);


    printf("//////////////////////////////////////////////////////\n");
    printf("/////////////////// END TEST ACCURACY ////////////////\n");
    printf("//////////////////////////////////////////////////////\n");

    // Free Memory
    free_layer(layer_one);
    free_layer(layer_two);
    free_layer(layer_three);
}

void test_loss_categorical() {

    printf("//////////////////////////////////////////////////////\n");
    printf("///////////////// START TEST LOSS ////////////////////\n");
    printf("//////////////////////////////////////////////////////\n");
    // Y_Pred (One Hot) (BATCH SIZE x BATCH FEATURES)
    matrix class_targets;
    class_targets.dim1 = BATCH_SIZE;
    class_targets.dim2 = NUM_BATCH_FEATURES;
    double class_target_data[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    class_targets.data = class_target_data;

    // Initialize input matrix
    matrix inputs;
    double input_data[9] = {1, 2, 3, 1, 2, 3, 1, 2, 3};
    inputs.data = input_data;

    inputs.dim1 = BATCH_SIZE;
    inputs.dim2 = NUM_BATCH_FEATURES;

    // Create Layers for Forward Pass
    layer_dense* layer_one = init_layer(NUM_BATCH_FEATURES, NUM_NEURONS_1, RELU, SGD_MOMENTUM);
    layer_dense* layer_two = init_layer(NUM_INPUT_FEATURES_2, NUM_NEURONS_2, RELU, SGD_MOMENTUM);
    layer_dense* layer_three = init_layer(NUM_INPUT_FEATURES_3, NUM_NEURONS_3, SOFTMAX, SGD_MOMENTUM);

    // Perform forward pass
    forward_pass(&inputs, layer_one);
    forward_pass(layer_one->post_activation_output, layer_two);
    forward_pass(layer_two->post_activation_output, layer_three);

    // Get loss for each example 
    matrix* losses_per_example = loss_categorical_cross_entropy(&class_targets, layer_three, ONE_HOT);

    // Check dimensions of loss (must equal batch_size x 1);
    if (losses_per_example->dim1 != BATCH_SIZE || losses_per_example->dim2 != 1) {
        fprintf(stderr, "Error: Dimenson of losses does not equal batch size in test_loss_categorical.\n");
        print_matrix(losses_per_example);
        printf("(%d x %d)\n", losses_per_example->dim1, losses_per_example->dim2);
        free(losses_per_example);
        free_layer(layer_one);
        free_layer(layer_two);
        free_layer(layer_three);
        exit(1);
    }

    // Calculate batch loss
    double batch_loss = 0.0;
    for (int i = 0; i < BATCH_SIZE; i++) {
        batch_loss += losses_per_example->data[i];
    }
    batch_loss = batch_loss / BATCH_SIZE;

    printf("Layer 3 Outputs (Softmax Outputs)\n");
    print_matrix(layer_three->post_activation_output);
    printf("Y_Pred (One Hot Labels)\n");
    print_matrix(&class_targets);
    printf("Individual example losses: \n");
    print_matrix(losses_per_example);
    printf("Expected Total Batch Loss: %f\n", 2.005522);
    printf("Calculated Total Batch Loss: %f\n", batch_loss);

    // Free memory
    free(losses_per_example);
    free_layer(layer_one);
    free_layer(layer_two);
    free_layer(layer_three);

    printf("//////////////////////////////////////////////////////\n");
    printf("//////////////////// END TEST LOSS ///////////////////\n");
    printf("//////////////////////////////////////////////////////\n");

}

void test_backward_pass() {
    printf("//////////////////////////////////////////////////////\n");
    printf("//////////// START TEST BACKWARDS ////////////////////\n");
    printf("//////////////////////////////////////////////////////\n");
    // Y_Pred (One Hot) (BATCH SIZE x BATCH FEATURES)
    matrix class_targets;
    class_targets.dim1 = BATCH_SIZE;
    class_targets.dim2 = NUM_BATCH_FEATURES;
    double class_target_data[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    class_targets.data = class_target_data;

    // Initialize input matrix
    matrix inputs;
    double input_data[9] = {1, 2, 3, 1, 2, 3, 1, 2, 3};
    inputs.data = input_data;

    inputs.dim1 = BATCH_SIZE;
    inputs.dim2 = NUM_BATCH_FEATURES;

    // Create Layers for Forward Pass
    layer_dense* layer_one = init_layer(NUM_BATCH_FEATURES, NUM_NEURONS_1, RELU, SGD_MOMENTUM);
    layer_dense* layer_two = init_layer(NUM_INPUT_FEATURES_2, NUM_NEURONS_2, RELU, SGD_MOMENTUM);
    layer_dense* layer_three = init_layer(NUM_INPUT_FEATURES_3, NUM_NEURONS_3, SOFTMAX, SGD_MOMENTUM);

    // Perform forward pass
    forward_pass(&inputs, layer_one);
    forward_pass(layer_one->post_activation_output, layer_two);
    forward_pass(layer_two->post_activation_output, layer_three);

    // Perform backwards pass
    
    // Outputs -> Layer 3
    backwards_softmax_and_loss(&class_targets, layer_three);

    // Check gradient dimensons

    // dweight dimenson check
    if (layer_three->dweights->dim1 != layer_three->weights->dim1 || layer_three->dweights->dim2 != layer_three->weights->dim2) {
        fprintf(stderr, "Error: Dimensionality mismatch between dweights and weights in layer 3, test_backwards.\n");
        printf("-----DWEIGHTS----\n");
        print_matrix(layer_three->dweights);
        printf("-----WEIGHTS-----\n");
        print_matrix(layer_three->weights);
        free_layer(layer_one);
        free_layer(layer_two);
        free_layer(layer_three);
        exit(1);
    }

    // dbias dimension check
    if (layer_three->dbiases->dim1 != layer_three->biases->dim1 || layer_three->dbiases->dim2 != layer_three->biases->dim2) {
        fprintf(stderr, "Error: Dimensionality mismatch between dbiases and biases in layer 3, test_backwards.\n");
        printf("-----DBIASES----\n");
        print_matrix(layer_three->dbiases);
        printf("-----BIASES-----\n");
        print_matrix(layer_three->biases);
        free_layer(layer_one);
        free_layer(layer_two);
        free_layer(layer_three);
        exit(1);
    }

    // dinputs dimension check
    if (layer_three->dinputs->dim1 != layer_three->inputs->dim1 || layer_three->dinputs->dim2 != layer_three->inputs->dim2) {
        fprintf(stderr, "Error: Dimensionality mismatch between dbiases and biases in layer 3, test_backwards.\n");
        printf("-----DINPUTS----\n");
        print_matrix(layer_three->dinputs);
        printf("-----INPUTS-----\n");
        print_matrix(layer_three->inputs);
        free_layer(layer_one);
        free_layer(layer_two);
        free_layer(layer_three);
        exit(1);
    }

    // Print gradients for layer 3
    printf("----------LAYER 3--------\n");
    printf("---------DWEIGHTS--------\n");
    print_matrix(layer_three->dweights);
    printf("---------DBIASES---------\n");
    print_matrix(layer_three->dbiases);
    printf("---------DINPUTS---------\n");
    print_matrix(layer_three->dinputs);

    // Check dimensons of dinputs for layer 3 match post activation outputs for layer 2
    if (layer_three->dinputs->dim1 != layer_two->post_activation_output->dim1 ||
         layer_three->dinputs->dim2 != layer_two->post_activation_output->dim2) {
            fprintf(stderr,"Error: Dimensionality mismatch between dinputs layer 3 and layer 2 outputs in test_backwards.\n");
            printf("-----DINPUTS----\n");
            print_matrix(layer_three->dinputs);
            printf("-----OUTPUTS-----\n");
            print_matrix(layer_two->post_activation_output);
            free_layer(layer_one);
            free_layer(layer_two);
            free_layer(layer_three);
            exit(1);        
         }

    // Layer 3 dInputs -> Layer 2
    backward_reLu(layer_three->dinputs, layer_two);

    // Check gradient dimensons

    // dweight dimenson check
    if (layer_two->dweights->dim1 != layer_two->weights->dim1 || layer_two->dweights->dim2 != layer_two->weights->dim2) {
        fprintf(stderr, "Error: Dimensionality mismatch between dweights and weights in layer 2, test_backwards.\n");
        printf("-----DWEIGHTS----\n");
        print_matrix(layer_two->dweights);
        printf("-----WEIGHTS-----\n");
        print_matrix(layer_two->weights);
        free_layer(layer_one);
        free_layer(layer_two);
        free_layer(layer_three);
        exit(1);
    }

    // dbias dimension check
    if (layer_two->dbiases->dim1 != layer_two->biases->dim1 || layer_two->dbiases->dim2 != layer_two->biases->dim2) {
        fprintf(stderr, "Error: Dimensionality mismatch between dbiases and biases in layer 2, test_backwards.\n");
        printf("-----DBIASES----\n");
        print_matrix(layer_two->dbiases);
        printf("-----BIASES-----\n");
        print_matrix(layer_two->biases);
        free_layer(layer_one);
        free_layer(layer_two);
        free_layer(layer_three);
        exit(1);
    }

    // dinputs dimension check
    if (layer_two->dinputs->dim1 != layer_two->inputs->dim1 || layer_two->dinputs->dim2 != layer_two->inputs->dim2) {
        fprintf(stderr, "Error: Dimensionality mismatch between dinputs and inputs in layer 2, test_backwards.\n");
        printf("-----DINPUTS----\n");
        print_matrix(layer_two->dinputs);
        printf("-----INPUTS-----\n");
        print_matrix(layer_two->inputs);
        free_layer(layer_one);
        free_layer(layer_two);
        free_layer(layer_three);
        exit(1);
    }

    // Print gradients for layer 2
    printf("----------LAYER 2--------\n");
    printf("---------DWEIGHTS--------\n");
    print_matrix(layer_two->dweights);
    printf("---------DBIASES---------\n");
    print_matrix(layer_two->dbiases);
    printf("---------DINPUTS---------\n");
    print_matrix(layer_two->dinputs);


    // Check dimensons of dinputs for layer 2 match post activation outputs for layer 1
    if (layer_two->dinputs->dim1 != layer_one->post_activation_output->dim1 ||
         layer_two->dinputs->dim2 != layer_one->post_activation_output->dim2) {
            fprintf(stderr,"Error: Dimensionality mismatch between dinputs layer 2 and layer 1 outputs in test_backwards.\n");
            printf("-----DINPUTS----\n");
            print_matrix(layer_two->dinputs);
            printf("-----OUTPUTS-----\n");
            print_matrix(layer_one->post_activation_output);
            free_layer(layer_one);
            free_layer(layer_two);
            free_layer(layer_three);
            exit(1);        
    }


    // Layer 2 dInputs -> Layer 1
    backward_reLu(layer_two->dinputs, layer_one);

    // Check gradient dimensons

    // dweight dimenson check
    if (layer_one->dweights->dim1 != layer_one->weights->dim1 || layer_one->dweights->dim2 != layer_one->weights->dim2) {
        fprintf(stderr, "Error: Dimensionality mismatch between dweights and weights in layer 1, test_backwards.\n");
        printf("-----DWEIGHTS----\n");
        print_matrix(layer_one->dweights);
        printf("-----WEIGHTS-----\n");
        print_matrix(layer_one->weights);
        free_layer(layer_one);
        free_layer(layer_two);
        free_layer(layer_three);
        exit(1);
    }

    // dbias dimension check
    if (layer_one->dbiases->dim1 != layer_one->biases->dim1 || layer_one->dbiases->dim2 != layer_one->biases->dim2) {
        fprintf(stderr, "Error: Dimensionality mismatch between dbiases and biases in layer 1, test_backwards.\n");
        printf("-----DBIASES----\n");
        print_matrix(layer_one->dbiases);
        printf("-----BIASES-----\n");
        print_matrix(layer_one->biases);
        free_layer(layer_one);
        free_layer(layer_two);
        free_layer(layer_three);
        exit(1);
    }

    // dinputs dimension check (see if they match the dimensons of the batch inputs)
    if (layer_one->dinputs->dim1 != layer_one->inputs->dim1 || layer_one->dinputs->dim2 != layer_one->inputs->dim2) {
        fprintf(stderr, "Error: Dimensionality mismatch between dinputs and batch inputs in layer 1, test_backwards.\n");
        printf("-----DINPUTS----\n");
        print_matrix(layer_one->dinputs);
        printf("-----INPUTS-----\n");
        print_matrix(layer_one->inputs);
        free_layer(layer_one);
        free_layer(layer_two);
        free_layer(layer_three);
        exit(1);
    }

    // Print gradients for layer 1
    printf("----------LAYER 1--------\n");
    printf("---------DWEIGHTS--------\n");
    print_matrix(layer_one->dweights);
    printf("---------DBIASES---------\n");
    print_matrix(layer_one->dbiases);
    printf("---------DINPUTS---------\n");
    print_matrix(layer_one->dinputs);


    // Free memory
    free_layer(layer_one);
    free_layer(layer_two);
    free_layer(layer_three);

    printf("//////////////////////////////////////////////////////\n");
    printf("//////////// END TEST BACKWARDS //////////////////////\n");
    printf("//////////////////////////////////////////////////////\n");

}

void test_update_params_sgd() {
    printf("//////////////////////////////////////////////////////\n");
    printf("////////////////// START TEST SGD ////////////////////\n");
    printf("//////////////////////////////////////////////////////\n");
    // Set learning rate
    double learning_rate = 0.01;

    // Initialize data and class targets
    matrix inputs;
    inputs.dim1 = BATCH_SIZE;
    inputs.dim2 = NUM_BATCH_FEATURES;
    double input_data[9] = {5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7};  // Example: 3 samples, 3 features each
    inputs.data = input_data;


    matrix class_targets;
    class_targets.dim1 = BATCH_SIZE;
    class_targets.dim2 = NUM_BATCH_FEATURES;
    double class_target_data[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // 3 samples, 3 classes
    class_targets.data = class_target_data;

    // Initialize 3 layer network
    layer_dense* layer_one = init_layer(NUM_BATCH_FEATURES, NUM_NEURONS_1, RELU, SGD_MOMENTUM);
    layer_dense* layer_two = init_layer(NUM_INPUT_FEATURES_2, NUM_NEURONS_2, RELU, SGD_MOMENTUM);
    layer_dense* layer_three = init_layer(NUM_INPUT_FEATURES_3, NUM_NEURONS_3, SOFTMAX, SGD_MOMENTUM);

    // Allocate memory for 

    // Test with epoch of 20
    for (int i = 0; i < 20; i++) {
      
        // Perform forward pass
        forward_pass(&inputs, layer_one);
        forward_pass(layer_one->post_activation_output, layer_two);
        forward_pass(layer_two->post_activation_output, layer_three);

        // Calculate loss and accuracy
        matrix* sample_losses = loss_categorical_cross_entropy(&class_targets, layer_three, ONE_HOT);
        double accuaracy = calculate_accuracy(&class_targets, layer_three, ONE_HOT);
        double batch_loss = 0.0;
        for (int j = 0; j < BATCH_SIZE; j++) {
            batch_loss+= sample_losses->data[j];
        }
        batch_loss = batch_loss / BATCH_SIZE;

        // Print loss and accuracy over epoch
        printf("Epoch: %d, Loss: %.3f, Accuaracy: %.3f\n", i, batch_loss, accuaracy);

        // Perform backward pass
        backwards_softmax_and_loss(&class_targets, layer_three);
        backward_reLu(layer_three->dinputs, layer_two);
        backward_reLu(layer_two->dinputs, layer_one);

        // Update parameters for the layers
        update_params_sgd(layer_one, &learning_rate, i, 0.0001);
        update_params_sgd(layer_two, &learning_rate, i, .0001);
        update_params_sgd(layer_three, &learning_rate, i, .0001);
    }

    printf("//////////////////////////////////////////////////////\n");
    printf("//////////////////// END TEST SGD ////////////////////\n");
    printf("//////////////////////////////////////////////////////\n");
}

///////////////////////////////////////////////////NETWORK FUNCTIONS////////////////////////////////////////////////////////////////

void test_init_neural_network() {
    printf("//////////////////////////////////////////////////////\n");
    printf("////////// START TEST INIT_NEURAL_NETWORK ////////////\n");
    printf("//////////////////////////////////////////////////////\n");
    // Create int array off defined constants
    int num_neurons_in_layers[NUM_LAYERS] = {NUM_NEURONS_LAYER_1, NUM_NEURONS_LAYER_2, NUM_NEURONS_LAYER_3};
    ActivationType activations[NUM_LAYERS] = {RELU, RELU, SOFTMAX};
    OptimizationType optimizations[NUM_LAYERS] = {SGD_MOMENTUM, SGD_MOMENTUM, SGD_MOMENTUM};
    bool regularizations[NUM_LAYERS] = {true, true, true};
    // Initialize layer based off defined constants
    NeuralNetwork* test_network = init_neural_network(NUM_LAYERS, num_neurons_in_layers, 
                                                LEARNING_RATE, activations, optimizations, regularizations, NUM_BATCH_FEATURES);

    // Check to see if layer was allocated properly
    if (test_network == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for test network in test_init_neural_network.\n");
        exit(1);
    }
    
    // Check to see if layers have correct dimensions
    if (test_network->layers[0]->num_inputs != NUM_BATCH_FEATURES || test_network->layers[0]->num_neurons!=NUM_NEURONS_LAYER_1) {
        fprintf(stderr, "Error: Layer 1 dimensions incorrect in test init neural network.\n");
        free_neural_network(test_network);
        exit(1);
    }

    if (test_network->layers[1]->num_inputs != NUM_NEURONS_LAYER_1 || test_network->layers[1]->num_neurons!=NUM_NEURONS_LAYER_2) {
        fprintf(stderr, "Error: Layer 2 dimensions incorrect in test init neural network.\n");
        free_neural_network(test_network);
        exit(1);
    }

    if (test_network->layers[2]->num_inputs != NUM_NEURONS_LAYER_2 || test_network->layers[2]->num_neurons!=NUM_NEURONS_LAYER_3) {
        fprintf(stderr, "Error: Layer 3 dimensions incorrect in test init neural network.\n");
        free_neural_network(test_network);
        exit(1);
    }
    
    // Check to see if layers have correct activation functions
    if (test_network->layers[0]->activation != RELU) {
        fprintf(stderr, "Error: Layer 1 has incorrect activation function.\n");
        free_neural_network(test_network);
        exit(1);
    }

    if (test_network->layers[1]->activation != RELU) {
        fprintf(stderr, "Error: Layer 2 has incorrect activation function.\n");
        free_neural_network(test_network);
        exit(1);
    }

    if (test_network->layers[2]->activation != SOFTMAX) {
        fprintf(stderr, "Error: Layer 3 has incorrect activation function.\n");
        free_neural_network(test_network);
        exit(1);
    }

    // Print Neural Network Info
    print_nn_info(test_network);
    
    // Free memory
    free_neural_network(test_network);
    printf("//////////////////////////////////////////////////////\n");
    printf("//////////// END TEST INIT_NEURAL_NETWORK ////////////\n");
    printf("//////////////////////////////////////////////////////\n");
}

void test_forward_pass_nn() {
    printf("//////////////////////////////////////////////////////\n");
    printf("//////////// START TEST FORWARD_PASS_NN //////////////\n");
    printf("//////////////////////////////////////////////////////\n");
    // Create int array off defined constants
    int num_neurons_in_layers[NUM_LAYERS] = {NUM_NEURONS_LAYER_1, NUM_NEURONS_LAYER_2, NUM_NEURONS_LAYER_3};
    ActivationType activations[NUM_LAYERS] = {RELU, RELU, SOFTMAX};
    OptimizationType optimizations[NUM_LAYERS] = {SGD_MOMENTUM, SGD_MOMENTUM, SGD_MOMENTUM};
    bool regularizations[NUM_LAYERS] = {true, true, true};

    // Initialize layer based off defined constants
    NeuralNetwork* network = init_neural_network(NUM_LAYERS, num_neurons_in_layers, 
                                                LEARNING_RATE, activations, optimizations,regularizations, NUM_BATCH_FEATURES);

    // Check to see if layer was allocated properly
    if (network == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for test network in test_forward_pass_nn.\n");
        exit(1);
    } 

    // Initialize input data
    matrix inputs;
    inputs.dim1 = BATCH_SIZE;
    inputs.dim2 = NUM_BATCH_FEATURES;
    double input_data[9] = {5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7};  // Example: 3 samples, 3 features each
    inputs.data = input_data;

    // Call forward pass, verify outputs
    forward_pass_nn(network, &inputs);
    printf("#################################\n");
    printf("------------LAYER 1-----------\n");
    printf("------------INPUTS-------------\n");
    print_matrix(network->layers[0]->inputs);
    printf("------------WEIGHTS-----------\n");
    print_matrix(network->layers[0]->weights);
    printf("------------BIASES-----------\n");
    print_matrix(network->layers[0]->biases);
    printf("-----PRE_ACTIVATION_OUTPUTS-----------\n");
    print_matrix(network->layers[0]->pre_activation_output);
    printf("-----POST_ACTIVATION_OUTPUTS-----------\n");
    print_matrix(network->layers[0]->post_activation_output);
    printf("------------------------------\n");
    printf("------------LAYER 2-----------\n");
    printf("------------INPUTS-------------\n");
    print_matrix(network->layers[1]->inputs);
    printf("------------WEIGHTS-----------\n");
    print_matrix(network->layers[1]->weights);
    printf("------------BIASES-----------\n");
    print_matrix(network->layers[1]->biases);
    printf("-----PRE_ACTIVATION_OUTPUTS-----------\n");
    print_matrix(network->layers[1]->pre_activation_output);
    printf("-----POST_ACTIVATION_OUTPUTS-----------\n");
    print_matrix(network->layers[1]->post_activation_output);
    printf("------------------------------\n");
    printf("------------LAYER 3-----------\n");
    printf("------------INPUTS-------------\n");
    print_matrix(network->layers[2]->inputs);
    printf("------------WEIGHTS-----------\n");
    print_matrix(network->layers[2]->weights);
    printf("------------BIASES-----------\n");
    print_matrix(network->layers[2]->biases);
    printf("-----PRE_ACTIVATION_OUTPUTS-----------\n");
    print_matrix(network->layers[2]->pre_activation_output);
    printf("-----POST_ACTIVATION_OUTPUTS-----------\n");
    print_matrix(network->layers[2]->post_activation_output);
    printf("#################################\n");
    
    // Free memory
    free_neural_network(network);

    printf("//////////////////////////////////////////////////////\n");
    printf("//////////// END TEST FORWARD_PASS_NN //////////////\n");
    printf("//////////////////////////////////////////////////////\n");

}

void test_backward_pass_nn() {
    printf("//////////////////////////////////////////////////////\n");
    printf("//////////// START TEST BACKWARDS_PASS_NN //////////////\n");
    printf("//////////////////////////////////////////////////////\n");
    // Create int array off defined constants
    int num_neurons_in_layers[NUM_LAYERS] = {NUM_NEURONS_LAYER_1, NUM_NEURONS_LAYER_2, NUM_NEURONS_LAYER_3};
    ActivationType activations[NUM_LAYERS] = {RELU, RELU, SOFTMAX};
    OptimizationType optimizations[NUM_LAYERS] = {SGD_MOMENTUM, SGD_MOMENTUM, SGD_MOMENTUM};
    bool regularizations[NUM_LAYERS] = {true, true, true};

    // Initialize layer based off defined constants
    NeuralNetwork* network = init_neural_network(NUM_LAYERS, num_neurons_in_layers, 
                                                LEARNING_RATE, activations, optimizations, regularizations, NUM_BATCH_FEATURES);

    // Check to see if layer was allocated properly
    if (network == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for test network in test_forward_pass_nn.\n");
        exit(1);
    } 

    // Initialize input data
    matrix inputs;
    inputs.dim1 = BATCH_SIZE;
    inputs.dim2 = NUM_BATCH_FEATURES;
    double input_data[9] = {5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7};  // Example: 3 samples, 3 features each
    inputs.data = input_data;

    // Initialize class target data
    matrix class_targets;
    class_targets.dim1 = BATCH_SIZE;
    class_targets.dim2 = NUM_BATCH_FEATURES;
    double class_target_data[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // 3 samples, 3 classes
    class_targets.data = class_target_data;

    // Call forward pass, 
    forward_pass_nn(network, &inputs);
    
    // Call backward pass
    backward_pass_nn(network, &class_targets);

    // Verify outputs
    printf("#################################\n");
    printf("------------LAYER 3-----------\n");
    printf("------------DWEIGHTS-------------\n");
    print_matrix(network->layers[2]->dweights);
    printf("------------DBIASES-----------\n");
    print_matrix(network->layers[2]->dbiases);
    printf("------------DINPUTS-----------\n");
    print_matrix(network->layers[2]->dinputs);
    printf("------------------------------\n");
    printf("------------LAYER 2-----------\n");
    printf("------------DWEIGHTS-------------\n");
    print_matrix(network->layers[1]->dweights);
    printf("------------DBIASES-----------\n");
    print_matrix(network->layers[1]->dbiases);
    printf("------------DINPUTS-----------\n");
    print_matrix(network->layers[1]->dinputs);
    printf("------------------------------\n");
    printf("------------LAYER 1-----------\n");
    printf("------------DWEIGHTS-------------\n");
    print_matrix(network->layers[0]->dweights);
    printf("------------DBIASES-----------\n");
    print_matrix(network->layers[0]->dbiases);
    printf("------------DINPUTS-----------\n");
    print_matrix(network->layers[0]->dinputs);
    printf("------------------------------\n");
    printf("#################################\n");

    // Free memory
    free_neural_network(network);
    printf("//////////////////////////////////////////////////////\n");
    printf("//////////// END TEST BACKWARDS_PASS_NN //////////////\n");
    printf("//////////////////////////////////////////////////////\n");  
}

void test_train_nn(){
    printf("//////////////////////////////////////////////////////\n");
    printf("///////////////// START TEST_TRAIN_NN ////////////////\n");
    printf("//////////////////////////////////////////////////////\n");
    // Create int array off defined constants
    int num_neurons_in_layers[NUM_LAYERS] = {NUM_NEURONS_LAYER_1, NUM_NEURONS_LAYER_2, NUM_NEURONS_LAYER_3};
    ActivationType activations[NUM_LAYERS] = {RELU, RELU, SOFTMAX};
    OptimizationType optimizations[NUM_LAYERS] = {SGD_MOMENTUM, SGD_MOMENTUM, SGD_MOMENTUM};
    bool regularizations[NUM_LAYERS] = {true, true, true};

    // Initialize layer based off defined constants
    NeuralNetwork* network = init_neural_network(NUM_LAYERS, num_neurons_in_layers, 
                                                LEARNING_RATE, activations, optimizations, regularizations, NUM_BATCH_FEATURES);

    // Check to see if layer was allocated properly
    if (network == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for test network in test_forward_pass_nn.\n");
        exit(1);
    } 

    // Initialize input data
    matrix inputs;
    inputs.dim1 = BATCH_SIZE;
    inputs.dim2 = NUM_BATCH_FEATURES;
    double input_data[9] = {5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7};  // Example: 3 samples, 3 features each
    inputs.data = input_data;

    // Initialize class target data
    matrix class_targets;
    class_targets.dim1 = BATCH_SIZE;
    class_targets.dim2 = NUM_BATCH_FEATURES;
    double class_target_data[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // 3 samples, 3 classes
    class_targets.data = class_target_data;

    // Check output
    // train_nn(network, &inputs,&class_targets);

    free_neural_network(network);
    printf("//////////////////////////////////////////////////////\n");
    printf("///////////////// END TEST_TRAIN_NN //////////////////\n");
    printf("//////////////////////////////////////////////////////\n");
}


//////////////////////////////////////////////////TEST ALL METHODS/////////////////////////////////////////////////////////////////

/*
Tests Every Method
*/
void test_all_methods() {
    test_matrix_transpose();
    test_matrix_mult();
    test_init_layer();
    test_forward_pass();
    test_backward_pass();
    test_accuracy();
    test_loss_categorical();
    test_update_params_sgd();
    test_init_neural_network();
    // Uneeded kind of overkill but useful maybe later for debugging.
    // test_forward_pass_nn();
    // test_backward_pass_nn();
    test_train_nn();
    
}