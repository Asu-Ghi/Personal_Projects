#include "network.h"
#include "test_functions.h"

void backup(){
    // init layers
    layer_dense* layer_1 = init_layer(3, 5,  RELU, 4);
    layer_1->id = "1";
    layer_dense* layer_2 = init_layer(5, 10, RELU, 4);
    layer_2->id = "2";
    layer_dense* layer_3 = init_layer(10, 5, RELU, 4);
    layer_3->id = "3";
    layer_dense* layer_4 = init_layer(5, 3, SOFTMAX, 4);
    layer_4->id = "4";



    // Define a batch of inputs (example with 3 inputs and 4 examples in the batch)
    matrix batch;
    batch.dim1 = 4;  // Batch size
    batch.dim2 = 3;  // Number of input features (e.g., 3 inputs per example)
    batch.data = (double*)calloc(batch.dim1 * batch.dim2, sizeof(double));

    // Example input data for a batch of 4 examples, each with 3 features
    batch.data[0] = 1.0; batch.data[1] = 0.5; batch.data[2] = -0.3;  // First input
    batch.data[3] = 0.7; batch.data[4] = -0.4; batch.data[5] = 0.2;  // Second input
    batch.data[6] = -0.1; batch.data[7] = 0.3; batch.data[8] = 0.9;  // Third input
    batch.data[9] = 0.5; batch.data[10] = -0.6; batch.data[11] = 0.8; // Fourth input

    // batch after first forward pass
    forward_pass(&batch, layer_1);
    printf("-------------------LAYER 1------------------------\n");
    print_matrix(layer_1->post_activation_output);

    forward_pass(layer_1->post_activation_output, layer_2);
    printf("-------------------LAYER 2------------------------\n");
    print_matrix(layer_2->post_activation_output);

    forward_pass(layer_2->post_activation_output, layer_3);
    printf("-------------------LAYER 3------------------------\n");
    print_matrix(layer_3->post_activation_output);

    forward_pass(layer_3->post_activation_output, layer_4);
    printf("-------------------LAYER 4------------------------\n");
    print_matrix(layer_4->post_activation_output);

    // create true label vectors
    // size 4 x 3 
    matrix one_hot_vector;
    double data1[] = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0};
    one_hot_vector.data = data1;
    one_hot_vector.dim1 = 4;
    one_hot_vector.dim2 = 3;

    // size 4 x 1 
    matrix sparse_vector;
    double data2[] = {0.0, 2.0, 1.0, 1.0};
    sparse_vector.data = data2;
    sparse_vector.dim1 = 4;
    sparse_vector.dim2 = 1;

    // calculate loss from softmax
    matrix* losses_one_hot = loss_categorical_cross_entropy( &one_hot_vector, layer_4, ONE_HOT);
    matrix* losses_sparse = loss_categorical_cross_entropy(&sparse_vector, layer_4, SPARSE);

    // // calculate  and print accuracy of the batch
    // double accuracy_one_hot = calculate_accuracy(&one_hot_vector, layer_4, ONE_HOT);
    // double accuracy_sparse = calculate_accuracy(&sparse_vector, layer_4, SPARSE);
    // printf("------------ACCURACY-----------\n");
    // printf("One_Hot: %f\n", accuracy_one_hot);
    // printf("Sparse: %f\n", accuracy_sparse);
    // printf("------------LOSS-----------\n");
    // printf("One_Hot:\n");
    // printf("%f\n",losses_one_hot);
    // printf("SPARSE:\n");
    // printf("%f\n",losses_sparse);



    /*
    Backpropogation
    */

    // Step 1: Compute gradients 

    /*
    FIX THIS MEMORY OVERFLOW -> SOME BS STORE EVERYTHING IN THE STRUCT FROM NOW ON.
    */

    backwards_softmax_and_loss(&one_hot_vector, layer_4);  // Backpropagate from softmax to layer 3
    printf("grad_layer3 dim: %d x %d\n", layer_4->dinputs->dim1, layer_4->dinputs->dim2);

    backward_reLu(layer_4->dinputs, layer_3);  // Backpropagate from layer 3 to layer 2
    printf("grad_layer2 dim: %d x %d\n", layer_3->dinputs->dim1, layer_3->dinputs->dim2);

    backward_reLu(layer_3->dinputs, layer_2); // Backpropagate from layer 2 to layer 1
    printf("grad_layer1 dim: %d x %d\n", layer_2->dinputs->dim1, layer_2->dinputs->dim2);

    backward_reLu(layer_2->dinputs, layer_1); // Backpropagate from layer 1 to batch inputs
    printf("grad_layer1 dim: %d x %d\n", layer_1->dinputs->dim1, layer_1->dinputs->dim2);

    // free memory

    // free(losses_one_hot.data);
    // free(losses_sparse.data);

    // free layers last
    free_layer(layer_1);
    free_layer(layer_2);
    free_layer(layer_3);
    free_layer(layer_4);

}

#define NUM_FEATURES 4
#define NUM_CLASSES 3
/*
Main Method
*/
int main(int argc, char** argv) {
    // Check command-line arguments (same as before)
    // if (argc < 5) {
    //     printf("Command usage %s num_layers, batch_size, num_epochs, learning_rate", argv[0]);
    //     exit(1);
    // }

    // test_all_methods();
    test_backward_pass();
}
