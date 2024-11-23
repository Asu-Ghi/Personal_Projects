/*
Header File for a Layer Dense "Object"
Author: Asutosh Ghimire
*/

#include "utils.h"
//////////////////////////////////////////////////// DATA STRUCTURES ///////////////////////////////////////////////////////////////////////////

/*
Layer Dense data structure. 
*/
typedef struct {
    int id; // Integer id of layer
    int num_neurons; // Number of Neurons in a layer
    int num_inputs; // Number of Input Features into a layer

    matrix* weights; // Layer Weights
    matrix* biases; // Layer Biases

    matrix* inputs; // Inputs used for training
    matrix* pred_inputs; // Inputs used for predictions 

    matrix* pre_activation_output; // Outputs used for training (before activation)
    matrix* post_activation_output; // Outputs used for training (after activation)
    matrix* pred_outputs; // Outputs used for predictions
    
    matrix* dweights; // Gradients for weights
    matrix* dbiases; // Gradients for biases
    matrix* dinputs; // Gradients for inputs

    matrix* w_velocity; // Momentums for weights (ADAM, SGD_MOMENTUM)
    matrix* b_velocity; // Momentums for biases (ADAM, SGD_MOMENTUM)
    matrix* cache_weights; // Cache for weights (RMS_PROP, ADAGRAD)
    matrix* cache_bias; // Cache for biases (RMS_PROP, ADAGRAD)
    matrix* binary_mask; // Dropout mask (1 or 0), used to determine what neurons are dropped.
    double drop_out_rate; // % of neurons to drop from layer (0 by default)

    ActivationType activation; // Activation Function
    OptimizationType optimization; // Optimizer to use

    bool useRegularization; // Determines if using L1 and L2 regularization
    double lambda_l1;  // L1 regularization coefficient
    double lambda_l2;  // L2 regularization coefficient 
    double clip_value; // clip value for gradients

}layer_dense;

//////////////////////////////////////////////////// LAYER METHODS ///////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////// MISC METHODS ///////////////////////////////////////////////////////////////////////////

/*
Clips gradients to a min and max value, useful if experiencing exploding gradients. Applied in backwards.
*/
void clip_gradients(matrix* gradients, double clip_value);


/*
Drops the outputs for a given percentage of neurons in a layer. Helps with regularization.
*/
void apply_drop_out(layer_dense* layer, double drop_out_rate);


/*
Returns a layer dense object. Allocates memory on the heap for the object and its children.
*/
layer_dense* init_layer(int num_inputs, int num_neurons, ActivationType activation, OptimizationType optimization);

/*
Frees layer dense memory.
*/
void free_layer(layer_dense* layer);

//////////////////////////////////////////////////// ACCURACY METHODS ///////////////////////////////////////////////////////////////////////////

/*
Calculates the accuracy of the network while training.
*/
double calculate_accuracy(matrix* class_targets, layer_dense* final_layer, ClassLabelEncoding encoding);

/*
Calculates accuracy for predictions.
*/
double pred_calculate_accuracy(matrix* class_targets, layer_dense* final_layer, ClassLabelEncoding encoding);


//////////////////////////////////////////////////// LOSS METHODS ///////////////////////////////////////////////////////////////////////////

/*
Calculates the categorical cross entropy loss of the network while training.
*/
matrix* loss_categorical_cross_entropy(matrix* true_pred, layer_dense* last_layer, ClassLabelEncoding encoding);


/*
Calculates loss for predictions.
*/
matrix* pred_loss_categorical_cross_entropy(matrix* true_pred, layer_dense* last_layer, ClassLabelEncoding encoding);

/*
Calculates regularization loss l1 and l2 for a given layer.
*/
double calculate_regularization_loss(layer_dense* layer);

////////////////////////////////////////////////// FORWARD METHODS ///////////////////////////////////////////////////////////////////////////

/*
Performs a forward pass using a batch of inputs and a given layer (Training).
*/
void forward_pass(matrix* inputs, layer_dense* layer);

/*
Forward Pass for prediction inputs only.
*/
void pred_forward_pass(matrix* inputs, layer_dense* layer);

/*
Forward Pass for layers with ReLu
*/
matrix* forward_reLu(matrix* batch_input);

/*
Forward Pass for layers with softMax
*/
matrix* forward_softMax(matrix* batch_input);

/*
Forward Pass for layers with sigMoid
*/
matrix* forward_sigmoid(matrix* inputs);


////////////////////////////////////////////////// BACKWARD METHODS ///////////////////////////////////////////////////////////////////////////

/*
Backward pass for layers with ReLu
*/
void backward_reLu(matrix* input_gradients, layer_dense* layer);

/*
Backward pass for layers with SoftMax
*/
void backwards_softmax_and_loss(matrix* true_labels, layer_dense* layer);

/*
Backward pass for layers with Sigmoid
*/
void backwards_sigmoid(matrix* input_gradients, layer_dense* layer);

/*
Calculate relu gradients -> used in backward_relu
Has flags for parallelization using omp
*/
void calculate_relu_gradients(matrix* relu_gradients, layer_dense* layer);

/*
Calculate softmax gradients -> used in backward_softmax_loss
Has flags for parallelization using omp
*/
void calculate_softmax_gradients(matrix* softmax_gradients, layer_dense* layer, matrix* true_labels);

/*
Calculate sigmoid gradients -> used in backward_sigmoid
Has flags for parallelization using omp
*/
void calculate_sigmoid_gradients(matrix* sigmoid_gradients, layer_dense* layer);


/*
Calculate bias gradients -> used in backward_relu and backward_softmax
Has flags for parallelization using omp
*/
void calculate_bias_gradients(matrix* input_gradients, layer_dense* layer);

/*
Applies l1 and l2 regularization to gradients -> used in backward_relu and backward_softmax
Has flags for parallelization using omp
*/
void apply_regularization_gradients(layer_dense* layer);

/*
Calculates gradients for binary mask drop applied to outputs-> used in backward_relu and backward_softmax
Has flags for parallelization using omp
*/
void apply_dropout_gradients(layer_dense* layer);


////////////////////////////////////////////////// OPTIMIZER METHODS ///////////////////////////////////////////////////////////////////////////

/*
SGD Optimization
*/
void update_params_sgd(layer_dense* layer, double* learning_rate, int current_epoch, double decay_rate);

/*
SGD Optimization with momentum
*/
void update_params_sgd_momentum(layer_dense* layer, double* learning_rate, int current_epoch, double decay_rate, double beta);

/*
ADA GRAD Optimization
*/
void update_params_adagrad(layer_dense* layer, double* learning_rate, double decay_rate, double epsilon);

/*
RMSPROP Optimization
*/
void update_params_rmsprop(layer_dense* layer, double* learning_rate, double decay_rate, double epsilon);

/*
Adam Optimization
Adaptive Momentum.
For Batch Gradient Descent t = current epoch
For Mini Batch Gradient Descent t is incriminted after every mini batch.
Beta_1 and Beta_2 are hyperparameters affecting momentum and RMSProp caches respectively. 
*/
void update_params_adam (layer_dense* layer, double* learning_rate, double decay_rate, 
                    double beta_1, double beta_2, double epsilon, int t, bool correctBias);



/*
Call optimization for dense
*/
void optimization_dense(layer_dense* layer, double* lr, double lr_decay, int current_epoch, double beta1, double beta2,
                        double epsilon, bool useBiasCorrection);