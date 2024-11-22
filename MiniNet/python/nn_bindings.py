import ctypes
import matplotlib.pyplot as plt
import numpy as np

# Load shared library
libnn = ctypes.CDLL('./libnn.so')  

########################################################## CTYPE STRUCTS ############################################################

# Label Types
class ClassLabelEncoding(ctypes.c_int):
    ONE_HOT = 0
    SPARSE = 1

# Activation Functions
class ActivationType(ctypes.c_int):
    RELU = 0
    SIGMOID = 1
    TANH = 2

# Optimization Functions
class OptimizationType(ctypes.c_int):
    SGD = 0
    ADAM = 1
    RMSPROP = 2

# Define matrix structure
class matrix(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("dim1", ctypes.POINTER(ctypes.c_int)),
        ("dim2", ctypes.POINTER(ctypes.c_int))
    ]

# Define the layer_dense structure 
class LayerDense(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int), # Integer id of layer
        ("num_neurons", ctypes.c_int), # Number of Neurons in a layer
        ("num_inputs", ctypes.c_int), # Number of Input Features into a layer


        ("weights", ctypes.POINTER(matrix)), # Layer Weights
        ("biases", ctypes.POINTER(matrix)), # Layer Biases

        ("inputs", ctypes.POINTER(matrix)), # Inputs used for training
        ("pred_inputs", ctypes.POINTER(matrix)), # Inputs used for predictions 

        ("pre_activation_output", ctypes.POINTER(matrix)), # Outputs used for training (before activation)
        ("post_activation_output", ctypes.POINTER(matrix)), # Outputs used for training (after activation)
        ("pred_outputs", ctypes.POINTER(matrix)), # Outputs used for predictions

        ("dweights", ctypes.POINTER(matrix)), # Gradients for weights
        ("dbiases", ctypes.POINTER(matrix)), # Gradients for biases
        ("dinputs", ctypes.POINTER(matrix)), # Gradients for inputs

        ("w_velocity", ctypes.POINTER(matrix)), # Momentums for weights (ADAM, SGD_MOMENTUM
        ("b_velocity", ctypes.POINTER(matrix)), # Momentums for biases (ADAM, SGD_MOMENTUM
        ("cache_weights", ctypes.POINTER(matrix)), # Cache for weights (RMS_PROP, ADAGRAD)
        ("cache_bias", ctypes.POINTER(matrix)), # Cache for biases (RMS_PROP, ADAGRAD)
        ("binary_mask", ctypes.POINTER(matrix)), # Dropout mask (1 or 0), used to determine what neurons are dropped.
        ("drop_out_rate", ctypes.c_double), # % of neurons to drop from layer (0 by default)

        ("activation", ActivationType), # Activation Function
        ("optimization", OptimizationType), # Optimizer to use

        ("useRegularization", ctypes.c_bool), # Determines if using L1 and L2 regularization
        ("lambda_l1", ctypes.c_double), # L1 regularization coefficient
        ("lambda_l2", ctypes.c_double), #  L2 regularization coefficient} 
        ("clip_value", ctypes.c_double), # clip value for gradients
    ]

# Define the NeuralNetwork structure
class NeuralNetwork(ctypes.Structure):
    _fields_ = [
        ("layers", ctypes.POINTER(ctypes.POINTER(LayerDense))),  # Pointer to array of pointers to LayerDense
        ("activations_per_layer", ctypes.POINTER(ActivationType)),  # Pointer to array of ActivationType
        ("optimizations_per_layer", ctypes.POINTER(OptimizationType)),  # Pointer to array of OptimizationType
        ("drop_out_per_layer", ctypes.POINTER(ctypes.c_double)),  # Pointer to array of dropout rates (double)
        ("regularizations_per_layer", ctypes.POINTER(ctypes.c_bool)),  # Pointer to array of bools
        ("num_neurons_in_layer", ctypes.POINTER(ctypes.c_int)),  # Pointer to array of ints (num neurons)
        ("num_layers", ctypes.c_int),  # Total number of layers
        ("batch_size", ctypes.c_int),  # Input batch size
        ("num_features", ctypes.c_int),  # Number of features in the input data
        ("learning_rate", ctypes.c_double),  # Optimization learning rate
        ("decay_rate", ctypes.c_double),  # Optimization learning decay rate
        ("num_epochs", ctypes.c_int),  # Total number of epochs
        ("current_epoch", ctypes.c_int),  # Current epoch iteration
        ("loss_history", ctypes.POINTER(ctypes.c_double)),  # Pointer to array of loss history
        ("momentum", ctypes.c_bool),  # Momentum flag
        ("beta_1", ctypes.c_double),  # Momentum hyperparameter
        ("beta_2", ctypes.c_double),  # Cache hyperparameter
        ("epsilon", ctypes.c_double),  # Optimization hyperparameter
        ("debug", ctypes.c_bool),  # Flag for debugging
        ("accuracy", ctypes.c_double),  # Network accuracy after each epoch
        ("loss", ctypes.c_double),  # Network loss after each epoch
        ("val_accuracy", ctypes.c_double),  # Validation accuracy after each epoch
        ("val_loss", ctypes.c_double),  # Validation loss after each epoch
        ("useBiasCorrection", ctypes.c_bool),  # Flag for using bias correction in ADAM
        ("early_stopping", ctypes.c_bool)  # Flag for early stopping based on val_loss
    ]

'''
Define argtypes for all C Methods

'''
######################################################## NETWORK METHODS ############################################################
# Init NN Methodd
libnn.init_neural_network.argtypes = [
    ctypes.c_int,  # num_layers
    ctypes.POINTER(ctypes.c_int),  # num_neurons_in_layer
    ctypes.c_double,  # learning_rate
    ctypes.POINTER(ActivationType),  # activations
    ctypes.POINTER(OptimizationType),  # optimizations
    ctypes.POINTER(ctypes.c_bool),  # regularizations
    ctypes.c_int  # num_batch_features
]
libnn.init_neural_network.restype = ctypes.POINTER(NeuralNetwork) # Return type for init nn is a pointer to nn struct


# Free Neural Network
libnn.free_neural_network.argtypes = [
    ctypes.POINTER(NeuralNetwork)
]
libnn.free_neural_network.restype = None # Returns Void


# Print Neural Net Info
libnn.print_nn_info.argtypes = [
    ctypes.POINTER(NeuralNetwork) # Network
]
libnn.print_nn_info.restype = None # Returns Void


# Forward Pass Neural Net
libnn.forward_pass_nn.argtypes = [
    ctypes.POINTER(NeuralNetwork), # Network 
    ctypes.POINTER(matrix) # Input Data
]
libnn.forward_pass_nn.restype = None # Returns Void


# Predictions Forward Pass
libnn.pred_forward_pass_nn.argtypes = [
    ctypes.POINTER(NeuralNetwork), # Network 
    ctypes.POINTER(matrix) # Validation Input Data
]
libnn.pred_forward_pass_nn.restype = None # Returns Void


# Backward Pass Neural Net
libnn.backward_pass_nn.argtypes = [
    ctypes.POINTER(NeuralNetwork), # Network
    ctypes.POINTER(matrix) # Y predictions (labels)
]
libnn.backward_pass_nn.restype = None # Returns Void


# Update Parameters Neural Net
libnn.update_parameters.argtypes = [
    ctypes.POINTER(NeuralNetwork), # Network
]
libnn.update_parametrs.restype = None # Returns void


# Train Neural Net Method
libnn.train_nn.argtypes = [
    ctypes.POINTER(NeuralNetwork), # Network
    ctypes.c_int, # Num Epochs
    ctypes.POINTER(matrix), # Training data
    ctypes.POINTER(matrix), # True Training Lables
    ctypes.POINTER(matrix), # Validate Data
    ctypes.POINTER(matrix) # Validate labels
]
libnn.train_nn.restype = None # Returns Void


# Predict Neural Network
libnn.predict.argtypes = [
    ctypes.POINTER(NeuralNetwork), # Network
    ctypes.POINTER(matrix) # Input data
]
libnn.predict.restype = None # Returns Void


# Validate Neural Network
libnn.validate_model.argtypes = [
    ctypes.POINTER(NeuralNetwork), # Network
    ctypes.POINTER(matrix), # Validate Inputs
    ctypes.POINTER(matrix), # Validate Labels
    ctypes.POINTER(ctypes.c_double), # Validate Loss
    ctypes.POINTER(ctypes.c_double) # Validate Accuracy
]
libnn.validate_model.restype = None # Returns Void

######################################################## LAYER METHODS ############################################################

# Dense Layer Clip Gradients
libnn.clip_gradients.argtypes = [
    ctypes.POINTER(matrix), # Gradients
    ctypes.c_double # Clip Values
]
libnn.clip_gradients.restype = None # Returns Void


# Dense Layer Apply Drop Out
libnn.apply_drop_out.argtypes = [
    ctypes.POINTER(LayerDense), # Dense Layer
    ctypes.c_double, # Drop out rate (% dropped)
]
libnn.apply_drop_out.restype = None # Retuns Void


# Dense Layer Init Layer
libnn.init_layer.argtypes = [
    ctypes.c_int, # Num inputs
    ctypes.c_int, # Num neurons
    ActivationType, # Activation function
    OptimizationType, # Optimization Function
]
libnn.init_layer.restype = None # Retuns Void


# Dense Layer Free Layer
libnn.free_layer.argtypes = [
    ctypes.POINTER(LayerDense) # Dense Layer
]
libnn.free_layer.restype = None # Retuns Void


# Dense Layer Calculate Accuracy
libnn.calculate_accuracy.argtypes = [
    ctypes.POINTER(matrix), # Target Labels
    ctypes.POINTER(LayerDense), # Final Dense Layer
    ClassLabelEncoding # Lable encoding
]
libnn.calculate_accuracy.restype = ctypes.c_double # Retuns Void


# Dense Layer Predictions Calculate Accuracy 
libnn.pred_calculate_accuracy.argtypes = [
    ctypes.POINTER(matrix), # Target Labels
    ctypes.POINTER(LayerDense), # Final Dense Layer
    ClassLabelEncoding # Lable encoding
]
libnn.pred_calculate_accuracy.restype = ctypes.c_double # Retuns Double


# Dense Layer Loss (Categorical Cross Entropy)
libnn.loss_categorical_cross_entropy.argtypes = [
    ctypes.POINTER(matrix), # Target Labels
    ctypes.POINTER(LayerDense), # Final Dense Layer
    ClassLabelEncoding # Lable encoding
]
libnn.loss_categorical_cross_entropy.restype = matrix # Retuns matrix


# Dense Layer Predictions Loss (Categorical Cross Entropy) 
libnn.pred_loss_categorical_cross_entropy.argtypes = [
    ctypes.POINTER(matrix), # Target Labels
    ctypes.POINTER(LayerDense), # Final Dense Layer
    ClassLabelEncoding # Lable encoding
]
libnn.pred_loss_categorical_cross_entropy.restype = matrix # Retuns matrix


# Dense Layer Calculate Regularization Loss 
libnn.calculate_regularization_loss.argtypes = [
    ctypes.POINTER(matrix), # Target Labels
    ctypes.POINTER(LayerDense), # Final Dense Layer
    ClassLabelEncoding # Lable encoding
]
libnn.calculate_regularization_loss.restype = ctypes.c_double # Retuns Double


# Dense Layer Forward Pass 
libnn.forward_pass.argtypes = [
    ctypes.POINTER(matrix), # Inputs (X)
    ctypes.POINTER(LayerDense), # Dense Layer
]
libnn.forward_pass.restype = None # Retuns Void


# Dense Layer Predictions Forward Pass 
libnn.pred_forward_pass.argtypes = [
    ctypes.POINTER(matrix), # Inputs (X)
    ctypes.POINTER(LayerDense), # Dense Layer
]
libnn.pred_forward_pass.restype = None # Retuns Void


# Dense Layer Forward ReLu 
libnn.forward_reLu.argtypes = [
    ctypes.POINTER(matrix) # Inputs (X)
]
libnn.forward_reLu.restype = None # Retuns Void


# Dense Layer Forward Softmax 
libnn.forward_softMax.argtypes = [
    ctypes.POINTER(matrix) # Inputs (X)
]
libnn.forward_softMax.restype = None # Retuns Void


# Dense Layer Backwards ReLu
libnn.backward_reLu.argtypes = [
    ctypes.POINTER(matrix), # Input Gradients 
    ctypes.POINTER(LayerDense) # Dense Layer
]
libnn.backward_reLu.restype = None # Retuns Void


# Dense Layer Backwards Softmax and Loss
libnn.backwards_softmax_and_loss.argtypes = [
    ctypes.POINTER(matrix), # True Labels 
    ctypes.POINTER(LayerDense) # Dense Layer
]
libnn.backwards_softmax_and_loss.restype = None # Retuns Void


# Dense Layer Calculate ReLu Gradients
libnn.calculate_relu_gradients.argtypes = [
    ctypes.POINTER(matrix), # ReLu Gradient Pointer 
    ctypes.POINTER(LayerDense) # Dense Layer
]
libnn.calculate_relu_gradients.restype = None # Retuns Void


# Dense Layer Calculate SoftMax Gradients
libnn.calculate_softmax_gradients.argtypes = [
    ctypes.POINTER(matrix), # SoftMax Gradient Pointer 
    ctypes.POINTER(LayerDense) # Dense Layer
]
libnn.calculate_softmax_gradients.restype = None # Retuns Void


# Dense Layer Calculate Bias Gradients
libnn.calculate_bias_gradients.argtypes = [
    ctypes.POINTER(matrix), # Input Gradient Pointer 
    ctypes.POINTER(LayerDense) # Dense Layer
]
libnn.calculate_bias_gradients.restype = None # Retuns Void


# Dense Apply Regularization Gradients
libnn.apply_regularization_gradients.argtypes = [
    ctypes.POINTER(LayerDense) # Dense Layer
]
libnn.apply_regularization_gradients.restype = None # Retuns Void


# Dense Apply Dropout Gradients
libnn.apply_dropout_gradients.argtypes = [
    ctypes.POINTER(LayerDense) # Dense Layer
]
libnn.apply_dropout_gradients.restype = None # Retuns Void


# Dense Layer Update Parameters SGD
libnn.update_params_sgd.argtypes = [
    ctypes.POINTER(LayerDense), # Dense Layer
    ctypes.POINTER(ctypes.c_double), # Network Learning Rate
    ctypes.c_int, # Network current epoch
    ctypes.c_double, # Learning Rate Decay
]
libnn.update_params_sgd.restype = None # Retuns Void


# Dense Layer Update Parameters SGD Momentum
libnn.update_params_sgd_momentum.argtypes = [
    ctypes.POINTER(LayerDense), # Dense Layer
    ctypes.POINTER(ctypes.c_double), # Network Learning Rate
    ctypes.c_int, # Network current epoch
    ctypes.c_double, # Learning Rate Decay
    ctypes.c_double # Beta 1
]
libnn.update_params_sgd_momentum.restype = None # Retuns Void


# Dense Layer Update Parameters ADAGRAD
libnn.update_params_adagrad.argtypes = [
    ctypes.POINTER(LayerDense), # Dense Layer
    ctypes.POINTER(ctypes.c_double), # Network Learning Rate
    ctypes.c_double, # Learning Rate Decay
    ctypes.c_double # Epsilon 
]
libnn.update_params_adagrad.restype = None # Retuns Void


# Dense Layer Update Parameters RMSPROP
libnn.update_params_rmsprop.argtypes = [
    ctypes.POINTER(LayerDense), # Dense Layer
    ctypes.POINTER(ctypes.c_double), # Network Learning Rate
    ctypes.c_double, # Learning Rate Decay
    ctypes.c_double # Epsilon 
]
libnn.update_params_rmsprop.restype = None # Retuns Void


# Dense Layer Update Parameters ADAM
libnn.update_params_adam.argtypes = [
    ctypes.POINTER(LayerDense), # Dense Layer
    ctypes.POINTER(ctypes.c_double), # Network Learning Rate
    ctypes.c_double, # Learning Rate Decay
    ctypes.c_double, # Beta 1
    ctypes.c_double, # Beta 2
    ctypes.c_double, # Epsilon 
    ctypes.c_int, # Current epoch
    ctypes.c_bool # UseBiasCorrection
]
libnn.update_params_adam.restype = None # Retuns Void


# Dense Layer Optimization Dense
libnn.optimization_dense.argtypes = [
    ctypes.POINTER(LayerDense), # Dense Layer
    ctypes.POINTER(ctypes.c_double), # Network Learning Rate
    ctypes.c_double, # Learning Rate Decay
    ctypes.c_int, # Current epoch
    ctypes.c_double, # Beta 1
    ctypes.c_double, # Beta 2
    ctypes.c_double, # Epsilon 
    ctypes.c_bool # UseBiasCorrection
]
libnn.optimization_dense.restype = None # Retuns Void


######################################################### UTIL METHODS #############################################################

# Util Methods Transpose Matrix
libnn.transpose_matrix.argtypes = [
    ctypes.POINTER(matrix) # Matrix w
]
libnn.transpose_matrix.restype = ctypes.POINTER(matrix) # Returns pointer to a matrix


# Util Methods Print Matrix
libnn.print_matrix.argtypes = [
    ctypes.POINTER(matrix) # Matrix w
]
libnn.print_matrix.restype = None # Returns Void


# Util Methods Matrix Multiplication
libnn.matrix_mult.argtypes = [
    ctypes.POINTER(matrix), # Matrix w
    ctypes.POINTER(matrix)  # Matrix v
]
libnn.matrix_mult.restype = ctypes.POINTER(matrix) # Returns pointer to a matrix


# Util Methods Element Wise Matrix Multiplication
libnn.element_matrix_mult.argtypes = [
    ctypes.POINTER(matrix), # Matrix w
    ctypes.POINTER(matrix)  # Matrix v
]
libnn.element_matrix_mult.restype = ctypes.POINTER(matrix) # Returns pointer to a matrix


# Util Methods Vector Dot Product
libnn.vector_dot_product.argtypes = [
    ctypes.POINTER(matrix), # Matrix w
    ctypes.POINTER(matrix)  # Matrix v
]
libnn.vector_dot_product.restype = ctypes.c_double # Returns dot product (double)


# Util Methods Matrix Scalar Multiplication
libnn.matrix_scalar_mult.argtypes = [
    ctypes.POINTER(matrix), # Matrix w
    ctypes.c_double # Scalar S
]
libnn.matrix_scalar_mult.restype = None # Returns Void
