from nn_bindings import *


# int, int, double, ENUM, ENUM, bool, int
def create_model(num_layers, num_neurons_in_layer, activations, optimizations,
                    regularizations, num_batch_features, learning_rate):
    
    # Convert to ctypes
    num_neurons_in_layer_array = (ctypes.c_int * num_layers)(*num_neurons_in_layer)
    activations_array = (ActivationType * num_layers)(*activations)
    optimizations_array = (OptimizationType * num_layers)(*optimizations)
    regularizations_array = (ctypes.c_bool * num_layers)(*regularizations)

    # Call init_neural_network
    model_ptr = libnn.init_neural_network (
        num_layers,
        num_neurons_in_layer_array,
        learning_rate,
        activations_array,
        optimizations_array,
        regularizations_array,
        num_batch_features
    )

    return model_ptr
    ctypes.POINTER(NeuralNetwork), # Network
    ctypes.c_int, # Num Epochs
    ctypes.POINTER(matrix), # Training data
    ctypes.POINTER(matrix), # True Training Lables
    ctypes.POINTER(matrix), # Validate Data
    ctypes.POINTER(matrix) # Validate labels


def main():
    batch_size = 64
    num_epochs = 100

    num_layers = 3
    num_neurons_in_layer = [64, 128, 10]
    learning_rate = 0.01
    activations = [ActivationType.RELU, ActivationType.RELU, ActivationType.SIGMOID]
    optimizations = [OptimizationType.SGD, OptimizationType.ADAM, OptimizationType.RMSPROP]
    regularizations = [True, False, True]
    num_batch_features = 32

    model_ptr = create_model(num_layers, num_neurons_in_layer, activations, optimizations, regularizations,
                num_batch_features, learning_rate)
    
    libnn.train_nn(model_ptr

        
    )


    
if __name__ == "__main__":
    main()