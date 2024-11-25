from nn_bindings import *
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

def main():
    num_epochs = 30
    # load in training data
    x = pd.read_csv("../data/Spiral/train_data_1000.csv")
    X = numpy_to_matrix(x)

    y = pd.read_csv("../data/Spiral/train_labels_1000.csv")
    Y = numpy_to_matrix(y)

    # load in validate data
    x_pred = pd.read_csv("../data/Spiral/train_data.csv")
    X_Pred = numpy_to_matrix(x_pred)

    y_pred = pd.read_csv("../data/Spiral/train_labels.csv")
    Y_Pred = numpy_to_matrix(y_pred)

    num_layers = 3
    num_neurons_in_layer = [512, 512, 3]
    learning_rate = 0.05
    activations = [ActivationType.RELU, ActivationType.RELU, ActivationType.SOFTMAX]
    optimizations = [OptimizationType.ADAM, OptimizationType.ADAM, OptimizationType.ADAM]
    regularizations = [True, True]
    num_batch_features = 2

    model_ptr = create_model(num_layers, num_neurons_in_layer, activations, optimizations, regularizations,
                num_batch_features, learning_rate)
    
    model_ptr.contents.beta_1 = 0.85
    print(model_ptr.contents.beta_1)  # Should be 0.85

    model_ptr.contents.beta_2 = 0.9
    model_ptr.contents.epsilon = 1e-7
    model_ptr.contents.decay_rate = 5e-5

    model_ptr.contents.layers[0].contents.lambda_l1 = 5e-4
    model_ptr.contents.layers[1].contents.lambda_l1 = 5e-4

    model_ptr.contents.layers[0].contents.drop_out_rate = .1

    model_ptr.contents.layers[0].contents.clip_value = 0.0
    model_ptr.contents.useBiasCorrection = True
    model_ptr.contents.debug = True


    # Update Parameters

    # Print network info
    libnn.print_nn_info(model_ptr)

    # Train NN
    # libnn.train_nn(
    #     model_ptr,
    #     num_epochs,
    #     ctypes.byref(X),
    #     ctypes.byref(Y),
    #     ctypes.byref(X_Pred),
    #     ctypes.byref(Y_Pred)
    # )



    
if __name__ == "__main__":
    main()