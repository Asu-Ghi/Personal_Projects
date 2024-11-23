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
    batch_size = 64
    num_epochs = 100
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
    num_neurons_in_layer = [64, 128, 10]
    learning_rate = 0.01
    activations = [ActivationType.RELU, ActivationType.RELU, ActivationType.SIGMOID]
    optimizations = [OptimizationType.SGD, OptimizationType.ADAM, OptimizationType.RMSPROP]
    regularizations = [True, False, True]
    num_batch_features = 32

    model_ptr = create_model(num_layers, num_neurons_in_layer, activations, optimizations, regularizations,
                num_batch_features, learning_rate)
    
    model_ptr.num_epochs
    # Update Parameters

    # Print network info
    libnn.print_nn_info(model_ptr)

    # Train NN
    libnn.train_nn(
        model_ptr,
        num_epochs,
        ctypes.addressof(X),
        ctypes.addressof(Y),
        ctypes.addressof(X_Pred),
        ctypes.addressof(Y_Pred)
    )

    # 



    
if __name__ == "__main__":
    main()