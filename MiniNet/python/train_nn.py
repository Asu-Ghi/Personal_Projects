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
    learning_rate = np.float64(0.05)
    activations = [ActivationType.RELU, ActivationType.RELU, ActivationType.SOFTMAX]
    optimizations = [OptimizationType.ADAM, OptimizationType.ADAM, OptimizationType.ADAM]
    regularizations = [True, True]
    num_batch_features = 2

    model_ptr = create_model(num_layers, num_neurons_in_layer, activations, optimizations, regularizations,
                num_batch_features, learning_rate)
    
    model_ptr.contents.beta_1 = np.float64(0.85)

    model_ptr.contents.beta_2 = np.float64(0.9)
    model_ptr.contents.epsilon = np.float64(1e-7)
    model_ptr.contents.decay_rate = np.float64(5e-5)

    model_ptr.contents.layers[0].contents.lambda_l1 = np.float64(5e-4)
    model_ptr.contents.layers[0].contents.lambda_l2 = np.float64(5e-4)
    model_ptr.contents.layers[1].contents.lambda_l1 = np.float64(5e-4)
    model_ptr.contents.layers[1].contents.lambda_l2 = np.float64(5e-4)


    model_ptr.contents.layers[0].contents.drop_out_rate = .1

    model_ptr.contents.layers[0].contents.clip_value = 1
    model_ptr.contents.useBiasCorrection = True
    model_ptr.contents.debug = True

    # init neccesarry pointers
    val_acc = ctypes.c_double()  # Create a c_double object to store the value
    val_acc_ptr = ctypes.pointer(val_acc)  # Create a pointer to the c_double objec

    val_loss = ctypes.c_double()  # Create a c_double object to store the value
    val_loss_ptr = ctypes.pointer(val_loss)  # Create a pointer to the c_double objec
    
    for epoch in range(10000):
        # Update Network Params
        model_ptr.contents.current_epoch = epoch
        reg_loss = 0.0

        # Step 1: Forward pass
        libnn.forward_pass_nn(
            model_ptr, # Network 
            ctypes.byref(X) # Input Data
        )

        # Step 2: Calculate Accuracy
        accuracy = libnn.calculate_accuracy(
            ctypes.byref(Y), # Target Labels
            model_ptr.contents.layers[num_layers - 1], # Final Dense Layer
            ClassLabelEncoding.ONE_HOT # Lable encoding for Y   
        )

        # Step 3: Calculate Loss (returns matrix)
        loss = libnn.loss_categorical_cross_entropy(
            ctypes.byref(Y), # Target Labels
            model_ptr.contents.layers[num_layers - 1], # Final Dense Layer
            ClassLabelEncoding.ONE_HOT # Lable encoding for Y          
        )

        # Sum loss over batch
        loss = matrix_to_numpy(loss.contents)
        loss = np.sum(loss)

        # Step 4: Calculate Reg loss
        for i in range(len(regularizations)):
            if regularizations[i] == True:
                reg_loss += libnn.calculate_regularization_loss (
                    ctypes.byref(Y), # Target Labels
                    model_ptr.contents.layers[i], # Final Dense Layer
                    ClassLabelEncoding.ONE_HOT # Lable encoding for Y       
                )

        # Step 5: Backward pass
        libnn.backward_pass_nn(
            model_ptr, # Network
            ctypes.byref(Y) # Y predictions (labels)
        )

        # Step 6: Optimization
        libnn.update_parameters(
            model_ptr # Network
        )

        # Step 7: Validation
        if (epoch % 100 == 0):
            libnn.validate_model(
                model_ptr, # Network
                ctypes.byref(X_Pred), # Validate Inputs
                ctypes.byref(Y_Pred), # Validate Labels
                val_loss_ptr, # Validate Loss
                val_acc_ptr # Validate Accuracy  
            )

        # Step 9: Print and save info
        print(f"Epoch {epoch}: Total Model Loss: {loss + reg_loss:.4f} \
              (data_loss = {loss:.4f}, Reg Loss = {reg_loss:.4f}) \
                Model Accuracy: {accuracy:.4f}, LR: {model_ptr.contents.learning_rate:.6f}")
        
        print(f"Validate Loss: {val_loss.value:.4f}, Validate Accuracy: {val_acc.value:.4f}")

        # Step 10: Free memory
        libnn.free_layers_memory (
            model_ptr # Network
        )

    
if __name__ == "__main__":
    main()