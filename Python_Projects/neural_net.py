import numpy as np

# Dense Layer Class
class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        # Init weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Calculate output from inputs, weights and biases
        self.output = np.dot(self.weights, inputs) + self.biases;
    

# ReLu Activation Function
class ReLU:
    
    # Forward Pass
    def forward(self, inputs):
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)
