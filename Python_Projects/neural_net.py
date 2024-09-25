import numpy as np
# Classification Neural Network Built from Scratch
# 0 Frameworks 

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

# Softmax Activation Function
class Softmax:
    
    def forward(self, inputs):

        # Prevent large exp by subtracting off the max
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        
        # Normalize exp -> keep dimension as (n, 1);
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                             keepdims=True)
        
        self.output = probabilities

# Resuable Loss class
class Loss:

    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Get mean loss
        data_loss = np.mean(sample_losses)

        return data_loss

# Cross-Entropy Loss
class Categorical_Cross_Entropy(Loss):
    
    # Forward Pass
    def forward(self, y_pred, y_true):

        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        