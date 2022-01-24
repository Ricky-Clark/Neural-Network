# Network Layers
# Creating a layer
# Normalize Data to be between -1 to 1 if possible
# Initialize Weights - typically random num -0.1 to 0.1 
# Biases may be zero. If network is dead may need to increase your bias

# Imports
import numpy as np

# Set numpy random seed
np.random.seed(0)

# Layer Class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def print_weights(self):
        print(self.weights)
    
    def print_biases(self):
        print(self.biases)

    def print_output(self):
        print(self.output)