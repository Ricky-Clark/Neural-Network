# Imports
import numpy as np

# Rectified Linear Function Class
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Calculate ouptut from given input
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we modify the original variable
        # Let's make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0