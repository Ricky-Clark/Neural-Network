# Imports
import math
import numpy as np

# Softmax Activation use Exponentation
# Exponentiation can lead to overflows
# Scalar subtract max value from layer outputs
# Exponentiate values to remove negatives
# Normalize values into a probably distribution
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize the probabilities for all samples
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        # Output probabilities
        self.output = probabilities