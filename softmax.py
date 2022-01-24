# Imports
import math
import numpy as np

# Softmax Activation use Exponentation
# Exponentiation can lead to overflows
# Scalar subtract max value from layer outputs
# Exponentiate values to remove negatives
# Normalize values into a probably distribution
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities