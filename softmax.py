# Imports
import numpy as np

from accuracy import Loss_CategoricalCrossEntropy

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
    
    # Backward pass
    def backward(self, dvalues):

        # Create an uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients 
        for index, (single_output, single_dvalues) in \
                    enumerate(zip(self.output, dvalues)):

            # Flatten output array
            single_output = single_output.reshape(-1, 1)

            # Calculate the Jacobian Matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)

            # Calculate gradient sample-wise
            # Then add it to array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

# Softmax Classifier - combined softmax
# and cross-entropy loss for a faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Sets softmax and loss functions
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()
    
    # Forward pass
    def forward(self, inputs, y_true):

        # Output softmax activation layer
        self.activation.forward(inputs)

        # Set the output
        self.output = self.activation.output

        # Calculate cross-entropy loss and return it
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)

        # If labels are one hot-encoded
        # convert them to discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        # Create copy of derivative values
        self.dinputs = dvalues.copy()

        # Calculate the gradient 
        self.dinputs[range(samples), y_true] -= 1

        # Normalize the gradient
        self.dinputs = self.dinputs / samples