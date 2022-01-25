# Imports
import numpy as np

# Common Loss Class
class Loss:

    # Given the model output and ground truth values
    # Claculate the data and regularization losses
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate the mean loss over the samples
        data_loss = np.mean(sample_losses)

        # Return the mean loss
        return data_loss
    
# Cross-Entropy Loss
class Loss_CategoricalCrossEntropy(Loss):

    # Forward pass data
    def forward(self, y_pred, y_true):
        # Number of samples
        samples = len(y_pred)
        
        # Clip predictions as to avoid calculating log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values
        # Only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)

        # Return losses
        return negative_log_likelihoods

# Accuracy Calculation Class
class Accuracy:

    # Calculate accuracy from given predictions and expecations
    def calculate(self, y_pred, y_true):
        # Get models predictions
        predictions = np.argmax(y_pred, axis=1)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Accuray
        accuracy = np.mean(predictions == y_true)

        # Return accuracy
        return accuracy