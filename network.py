# Imports
import numpy as np
from layer import Layer_Dense
from activation import Activation_ReLU
from datapoints import spiral_data
from softmax import Activation_Softmax
from loss import Loss, Loss_CategoricalCrossEntropy, Accuracy

# Features
# Batch of Inputs 
# A typical batch size may be 32
# Create spiral dataset
X, y = spiral_data(100, 3)

# Create dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation for first dense layer
activation1 = Activation_ReLU()

# Create second dense layer with input features
# This is because we take the output of the previous
# layer. Then we will have 3 output values
dense2 = Layer_Dense(3, 3)

# Create ReLU activation for second dense layer
activation2 = Activation_Softmax()

# Make a foward pass of training data through first layer
dense1.forward(X)

# Make a foward pass through the first ReLU activation function
# It takes the output of the first dense layer
activation1.forward(dense1.output)

# Make a forward pass through the second dense layer
# It takes the output of the first ReLU activation function
dense2.forward(activation1.output)

# Make a forward pass through the second ReLU activation function
# It takes the output of the second dense layer
activation2.forward(dense2.output)

# Print the data of the first few samples
print(activation2.output[:5])

# Perform a forward pass through loss
loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

# Forward pass data through accuracy
accuracy_function = Accuracy()
accuracy = accuracy_function.calculate(activation2.output, y)

# Print loss and accuracy
print('Loss: ', loss)
print('Accuracy: ', accuracy)