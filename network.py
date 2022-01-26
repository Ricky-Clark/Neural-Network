# Imports
from pickletools import optimize
import numpy as np
from layer import Layer_Dense
from activation import Activation_ReLU
from datapoints import spiral_data
from softmax import Activation_Softmax_Loss_CategoricalCrossentropy
from accuracy import Accuracy
from optimize import SGD_Optimizer

# Features
# Batch of Inputs 
# A typical batch size may be 32
# Create spiral dataset
X, y = spiral_data(100, 3)

# Create dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 64)

# Create ReLU activation for first dense layer
activation1 = Activation_ReLU()

# Create second dense layer with input features
# This is because we take the output of the previous
# layer. Then we will have 3 output values
dense2 = Layer_Dense(64, 3)

# Create softmax classifier with combined loss and softmax 
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create SGD optimizer
optimizer = SGD_Optimizer(decay=1e-3, momentum=0.9)

# Training loop
for epoch in range(10001):

    # Make a foward pass of training data through first layer
    dense1.forward(X)

    # Make a foward pass through the first ReLU activation function
    # It takes the output of the first dense layer
    activation1.forward(dense1.output)

    # Make a forward pass through the second dense layer
    # It takes the output of the first ReLU activation function
    dense2.forward(activation1.output)

    # Make a forward pass through the loss function 
    # which takes the output of the second dense layer
    loss = loss_activation.forward(dense2.output, y)

    # Calculate accuracy
    accuracy_function = Accuracy()
    accuracy = accuracy_function.calculate(loss_activation.output, y)

    # Prints epoch information every 100 hundred epoch
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')

    # Backward pass through network
    # Starting with loss/activation function
    loss_activation.backward(loss_activation.output, y)

    # Backward pass through second layer layer
    dense2.backward(loss_activation.dinputs)

    # Backward pass through first activation function
    activation1.backward(dense2.dinputs)

    # Backward pass through first dense layer
    dense1.backward(activation1.dinputs)

    # Optimize layers
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
