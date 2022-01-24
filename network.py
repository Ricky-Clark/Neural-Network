# Imports
import numpy as np
from layer import Layer_Dense
from activation import Activation_ReLU
from datapoints import spiral_data
from softmax import Activation_Softmax

# Features
# Batch of Inputs 
# A typical batch size may be 32
X, y = spiral_data(100, 3)

# Create network layer and activation function
layer1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# Create second network layer and activation function
layer2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

# Pass data through network
layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output[:5])