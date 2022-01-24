# Imports
import numpy as np
import layer as ly

# Features
# Batch of Inputs 
# A typical batch size may be 32
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# Create neural network layers
layer1 = ly.Layer_Dense(4, 5)
layer2 = ly.Layer_Dense(5, 2)

# Pass data forward to first layer
layer1.forward(X)

# Pass layer one to second layer
layer2.forward(layer1.output)

# Print layer two output
layer2.print_output()