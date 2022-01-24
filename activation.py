# Imports
import numpy as np

# Rectified Linear Function Class
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)