# Imports
import numpy as np

# Stochastic Gradient Descent Class
class SGD_Optimizer():

    # Initialize optimizer 
    # Set learning rate - default 1
    # Set learning decay - default 0
    # Set momentum - default 0
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    
    # Update optimizer learning
    def pre_update_params(self):

        # If using learning decay
        if self.decay:

            # Update current learning rate
            # with learning decay
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If using momentum
        if self.momentum:

            # If layer does not have
            # momentum arrays, create them
            if not hasattr(layer, 'weight_momentums'):

                # Initialize weight momentums
                # to zero filled array
                layer.weight_momentums = np.zeros_like(layer.weights)

                # If there are no weight
                # momentums array then there is no
                # bias momentums array
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum
            # This is done by taking the previous
            # updates multiplied with the retain
            # factor and updated with the gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates using the same
            # principal used in weight updates 
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentum = bias_updates

        # Else perform standard SGD optimizations
        else:

            # Set weight updates by
            # taking the learning rate
            # multiplied by the differentiated weights
            weight_updates = -self.current_learning_rate * layer.dweights

            # Set bias updates using
            # the same principle as the weights
            bias_updates = -self.current_learning_rate * layer.dbiases
        
        # Update weights and biases
        # using standard SGD or
        # SGD with momentum 
        layer.weights += weight_updates
        layer.biases += bias_updates
    
    # Update model iterations
    def post_update_params(self):
        self.iterations += 1
