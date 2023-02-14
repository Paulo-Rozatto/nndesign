import numpy as np

class Layer:
    def __init__(self, neurons: list):
        self.weights = np.array([n.get_weights() for n in neurons])
        self.biases = np.array([n.get_bias() for n in neurons])
        self.activations = [n.activation for n in neurons]
    
    def activate(self, inputs):
        inputs = np.array(inputs, dtype=np.float64)
        product = np.dot(self.weights, inputs.T) + self.biases
        result = []
        for i in range(len(self.activations)):
            result.append(self.activations[i](product[i]))

        return result