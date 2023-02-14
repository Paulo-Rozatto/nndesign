import numpy as np

class Layer:
    def __init__(self, neurons: list):
        self._weights = np.array([n.get_weights() for n in neurons])
        self._biases = np.array([n.get_bias() for n in neurons])
        self._activations = [n.activation for n in neurons]
    
    def activate(self, inputs):
        inputs = np.array(inputs, dtype=np.float64)
        product = np.dot(self._weights, inputs.T) + self._biases
        result = []
        for i in range(len(self._activations)):
            result.append(self._activations[i](product[i]))

        return result
    
    def set_weights(self, w: list) -> None:
        self._weights = np.array(w, dtype=np.float64)
    
    def get_weights(self):
        return self._weights
    
    def set_biases(self, biases: list):
        self._biases = np.array(biases, dtype=np.float64)
    
    def get_biases(self):
        return self._biases

    def set_activations(self, activations: list):
        self._activations = activations
    
    def get_activations(self):
        return self._activations