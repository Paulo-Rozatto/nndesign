import numpy as np

class Layer:
    def __init__(self, neurons = None, weights = None, biases = None, activations = None, multipleActivations = False):
        if neurons is None:
            self._weights = weights
            self._biases = biases
            self._multipleActivations = multipleActivations
            self._activation = activations
        else:
            self._weights = np.array([n.get_weights() for n in neurons])
            self._biases = np.array([n.get_bias() for n in neurons])
            self._activation = [n.activation for n in neurons]
            self._multipleActivations = True
            
    def activate(self, inputs):
        inputs = np.array(inputs, dtype=np.float64)
        product = np.dot(self._weights, inputs) + self._biases
        result = []

        if self._multipleActivations:
            for i in range(len(self._activation)):
                result.append(self._activation[i](product[i]))
        else:
            for prod in product: 
                result.append(self._activation(prod[0]))

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
        self._activation = activations
    
    def get_activations(self):
        return self._activation