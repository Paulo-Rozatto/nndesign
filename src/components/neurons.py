import numpy as np

class Neuron:
    def __init__(self, size, activation, bias = 0):
        self._weights = np.ones((1, size), dtype=np.float64)
        self.activation = activation
        self._bias = bias
    
    def activate(self, inputs):
        inputs = np.array(inputs, dtype=np.float64)
        return self.activation(np.dot(self._weights.T, inputs) + self._bias)
    
    def set_weights(self, w: list) -> None:
        self._weights = np.array(w, dtype=np.float64)
    
    def get_weights(self):
        return self._weights
    
    def get_bias(self):
        return self._bias
    
    def set_bias(self, x):
        self.bias = x

# TODO: use inheritance, wtf python why not having `extends` keyword?
class Delay:
    def __init__(self, size, activation, bias):
        self._weights = np.ones((1, size), dtype=np.float64)
        self.inputs = None
        self.activation = activation
        self.bias = bias
    
    def activate(self, inputs):
        inputs = np.array(inputs, dtype=np.float64)
        result = inputs
        if self.inputs is not None:
            result = self.activation(np.dot(self._weights, self.inputs.T) + self.bias)
        self.inputs = result
        return result
    
    def set_weights(self, w: list) -> None:
        self._weights = np.array(w, dtype=np.float64)
    
    def get_weights(self):
        return self._weights
    
    def get_bias(self):
        return self._bias
    
    def set_bias(self, x):
        self.bias = x