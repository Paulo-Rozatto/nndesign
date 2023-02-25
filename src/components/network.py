import numpy as np

class Network:
    def __init__(self, layers: list):
        self._layers = layers

    def activate(self, input):
        output = None
        for layer in self._layers:
            output = layer.activate(input)
            input = output
        return output
        