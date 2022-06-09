import numpy as np
from layers import FullyConnected, Sigmoid, Convolutional, AveragePooling, Flattener


class LeNet:
    def __init__(self, n_output_classes):
        self.layers = []
        self.layers.append(Convolutional(1, 64, filter_size=5, stride=1, padding=2))
        self.layers.append(Sigmoid())
        self.layers.append(AveragePooling(pool_size=2, stride=2))
        self.layers.append(Convolutional(64, 192, filter_size=5, stride=1, padding=0))
        self.layers.append(Sigmoid())
        self.layers.append(AveragePooling(pool_size=2, stride=2))
        self.layers.append(Flattener())
        self.layers.append(FullyConnected(4800, 120))
        self.layers.append(Sigmoid())
        self.layers.append(FullyConnected(120, 84))
        self.layers.append(Sigmoid())
        self.layers.append(FullyConnected(84, n_output_classes))

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        pred = np.argmax(X, axis=1)
        return pred

    def params(self):
        result = {}
        for layer_num, layer in enumerate(self.layers):
            for param_name, param in layer.params().items():
                result[param_name + "_" + str(layer_num)] = param
        return result
