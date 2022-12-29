import layers
import numpy as np


class LeNet:
    def __init__(self):
        self.conv1 = layers.Conv(1, 6, 5)
        self.relu1 = layers.ReLu()
        self.pool1 = layers.MaxPool((2, 2))
        self.conv2 = layers.Conv(6, 16, 5)
        self.relu2 = layers.ReLu()
        self.pool2 = layers.MaxPool((2, 2))
        self.fc1 = layers.Linear(16 * 5 * 5, 120)
        self.relu3 = layers.ReLu()
        self.fc2 = layers.Linear(120, 84)
        self.relu4 = layers.ReLu()
        self.fc3 = layers.Linear(84, 10)

    def forward(self, X):
        X = self.conv1.forward(X)
        X = self.relu1.forward(X)
        X = self.pool1.forward(X)
        X = self.conv2.forward(X)
        X = self.relu2.forward(X)
        X = self.pool2.forward(X)
        X = X.reshape(X.shape[0], -1)
        X = self.fc1.forward(X)
        X = self.relu3.forward(X)
        X = self.fc2.forward(X)
        X = self.relu4.forward(X)
        X = self.fc3.forward(X)
        return X

    def backward(self, grad):
        grad = self.fc3.backward(grad)
        grad = self.relu4.backward(grad)
        grad = self.fc2.backward(grad)
        grad = self.relu3.backward(grad)
        grad = self.fc1.backward(grad)
        grad = grad.reshape(grad.shape[0], 16, 5, 5)
        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)
        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)

    def get_params(self):
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.fc1.W, self.fc1.b, self.fc2.W, self.fc2.b, self.fc3.W, self.fc3.b]

    def set_params(self, params):
        self.conv1.weights = params[0]
        self.conv1.biases = params[1]
        self.conv2.weights = params[2]
        self.conv2.biases = params[3]
        self.fc1.W = params[4]
        self.fc1.b = params[5]
        self.fc2.W = params[6]
        self.fc2.b = params[7]
        self.fc3.W = params[8]
        self.fc3.b = params[9]