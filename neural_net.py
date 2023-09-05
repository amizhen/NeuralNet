import numpy as np


class NN:
    """
    Each row of the input matrix is a specific data point.
    Is of form: number_examples X number_parameters
    Weight matrix is of form: number_parameters X layer_size
    Product is of form: number_examples X layer_size
    Constant matrix is of form 1 X layer_size
    """
    def __init__(self):
        self.layers = None  # [np.array, ....]
        self.consts = None
        self.activ_funcs = None
        pass

    def forward_prop(self, given, cache=False):
        c = given
        for i in range(len(self.layers)):
            c = c @ self.layers[i] + self.consts[i]
            c = self.activ_funcs[i](c)
        return c

    def init_params(self, shape, activ_funcs):
        table = {'relu': self.relu, 'sigmoid': self.sigmoid(), 'tanh': self.tanh}
        for i in range(len(shape)-1):
            self.layers.append(np.random.randn(i, i+1)*0.01)
            self.consts.append(np.zeros((1, i+1)))
            self.activ_funcs.append(table[activ_funcs[i]])

    def load_params(self, fd):
        pass

    def train(self, train_set, step, iters):
        pass

    def cost(self, result, actual):
        pass

    def sigmoid(self):
        pass

    def relu(self):
        pass

    def tanh(self):  # ALREADY DEFINED
        pass
