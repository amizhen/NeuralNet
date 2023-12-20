import random

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
        self.layers = []  # [weight0, weight1 ...]
        self.consts = []  # [consts0, consts1 ...]
        self.activ_funcs = []  # [func, ...]
        self.grad_table = {self.lin: self.lin_grad, self.relu: self.relu_grad, self.sigmoid: self.sigmoid_grad,
                           self.tanh: self.tanh_grad, self.mse: self.mse_grad}
        pass

    def init_params(self, shape, activ_funcs): # TODO add default init hyperparams
        # shape is of format [<num params>, <l1_size>, <l2_size>, ... <output_size>]
        # activ_funcs is of format ['<activ0>', '<activ1>' ... '<loss>']
        activ_table = {'lin': self.lin, 'relu': self.relu, 'sigmoid': self.sigmoid, 'tanh': self.tanh, 'mse': self}
        for i in range(len(shape)-1):
            self.layers.append(np.random.randn(shape[i], shape[i+1])*0.01)
            self.consts.append(np.zeros((1, shape[i+1])))
            self.activ_funcs.append(activ_table[activ_funcs[i]])

    def load_params(self, fd):
        '''
        Loads weights from file (uses whatever default numpy read/writing behavior
        '''
        pass

    def forward_prop(self, given, do_cache=False): # TODO Change to not use so much memory
        cache = []  # [(z_i, a_i), ...]
        a = given
        for layer, weight in enumerate(self.layers):
            const = self.consts[layer]
            activ = self.activ_funcs[layer]

            z = a @ weight + const
            if do_cache:
                cache.append((np.copy(z), np.copy(a)))
            a = activ(z)

        return a, cache

    def train(self, train_data, actual, step=0.001, passes=100000):
        pass

    def train_pass(self, train_data, actual, step):
        pass

    def one_deriv(self, i, dA):
        pass

    def compute_cost(self, func, pred, actual): # TODO split cost/grad into 2 funcs
        cost = self.activ_funcs[func](pred, actual)
        grad = self.grad_table[func](pred, actual)
        return cost, grad


    def mse(self, pred, actual):
        return 1/2*np.sum(np.square(pred - actual))

    def mse_grad(self, pred, actual):
        return np.sum(pred, actual)

    def cross_entropy(self, pred, actual):
        # Cost Function
        return -1/actual.shape[0]*np.sum(actual * np.log(pred) + (1 - actual) * np.log(1 - pred))

    def cross_entropy_grad(self, pred):
        return

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_grad(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return np.vectorize(lambda x: 1 if x>0 else 0)(x)

    def tanh(self, x):  # ALREADY DEFINED
        return np.tanh(x)

    def tanh_grad(self, x):
        return 1 - np.tanh(x)**2

    def lin(self, x):
        return x

    def lin_grad(self, x):
        return 1


if __name__ == '__main__':
    pass