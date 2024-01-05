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
        self.cost_func = self.mse

        pass

    def init_params(self, shape, activ_funcs, cost): # TODO add default init hyperparams
        # shape is of format [<num params>, <l1_size>, <l2_size>, ... <output_size>] Of size 1 + num layers + 1
        # activ_funcs is of format ['<activ0>', '<activ1>' ...] Of size num layers
        # cost is 'mse' etc
        activ_table = {'lin': self.lin, 'relu': self.relu, 'sigmoid': self.sigmoid, 'tanh': self.tanh, 'mse': self.mse}
        self.cost = activ_table[cost]

        for i in range(len(shape)-1):
            self.layers.append(np.random.randn(shape[i], shape[i+1])*0.01)
            self.consts.append(np.zeros((1, shape[i+1])))
            self.activ_funcs.append(activ_table[activ_funcs[i]])

    def load_params(self, fd):
        '''
        Loads weights from file (uses whatever default numpy read/writing behavior
        '''
        pass

    def forward_prop(self, given): # TODO Change to not use so much memory
        cache = []  # [(z_i, a_i), ...]
        a = given
        for layer, weight in enumerate(self.layers):
            const = self.consts[layer]
            activ = self.activ_funcs[layer]

            z = a @ weight + const
            cache.append((np.copy(z), np.copy(a)))
            a = activ(z)
            # print(layer)
            # print(1, z)
            # print(2, a)
            # print('\n\n')

        return a, cache

    def train(self, train_data, actual, alpha=0.1, passes=1000):
        for i in range(passes):
            self.train_pass(train_data, actual, alpha=alpha)

    def train_pass(self, train_data, actual, alpha):
        '''
        Computes one training pass
        '''
        pred, cache = self.forward_prop(train_data)
        cost_grad = self.grad_table[self.cost]

        da = cost_grad(pred, actual)
        # print('da: '+ str(da))
        # print('Cost: ' + str(self.cost(pred, actual)))
        for layer in range(len(self.layers)-1, -1, -1):
            dw, db, da = self.one_deriv(layer, da, cache[layer])
            # print(layer)
            # print('dw: ' + str(dw))
            # print('db: ' + str(db))
            # print('da: ' + str(da))
            self.layers[layer] -= alpha * dw
            self.consts[layer] -= alpha * db

    def one_deriv(self, i, da_prev, cache_i):
        '''
        Computes the gradients of the weights and consts for layer i
        '''
        g_prime = self.grad_table[self.activ_funcs[i]]
        dz = da_prev * g_prime(cache_i[0])
        da = dz @ self.layers[i].T
        dw = cache_i[1].T @ dz / dz.shape[0]
        db = np.sum(dz, axis=0) / dz.shape[0]
        return dw, db, da


    def mse(self, pred, actual):
        return 1/2*np.sum(np.square(pred - actual))/actual.shape[0]

    def mse_grad(self, pred, actual):
        return np.sum(pred - actual)/actual.shape[0]

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
        return np.ones(x.shape)

    def softmax(self, x):

    def softmax_grad(self, x):



def simple_test():
    nn = NN()
    nn.init_params([2, 1], ['lin'], 'mse')
    x = []
    y = []
    for i in range(1000):
        a = random.random()
        b = random.random()
        c = a + b
        x.append(([a, b]))
        y.append([c])

    x = np.array(x)
    y = np.array(y)
    nn.train(x, y)
    a, _ = nn.forward_prop(x)
    # print(a)
    # print(y)

    # print(a[0:10])
    # print(y[0:10])
    test = np.array([[0.1,0.2]])
    a, _ = nn.forward_prop(test)
    print(a)


def xor_test():
    nn = NN()
    nn.init_params([2, 4, 4, 1], ['sigmoid', 'sigmoid', 'sigmoid'], 'mse')
    x = []
    y = []
    for i in range(100):
        r = random.random()
        if r > 0.75:
            x.append([1,1])
            y.append([0])
        elif r > 0.5:
            x.append([1,0])
            y.append([1])
        elif r > 0.25:
            x.append([0,1])
            y.append([1])
        else:
            x.append([0,0])
            y.append([0])
    x = np.array(x)
    y = np.array(y)
    nn.train(x, y)
    t = np.array([[0,0],[1,0],[0,1],[1,1]])
    a, _ = nn.forward_prop(t)
    print(a)


if __name__ == '__main__':
    # simple_test()
    xor_test()
    pass