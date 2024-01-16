import numpy as np
from keras.datasets import mnist

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
                           self.tanh: self.tanh_grad, self.softmax:self.softmax_grad,
                           self.cross_entropy:self.cross_entropy_grad, self.mse: self.mse_grad}

        self.cost_func = lambda x, y: 0

        self.hyperparams = {'alpha':0.01, 'epochs':100000, 'batch_size':-1}
        pass

    def init_params(self, shape, activ_funcs, cost, **kwargs):  # TODO add default init hyperparams
        # shape is of format [<num params>, <l1_size>, <l2_size>, ... <output_size>] Of size 1 + num layers + 1
        # activ_funcs is of format ['<activ0>', '<activ1>' ...] Of size num layers
        # cost is 'mse' etc
        activ_table = {'lin': self.lin, 'relu': self.relu, 'sigmoid': self.sigmoid, 'tanh': self.tanh,
                       'softmax': self.softmax}
        cost_table = {'mse': self.mse, 'cross_entropy': self.cross_entropy}
        self.cost_func = cost_table[cost]

        for i in range(len(shape) - 1):
            self.layers.append(np.random.randn(shape[i], shape[i + 1]))
            self.consts.append(np.zeros((1, shape[i + 1])))
            self.activ_funcs.append(activ_table[activ_funcs[i]])

        for k in kwargs:
            self.hyperparams[k] = kwargs[k]
        #TODO : Create special last layer


    def load_params(self, fd):
        '''
        Loads weights from file (uses whatever default numpy read/writing behavior
        '''
        pass

    def forward_prop(self, given):  # TODO Change to not use so much memory
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

    def train(self, train_data, actual):
        passes = self.hyperparams['epochs']
        for i in range(passes):
            self.train_pass(train_data, actual)

    '''
    x=a_0 -> z0 -> a1 -> z1 -> a2 = y 
    '''
    def train_pass(self, train_data, actual):
        '''
        Computes one training pass

        '''
        alpha = self.hyperparams['alpha']
        pred, cache = self.forward_prop(train_data)
        cost_grad = self.grad_table[self.cost_func]
        da = cost_grad(pred, actual)


        steps = len(self.layers) - 1
        for layer in range(steps, -1, -1):
            dw, db, da = self.one_deriv(layer, da, cache[layer])

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
        return 1 / 2 * np.sum(np.square(pred - actual)) / actual.shape[0]

    def mse_grad(self, pred, actual):
        return pred - actual

    def cross_entropy(self, pred, actual):
        # Cost Function
        return -1 / actual.shape[0] * np.sum(actual * np.log(pred) + (1 - actual) * np.log(1 - pred))

    def cross_entropy_grad(self, pred):
        return

    def last_grad(self, x):
        return np.ones(x.shape)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_grad(self, x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return (x > 0) * 1

    def tanh(self, x):  # ALREADY DEFINED
        return np.tanh(x)

    def tanh_grad(self, x):
        return 1 - np.tanh(x) ** 2

    def lin(self, x):
        return x

    def lin_grad(self, x):
        return np.ones(x.shape)

    def softmax(self, x):
        tmp = np.exp(x)
        return tmp / np.sum(tmp)

    def softmax_grad(self, x):
        return x * (np.eye(x.shape)-x.T)



def xor_test():
    nn = NN()
    nn.init_params([2, 8, 8, 1], ['tanh', 'tanh', 'tanh'], 'mse')
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    a, _ = nn.forward_prop(x)
    print('Pre train:', a)
    print(nn.cost_func(a, y))
    nn.train(x, y)
    a, _ = nn.forward_prop(x)
    print('Post train:', a)
    print(nn.cost_func(a, y))


def mnist_test():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  ' + str(test_X.shape))
    print('Y_test:  ' + str(test_y.shape))
    train_X = train_X / 255.0
    test_X = test_X / 255.0

    img_size = train_X.shape[1] * train_X.shape[2]
    output_size = 10

    nn = NN()
    nn.init_params([img_size, 16, 16, output_size], ['tanh', 'tanh', 'softmax'],
                   'cross_entropy', alpha=0.01, epochs=10000)


def fixed_size_test():
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_grad(x):
        return sigmoid(x) * (1.0 - sigmoid(x))

    def mse(pred, actual):
        return 1 / 2 * np.sum(np.square(pred - actual)) / actual.shape[0]

    def mse_grad(pred, actual):
        return np.sum(pred - actual) / actual.shape[0]

    # x = np.array([[0.5, 0.5], [-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5]]).T
    # y = np.array([[0], [0], [1], [1]]).T
    x = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    # These are XOR outputs
    y = np.array([[0, 1, 1, 0]])
    # 2 16 1
    w1 = np.random.randn(16, 2)
    b1 = np.zeros((16, 1))
    # print(w1)
    w2 = np.random.randn(1, 16)
    b2 = np.zeros((1, 1))
    for _ in range(100000):
        z1 = np.dot(w1, x) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = sigmoid(z2)
        cost = mse(a2, y)

        # print(cost)
        dz2 = a2 - y
        dw2 = np.dot(dz2, a1.T) / 4
        db2 = np.sum(dz2, axis=1, keepdims=True)/4
        dz1 = np.dot(w2.T, dz2) * a1 * (1.0-a1)
        dw1 = np.dot(dz1, x.T) / 4
        db1 = np.sum(dz1, axis=1, keepdims=True) / 4

        alpha = 0.01
        w1 -= alpha * dw1
        w2 -= alpha * dw2
        b1 -= alpha * db1
        b2 -= alpha * db2

    print('Done Training')
    z1 = np.dot(w1, x) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    cost = mse(a2, y)


    print(a2, cost)


if __name__ == '__main__':
    np.random.seed(2)
    # xor_test()
    # fixed_size_test()
    # mnist_test()
    pass
