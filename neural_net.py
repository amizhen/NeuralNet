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
        self.layers = []  # [np.array, ....]
        self.consts = []
        self.activ_funcs = []
        self.cache = []
        pass

    def forward_prop(self, given, cache=False): # TODO Change to not use so much memory
        c = given ## ???
        for i in range(len(self.layers)):
            a = np.copy(c)
            c = c @ self.layers[i] + self.consts[i]
            z = np.copy(c)
            c = self.activ_funcs[i](c)
            if cache:
                self.cache.append((z, a))
        return c

    def init_params(self, shape, activ_funcs):
        # shape is of format [<num params>, <l1_size>, <l2_size>, ... <output_size>]
        table = {'lin': self.lin, 'relu': self.relu, 'sigmoid': self.sigmoid, 'tanh': self.tanh}
        for i in range(len(shape)-1):
            self.layers.append(np.random.randn(shape[i], shape[i+1])*0.01)
            self.consts.append(np.zeros((1, shape[i+1])))
            self.activ_funcs.append(table[activ_funcs[i]])

    def load_params(self, fd):
        pass


    def train(self, train_data, actual, step=0.00001, passes=100000):
        for i in range(passes):
            self.train_pass(train_data, actual, step=step)

    def train_pass(self, train_data, actual, step=0.00001):
        result = self.forward_prop(train_data)
        dA = - (np.divide(actual, result) - np.divide(1 - actual, 1 - result))

        self.cache = []
        for i in range(len(self.layers), 0, -1):
            dW, db, dA = self.one_deriv(i, dA)
            self.layers[i] *= [1-dW*step]
            self.consts[i] *= [1-db*step]


    def one_deriv(self, i, dA):
        table = {self.relu: self.relu_grad, self.sigmoid: self.sigmoid_grad} ## ADD TANH
        g = table[self.activ_funcs[i]]
        dZ = dA * g(self.cache[i][0])
        # d_const =
        m = self.cache[-1][1].shape[0]
        dW = 1/m * self.cache[i-1][1] @ dZ.T
        db = 1/m * dZ.sum(axis=0, keepdims=True)
        dA = dZ.T @ self.layers[i]
        return dW, db, dA


    def mse(self, actual, result):
        pass

    def cross_entropy(self, actual, result):
        # Y is number_examples X 1
        return -1/actual.shape[0]*np.sum(actual * np.log(result) + (1 - actual) * np.log(1 - result))

    def sigmoid(self, x):
        pass

    def sigmoid_grad(self, x):
        pass

    def relu(self, x):
        pass

    def relu_grad(self, x):
        pass

    def tanh(self):  # ALREADY DEFINED
        pass

    def lin(self, x):
        return x


if __name__ == '__main__':
    # t = np.array([[1,2,3,4]])
    # n = NN()
    # n.init_params([4, 10, 10, 3, 1], ['lin', 'lin', 'lin', 'lin'])
    # print(n.forward_prop(t))
    # print(np.array([[1,2],[3,4], [5,6], [7,8]]).shape)
    # n.layers = [np.array([[1,2],[3,4], [5,6], [7,8]]), np.array([[1,2],[3,4]])]
    # n.consts = [np.array([100, 1000]), np.array([100, 1000])]
    # n.activ_funcs = [n.lin, n.lin]
    # print(n.forward_prop(t))
    pass