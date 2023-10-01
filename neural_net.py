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
        pass

    def forward_prop(self, given, cache=False):
        c = given ## ???
        for i in range(len(self.layers)):
            c = c @ self.layers[i]
            c = c + self.consts[i]
            c = self.activ_funcs[i](c)
        return c

    def init_params(self, shape, activ_funcs):
        # shape is of format [<num params> <l1_size> <l2_size> ... <output_size>]
        table = {'lin': self.lin, 'relu': self.relu, 'sigmoid': self.sigmoid, 'tanh': self.tanh}
        for i in range(len(shape)-1):
            self.layers.append(np.random.randn(shape[i], shape[i+1])*0.01)
            self.consts.append(np.zeros((1, shape[i+1])))
            self.activ_funcs.append(table[activ_funcs[i]])

    def load_params(self, fd):
        pass

    def train(self, train_set, step, iters):
        pass

    # def one_deriv(self, i, cache):
    #     table = {self.relu: self.relu_grad, self.sigmoid: self.sigmoid_grad} ## ADD TANH
    #     g = table[self.activ_funcs[i]]
    #     d_const =

    def cross_entropy(self, result, actual):
        # Y is number_examples X 1
        return -1/result.shape[0]*np.sum(result*np.log(actual) + (1-result)*np.log(1-actual))

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