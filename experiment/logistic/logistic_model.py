import numpy as np


class logistic:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.w, self.b = self.init_v(data.shape[1])

    def sigmoid(self, Z):
        return 1/(1 + np.exp(-1 * Z))

    def init_v(self, dim):
        w = np.zeros([dim])
        b = np.random.rand(1)
        return w, b

    def infer(self, X):
        print(self.w.shape)
        print(X.T.shape)
        print(self.b.shape)
        Z = np.dot(self.w, X.T) + self.b
        A = self.sigmoid(Z)
        return A

    def propagate(self, w, b, X, Y):
        m = X.shape[0]
        A = self.infer(X)
        cost = (-1/m) * (np.dot(Y, np.exp(A)) + np.dot((1-Y), np.exp(1-A)))
        dw = (1/m) * np.dot((A-Y), X)
        db = (1/m) * np.sum(A-Y)
        return cost, dw, db

    def optimize(self, X, Y, number_iterations, learning_rate):
        costs = []
        for i in range(number_iterations):
            cost, dw, db = self.propagate(self.w, self.b, X, Y)
            self.w = self.w - learning_rate * dw
            self.b = self.b - learning_rate * db
            if i % 100 == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" % (i, cost))

    def predit(self, X):
        Y_pred = np.zeros(X.shape[0])
        A_list = self.infer(X)
        for i, v in enumerate(A_list):
            if v > 0.5:
                print(v)
                Y_pred[i] = 1
        return Y_pred

    def fit(self):
        self.optimize(self.data, self.label, number_iterations=100, learning_rate=0.01)
