import torch
import numpy as np
from sklearn.datasets import load_iris, load_digits, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def CategoricalCrossEntropy(y_proba, y_true):
    "equivalent to nn.CrossEntropyLoss"
    # convert to prob using softmax
    prob = torch.exp(y_proba) 
    p = prob / prob.sum(keepdim=True, dim=1)
    # cross entropy
    p = p[range(len(p)), y_true]
    return torch.mean(- torch.log(p))

class Classifier(object):
    def __init__(self, weights):
        self.weights = weights
        self._init_param(weights)

    def _init_param(self, weights):
        params = []
        for in_weights, out_weights in zip(weights[:-1], weights[1:]):
            W = torch.normal(0, 0.01, size=(in_weights, out_weights), requires_grad=True)
            b = torch.zeros(out_weights, requires_grad=True)
            params.append(W)
            params.append(b)
        self.params = params

    def compile(self, optimizer, loss, batch_size=32, num_epochs=20, learning_rate=0.01):
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = learning_rate

    def forward(self, X, activation):
        Ws = self.params[0::2]
        bs = self.params[1::2]
        x = X.reshape((-1, self.weights[0]))
        # excluding input
        num_layers = len(self.weights) - 1
        layer_number = 1
        # no activation at the end
        for W, b in zip(Ws, bs):
            if layer_number < num_layers:
                x = activation(torch.matmul(x, W) + b)
            else:
                x = torch.matmul(x, W) + b
            layer_number += 1
        return x

    def reset_grad(self):
        for param in self.params:
            param.grad.zero_()

    def fit(self, train_set, eval_set=None):
        X, y = train_set
        for k in range(self.num_epochs):
            train_itr = batch_data_generator(X, y, self.batch_size)
            for data, label in train_itr:
                y_pred = self.forward(data, relu)
                err = self.loss(y_pred, label)
                err.mean().backward()
                self.optimizer(self.params, self.batch_size, self.lr)
                self.reset_grad()
            if k%50 == 0 and eval_set is not None:
                X_test, y_test = eval_set
                acc = self.evaluate(X_test, y_test)
                # errors = np.mean(errors)
                # # reporting
                print("epochs {:3d}: {:.3f}".format(k, acc))

    def evaluate(self, X, y):
        "out of sample testing"
        num_correct = 0
        total = 0
        itr = batch_data_generator(X, y, self.batch_size)
        with torch.no_grad():
            for data, label in itr:
                y_pred = self.forward(data, relu)
                nc, nt = accuracy(y_pred, label)
                num_correct += nc
                total += nt
        return num_correct / total


def batch_data_generator(X, y, batch_size):
    """
        X           : numpy array
        y           : numpy array
        batch_size  : int
        RETURN      : torch tensor
    """
    num_samples, num_features = X.shape
    batch_size = min(num_samples, batch_size)
    idx = list(range(num_samples))
    random.shuffle(idx)
    for k in range(0, num_samples, batch_size):
        indices = idx[k:min(k+batch_size, num_samples)]
        samples = X[indices]
        label = y[indices]
        yield torch.tensor(samples, dtype=torch.float, requires_grad=True), \
                torch.tensor(label, dtype=torch.long)
                # torch.tensor(label, dtype=torch.long, requires_grad=True)

def sgd(params, batch_size, learning_rate=0.01):
    """
        params: model param
    """
    with torch.no_grad():
        for param in params:
            # slice replacement
            param[:] = param - learning_rate * param.grad / batch_size
            # print(param.grad)

def accuracy(y_prob, y_true):
    """
        y_prob: predicted prob for each class
        y_true: index of the true class (index base = 1)
    """
    num_correct = torch.sum(torch.argmax(y_prob, dim=1) == y_true).item()
    return num_correct, len(y_true)

def test01():
    lr = 0.5 
    num_epochs = 600
    batch_size = 32

    iris = load_iris()
    y = iris['target']
    X = iris['data']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # standardize
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # weights = [4, 12, 6, 3]
    weights = [4, 6, 3]
    model = Classifier(weights)
    loss = CategoricalCrossEntropy
    # loss = torch.nn.CrossEntropyLoss()
    model.compile(sgd, loss, batch_size, num_epochs, lr)
    model.fit((X_train, y_train), (X_test, y_test))

def test02():
    lr = 2.5 
    num_epochs = 2000
    batch_size = 32

    iris = load_iris()
    y = iris['target']
    X = iris['data']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # standardize
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    weights = [4, 2, 2, 3]
    model = Classifier(weights)
    loss = CategoricalCrossEntropy
    model.compile(sgd, loss, batch_size, num_epochs, lr)
    model.fit((X_train, y_train), (X_test, y_test))

def main():
    test02()
    # test01()

main()