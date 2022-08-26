from sklearn.datasets import load_iris, load_digits, load_boston
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def model_evaluate(itr, params, loss):
    errors = []
    with torch.no_grad():
        for data, label in itr:
            y_pred = linear_model(data, params)
            err = loss(y_pred, label)
            errors.append(np.mean(err.numpy()))
    return errors


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
                torch.tensor(label, dtype=torch.float, requires_grad=True)


def init_param(num_features):
    W = torch.normal(0, 0.01, (num_features, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [W, b]

def linear_model(X, params):
    """
        X: torch tensor
    """
    W = params[0]
    b = params[1]
    return X@W + b

def sgd(params, batch_size, learning_rate=0.01):
    """
        params: model param
    """
    with torch.no_grad():
        for param in params:
            # slice replacement
            param[:] = param - learning_rate * param.grad / batch_size
            # reset grad to zero
            # param.grad.zero_()
    

def mserror(y_pred, y):
    """
        y_pred: torch tensor
        y: torch tensor
    """
    # return torch.mean((y_pred - y.reshape(y_pred.shape))**2)
    return (y_pred - y.reshape(y_pred.shape))**2 / 2


class LinearModel(object):
    def __init__(self, num_features):
        self.num_features = num_features
        self._init_param(num_features)

    def _init_param(self, num_features):
        W = torch.normal(0, 0.01, (num_features, 1), requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        self.params = [W, b]

    def compile(self, optimizer, loss, batch_size=32, num_epochs=20, learning_rate=0.01):
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = learning_rate

    def forward(self, X):
        W = self.params[0]
        b = self.params[1]
        return X@W + b

    def reset_grad(self):
        for param in self.params:
            param.grad.zero_()

    def fit(self, train_set, eval_set=None):
        X, y = train_set
        for k in range(self.num_epochs):
            train_itr = batch_data_generator(X, y, self.batch_size)
            for data, label in train_itr:
                y_pred = self.forward(data)
                err = self.loss(y_pred, label)
                err.sum().backward()
                self.optimizer(self.params, self.batch_size, self.lr)
                self.reset_grad()
            if k%5 == 0 and eval_set is not None:
                X_test, y_test = eval_set
                errors = self.evaluate(X_test, y_test)
                mse = np.mean(errors)
                # reporting
                print("epochs {:3d}: {:.3f}".format(k, mse))

    def evaluate(self, X, y):
        errors = []
        itr = batch_data_generator(X, y, self.batch_size)
        with torch.no_grad():
            for data, label in itr:
                y_pred = self.forward(data)
                err = self.loss(y_pred, label)
                errors.append(np.mean(err.numpy()))
        return errors


def test01():

    lr = 0.05
    num_epochs = 100
    batch_size = 32
    boston = load_boston()
    y = boston['target']
    X = boston['data']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # standardize
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    num_features = X_train.shape[1]
    params = init_param(num_features)
    loss = mserror
    for k in range(num_epochs):
        train_itr = batch_data_generator(X_train, y_train, batch_size)
        for data, label in train_itr:
            y_pred = linear_model(data, params)
            err = loss(y_pred, label)
            err.sum().backward()
            sgd(params, batch_size, lr)
        if k%5 == 0:
            test_itr = batch_data_generator(X_test, y_test, batch_size)
            errors = model_evaluate(test_itr, params, loss)
            mse = np.mean(errors)
            print("epochs {}: {}".format(k, mse))

def test02():
    lr = 0.1
    num_epochs = 100
    batch_size = 32
    boston = load_boston()
    y = boston['target']
    X = boston['data']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # standardize
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    num_features = X_train.shape[1]
    model = LinearModel(num_features)
    model.compile(sgd, mserror, batch_size, num_epochs, lr)
    model.fit((X_train, y_train), (X_test, y_test))

def main():
    # test01()
    test02()


main()
