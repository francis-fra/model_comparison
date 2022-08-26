import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def load_data_fashion_mnist(batch_size):
    train_data = '/home/fra/DataMart/images/fashion-mnist/fashion-mnist_train.csv'
    test_data = '/home/fra/DataMart/images/fashion-mnist/fashion-mnist_test.csv'
    
    train = pd.read_csv(train_data)
    train_tensor = torch.tensor(train.values, dtype=torch.float)
    X_train = train_tensor[:,:-1] / 255.
    y_train = train_tensor[:,0].type(torch.int64)

    test = pd.read_csv(test_data)
    test_tensor = torch.tensor(test.values, dtype=torch.float)
    X_test = test_tensor[:,:-1] / 255.
    y_test = test_tensor[:,0].type(torch.int64)
    
    dataset = TensorDataset(*(X_train, y_train))
    train_iter = DataLoader(dataset, batch_size, shuffle=True)
    dataset = TensorDataset(*(X_test, y_test))
    test_iter = DataLoader(dataset, batch_size, shuffle=False)
    return train_iter, test_iter

def accuracy(y_hat, y):  #@save
    """Compute the number of correct predictions.
        PARAMETERS
        ---------
        y_hat   : estimated probabilities for each class
        y       : integer, indices of labels
    """
    # find the hard decision
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def softmax(X):
    X_exp = torch.exp(X)
    # normalizer
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition 
 
def net(X, W, b):
    "single dense layer with softmax activation"
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def relu(X):
    mask = (X > 0).float()
    return X * mask

def deep_model(X, W1, W2, b1, b2):
    "two dense layers with softmax activation"
    x = torch.matmul(X.reshape((-1, W1.shape[0])), W1) + b1
    x = relu(x)
    x = torch.matmul(x.reshape((-1, W2.shape[0])), W2) + b2
    return softmax(x)

def cross_entropy(y_hat, y):
    """
        PARAMETERS
        ---------
        y_hat   : estimated probabilities for each class
        y       : integer, indices of labels
    """
    return -torch.log( y_hat[range(len(y_hat)), y] )

def sgd(params, lr, batch_size):
    "sgd in-place param update"
    # update param, not for propagation
    with torch.no_grad():
        for param in params:
            param[:] = param - lr * param.grad / batch_size
            param.grad.zero_()

def manual_run_deep_model(train_set, val_set, metadata):
    batch_size = metadata['batch_size']
    epochs = metadata['epochs']
    lr = metadata['lr']
    num_inputs = metadata['num_inputs']
    num_classes = metadata['num_classes']

    # single dense layer
    W1 = torch.normal(0, 0.01, (num_inputs, 64), dtype=torch.float, requires_grad=True)
    b1 = torch.zeros(64, dtype=torch.float, requires_grad=True)
    W2 = torch.normal(0, 0.01, (64, num_classes), dtype=torch.float, requires_grad=True)
    b2 = torch.zeros(num_classes, dtype=torch.float, requires_grad=True)

    # train
    for k in range(epochs):
        for X, y in train_set:
            y_hat = deep_model(X, W1, W2, b1, b2)
            l = cross_entropy(y_hat, y)
            l.sum().backward()
            sgd([W1, W2, b1, b2], lr, batch_size)
        print(f'epoch {k + 1}, loss {float(l.mean()):f}')

    evaluate_accuracy(deep_model, val_set, [W1, W2, b1, b2])

def manual_run(train_set, val_set, metadata):
    batch_size = metadata['batch_size']
    epochs = metadata['epochs']
    lr = metadata['lr']
    num_inputs = metadata['num_inputs']
    num_classes = metadata['num_classes']

    # single dense layer
    W = torch.normal(0, 0.01, (num_inputs, num_classes), dtype=torch.float, requires_grad=True)
    b = torch.zeros(num_classes, dtype=torch.float, requires_grad=True)
    # train
    for k in range(epochs):
        for X, y in train_set:
            y_hat = net(X, W, b)
            l = cross_entropy(y_hat, y)
            l.sum().backward()
            sgd([W, b], lr, batch_size)
        print(f'epoch {k + 1}, loss {float(l.mean()):f}')

    evaluate_accuracy(net, val_set, [W, b])

def auto_run(train_set, val_set, metadata):
    batch_size = metadata['batch_size']
    epochs = metadata['epochs']
    lr = metadata['lr']
    num_inputs = metadata['num_inputs']
    num_classes = metadata['num_classes']

    net = torch.nn.Sequential(
        torch.nn.Linear(num_inputs, 64),
        torch.nn.Linear(64, num_classes)
    )

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # train
    for k in range(epochs):
        for X, y in train_set:
            optimizer.zero_grad()
            y_hat = net(X)
            # l = loss(y_hat, y.reshape((-1, 1)))
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
        print(f'epoch {k + 1}, loss {float(l.mean()):f}')

    evaluate_accuracy(net, val_set)


def evaluate_accuracy(model, dataset, params=None):
    num_correct = 0
    total = 0
    for X, y in dataset:
        if params is not None:
            pred = model(X, *params)
        else:
            pred = model(X)
        # get hard decision
        _, predictions = torch.max(pred, 1)
        num_correct += int(sum(predictions == y))
        total += len(y)
    print(f"accuracy: {num_correct / total:.2f}")


if __name__ == '__main__':
    metadata = {
        'epochs': 10,
        'lr' : 1e-2,
        'batch_size': 256, 
        'num_classes': 10,
        'num_inputs': 784
    } 
    train_set, val_set = load_data_fashion_mnist(batch_size=metadata['batch_size'])
    # manual_run(train_set, val_set, metadata)
    manual_run_deep_model(train_set, val_set, metadata)
    # auto_run(train_set, val_set, metadata)
