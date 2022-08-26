import torch
import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def load_data():
    location = '/home/fra/DataMart/datacentre/opendata/mtcars.csv'
    df = pd.read_csv(location)
    target_col = 'mpg'
    # feature_col = ['hp', 'wt']
    feature_col = ['hp', 'wt', 'cyl']
    cols = feature_col + [target_col]
    return df[cols]

def data_iter(batch_size, X, y):
    """create an iterator to yield batched tensors
        PARAMETERS 
        ----------
        X, y: torch tensors

        RETURN
        ------
        batched X, y
    
    """
    num_examples = len(X)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(indices[i:min(i + batch_size, num_examples)])
        yield X[batch_indices], y[batch_indices]

def linear_regression(X, y):
    "sklearn regression"
    reg = LinearRegression().fit(X, y)
    print(f"R2 score: {reg.score(X, y)}")
    print(f"coeff: {reg.coef_}")
    print(f"intercept: {reg.intercept_}")

def linreg(X, w, b):
    """The linear regression model."""
    return torch.matmul(X, w) + b

def loss(y_hat, y):
    """MSE loss"""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

def auto_run(X, y, metadata):
    batch_size = metadata['batch_size']
    epochs = metadata['epochs']
    lr = metadata['lr']

    features = torch.as_tensor(X, dtype=torch.float)
    label = torch.as_tensor(y, dtype=torch.float)
    num_features = list(features.shape)[1]

    net = torch.nn.Sequential(
        torch.nn.Linear(num_features, 1)
    )
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # train
    for k in range(epochs):
        for X, y in data_iter(batch_size, features, label):
            optimizer.zero_grad()
            y_hat = net(X)
            # l = loss(y_hat, y.reshape((-1, 1)))
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

    with torch.no_grad():
        y_pred = net(features)
        score = r2_score(label, y_pred)
        print(f"R2 score: {score}")
        print(f"coeff: {net[0].weight}")
        print(f"intercept: {net[0].bias}")

def manual_run(X, y, metadata):
    batch_size = metadata['batch_size']
    epochs = metadata['epochs']
    lr = metadata['lr']

    features = torch.as_tensor(X, dtype=torch.float)
    label = torch.as_tensor(y, dtype=torch.float)

    num_features = list(features.shape)[1]
    # set params
    W = torch.normal(0, 0.01, (num_features, 1), dtype=torch.float, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float, requires_grad=True)
    # train
    for k in range(epochs):
        for X, y in data_iter(batch_size, features, label):
            y_hat = linreg(X, W, b)
            l = loss(y_hat, y)
            l.sum().backward()
            # sgd reset grad
            sgd([W, b], lr, batch_size)
        # print(f'epoch {k + 1}, loss {float(l.mean()):f}')

    # show coeff
    with torch.no_grad():
        y_pred = linreg(features, W, b)
        score = r2_score(label, y_pred)
        print(f"R2 score: {score}")
        print(f"coeff: {W}")
        print(f"intercept: {b}")

def sgd(params, lr, batch_size):
    "sgd in-place param update"
    # update param, not for propagation
    with torch.no_grad():
        for param in params:
            param[:] = param - lr * param.grad / batch_size
            param.grad.zero_()

def torch_regression(X, y):
    metadata = {
        'epochs': 100,
        'lr' : 1e-1,
        'batch_size': 16 
    } 
    manual_run(X, y, metadata)
    # auto_run(X, y, metadata)

def split_and_standardize(df, target_col):
    "standardize feature columns"
    y = df[target_col].values
    y = y.reshape(-1, 1)
    X = df.drop(target_col, axis=1).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

if __name__ == '__main__':
    df = load_data()
    X, y = split_and_standardize(df, 'mpg')
    # ok!
    linear_regression(X, y)
    # R2 score: 0.8267854518827914
    # coeff: [-2.14413603 -3.73453598]
    # intercept: 20.090625000000003
    torch_regression(X, y)
