import torch
import numpy as np
from sklearn.datasets import load_iris, load_digits, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch import nn, optim

def accuracy(y_prob, y_true):
    """
        y_prob: predicted prob for each class
        y_true: index of the true class (index base = 1)
    """
    num_correct = torch.sum(torch.argmax(y_prob, dim=1) == y_true).item()
    return num_correct, len(y_true)

def test01():
    lr = 0.1 
    num_epochs = 100
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

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    num_inputs = 4
    num_hiddens = 6
    num_outputs = 3
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)

    for k in range(num_epochs):
        for X, y in train_loader:
            optimizer.zero_grad()
            y_proba = net(X)
            loss = criterion(y_proba, y)
            loss.backward()
            optimizer.step()
        if k%20 == 0:
            num_correct = 0
            total = 0
            with torch.no_grad():
                for X, y in test_loader:
                    y_proba = net(X)
                    nc, nt = accuracy(y_proba, y)
                    num_correct += nc
                    total += nt
            acc = num_correct / total
            print("epochs {:3d}: {:.3f}".format(k, acc))

def test02():
    pass

def main():
    test01()

main()