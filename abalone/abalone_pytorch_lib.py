
import pandas as pd
import sys
mylib = '/home/fra/Project/pyProj/mlib'
sys.path.append(mylib)

data_folder = "/home/fra/DataMart/datacentre/opendata/UCI/"
train_filename = 'abalone.csv'

# ------------------------------------------------------------
# load data
# ------------------------------------------------------------
df = pd.read_csv(data_folder + train_filename, skipinitialspace=True)
# encode target
mapping = {'M':0, 'F':1, 'I':2}

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.2)

import model as ml
import pipeline as pl

exclusions=[]
target_col = "SEX"

# ------------------------------------------------------------
# pipeline
# ------------------------------------------------------------
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
stdpl = pl.StandardPipeline(target_col, exclusions)
stdpl.make_pipeline(ntransformer=StandardScaler, ctransformer=OrdinalEncoder)

(X_train, y_train) = stdpl.fit_transform(df_train)
(X_test, y_test) = stdpl.transform(df_test)

# ------------------------------------------------------------
# pytorch
# ------------------------------------------------------------
libpath = "/home/fra/Project/DLProj/tchlib"
sys.path.append(libpath)

import classifiers as pcls
from torch import optim
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset

# reload(pcls)
# model = pcls.ClassifierNetwork(input_size, 2, [256, 128, 64])
input_size = X_train.shape[1]

model = pcls.ClassifierNetwork(input_size, 3, [64, 64])
optimizer = optim.SGD(model.parameters(), lr=0.05)
criterion = nn.CrossEntropyLoss()

train_data = list(zip(X_train, y_train))
test_data = list(zip(X_test, y_test))

# feed into a data loader
trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=100)
testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=100)

model = pcls.train_and_validate(model, trainloader, testloader, criterion, optimizer)

# 
tmp = torch.from_numpy(X_test)
x = tmp.view(-1, input_size)
out = model(x.float())

# prob output
out = out.detach().numpy()
proba = out

# ------------------------------------------------------------
# performance evaluation
# ------------------------------------------------------------
# TODO: automate this
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import auc
fpr = dict()
tpr = dict()
n_classes = 3

for idx in range(n_classes):
    fpr[idx], tpr[idx], _ = roc_curve(y_test, proba[:, idx], pos_label=idx)

result = dict()
for idx in range(n_classes):
    result[idx] = auc(fpr[idx], tpr[idx])
result
for k, v in result.items():
    result[k] = 2*v - 1

print(result)
# {0: 0.3720516802696361, 1: 0.774429917284128, 2: 0.33882350097880454}