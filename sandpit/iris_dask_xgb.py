# version problems...
import xgboost as xgb

import numpy as np
# from sklearn.model_selection import KFold, train_test_split, GridSearchCV
# from sklearn.metrics import confusion_matrix, mean_squared_error, roc_auc_score 
from sklearn.datasets import load_iris, load_digits, load_boston

from dask.distributed import LocalCluster, Client
# from distributed import LocalCluster, Client
import multiprocessing
import dask.array as da

cores = multiprocessing.cpu_count()
# cluster = LocalCluster(n_workers=cores)
cluster = LocalCluster()
client = Client(cluster)

data = load_iris()
y = data['target']
X = data['data']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# data needed to be in dask
y_train = da.from_array(y_train)
X_train = da.from_array(X_train)

y_test = da.from_array(y_test)
X_test = da.from_array(X_test)

# use xgb dask
clf = xgb.dask.DaskXGBClassifier()
clf.client = client  # assign the client
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])

import sys
mylib = '/home/fra/Project/pyProj/mlib'
sys.path.append(mylib)
import model as ml

xgb_eval = ml.MultiClassificationEvaluator(clf)
results = xgb_eval.performance_summary(X_test, y_test)
print(results)

# proba = clf.predict_proba(X).compute()
# # need to convert back to numpy for sklearn function
# score = roc_auc_score(y.compute(), proba[:,1])
# print(f"score = {score}") 

client.shutdown()
cluster.close()