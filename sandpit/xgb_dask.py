# https://xgboost.readthedocs.io/en/latest/tutorials/dask.html

import pickle
import xgboost as xgb

import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error, roc_auc_score 
from sklearn.datasets import load_iris, load_digits, load_boston

rng = np.random.RandomState(31337)
num_workers = 8


def dask_test01():
    from distributed import LocalCluster, Client
    import multiprocessing
    import dask.array as da

    cores = multiprocessing.cpu_count()
    cluster = LocalCluster(n_workers=cores)
    client = Client(cluster)

    # X, y = load_digits(n_class=2)
    digits = load_digits(n_class=2)
    y = digits['target']
    X = digits['data']
    # data needed to be in dask
    y = da.from_array(y)
    X = da.from_array(X)
    # use xgb dask
    clf = xgb.dask.DaskXGBClassifier(n_estimators=10, tree_method="hist")
    clf.client = client  # assign the client
    clf.fit(X, y, eval_set=[(X, y)])
    proba = clf.predict_proba(X).compute()
    # need to convert back to numpy for sklearn function
    score = roc_auc_score(y.compute(), proba[:,1])
    print(f"score = {score}") 

    client.shutdown()
    cluster.close()


if __name__ == '__main__':
    dask_test01()