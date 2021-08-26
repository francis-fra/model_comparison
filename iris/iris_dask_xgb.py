import xgboost as xgb

import numpy as np
from sklearn.datasets import load_iris, load_digits, load_boston

from dask.distributed import LocalCluster, Client
import multiprocessing
import dask.array as da

# must be in main to run client
if __name__ == '__main__':
    cores = multiprocessing.cpu_count()
    cluster = LocalCluster(n_workers=cores)
    # cluster = LocalCluster()
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

    # need to compute before evaluating
    pred = clf.predict_proba(X_test).compute()
    y_test = y_test.compute()

    from sklearn.metrics import roc_auc_score, accuracy_score
    results = roc_auc_score(y_test, pred, multi_class='ovr')
    print(results)
    # 1.0

    y_pred = clf.predict(X_test).compute()
    score = accuracy_score(y_test, y_pred)
    print(score)
    # 1.0

    client.shutdown()
    cluster.close()
