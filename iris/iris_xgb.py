from sklearn.datasets import load_iris, load_digits, load_boston
import xgboost as xgb

num_workers = 8

print("Iris: multiclass classification")
iris = load_iris()
y = iris['target']
X = iris['data']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

xgb_model = xgb.XGBClassifier(n_jobs=num_workers, use_label_encoder=False)
xgb_model.fit(X_train, y_train)

import pandas as pd
import sys
mylib = '/home/fra/Project/pyProj/mlib'
sys.path.append(mylib)
import model as ml

xgb_eval = ml.MultiClassificationEvaluator(xgb_model)
results = xgb_eval.performance_summary(X_test, y_test)
print(results)

# {'Accuracy': 0.9333333333333333, 
# 'Consfusion Matrix': array([[ 9,  0,  0],
#        [ 0,  6,  1],
#        [ 0,  1, 13]]), 
# 'AUC': {0: 1.0, 1: 0.9813664596273293, 2: 0.9910714285714286}, 
# 'GINI': {0: 1.0, 1: 0.9627329192546585, 2: 0.9821428571428572}}



# kf = KFold(n_splits=2, shuffle=True, random_state=rng)
# for train_index, test_index in kf.split(X):
#     xgb_model = xgb.XGBClassifier(n_jobs=num_workers).fit(X[train_index], y[train_index])
#     predictions = xgb_model.predict(X[test_index])
#     actuals = y[test_index]
#     print(confusion_matrix(actuals, predictions))