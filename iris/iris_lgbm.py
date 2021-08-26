from sklearn.datasets import load_iris, load_digits, load_boston
import xgboost as xgb
import lightgbm as lgb

num_workers = 8

print("Iris: multiclass classification")
iris = load_iris()
y = iris['target']
X = iris['data']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

import pandas as pd
import sys
mylib = '/home/fra/Project/pyProj/mlib'
sys.path.append(mylib)
import model as ml

lgb_eval = ml.MultiClassificationEvaluator(lgb_model)
results = lgb_eval.performance_summary(X_test, y_test)
print(results)
# {'Accuracy': 0.9333333333333333, 
# 'Consfusion Matrix': array([[ 8,  0,  0],
#        [ 0, 12,  0],
#        [ 0,  2,  8]]), 
# 'AUC': {0: 1.0, 1: 0.9675925925925926, 2: 0.9600000000000001}, 
# 'GINI': {0: 1.0, 1: 0.9351851851851851, 2: 0.9200000000000002}}