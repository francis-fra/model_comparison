
from sklearn.datasets import load_iris, load_digits, load_boston
import xgboost as xgb

num_workers = 8

digits = load_digits(n_class=2)
y = digits['target']
X = digits['data']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

xgb_model = xgb.XGBClassifier(n_jobs=num_workers, use_label_encoder=False)
xgb_model.fit(X_train, y_train)

import sys
mylib = '/home/fra/Project/pyProj/mlib'
sys.path.append(mylib)
import model as ml

xgb_eval = ml.MultiClassificationEvaluator(xgb_model)
results = xgb_eval.performance_summary(X_test, y_test)
print(results)
# {'Accuracy': 0.9722222222222222, 'Consfusion Matrix': array([[36,  0],
#        [ 2, 34]]), 'AUC': {0: 1.0, 1: 1.0}, 'GINI': {0: 1.0, 1: 1.0}}