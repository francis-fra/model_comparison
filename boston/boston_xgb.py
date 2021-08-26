from sklearn.datasets import load_iris, load_digits, load_boston
import xgboost as xgb
from sklearn.metrics import confusion_matrix, mean_squared_error

num_workers = 8

boston = load_boston()
y = boston['target']
X = boston['data']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

xgb_model = xgb.XGBRegressor(n_jobs=num_workers)
xgb_model.fit(X_train, y_train)

predictions = xgb_model.predict(X_test)

print(mean_squared_error(y_test, predictions))
# 6.26