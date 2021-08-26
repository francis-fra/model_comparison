from sklearn.datasets import load_iris, load_digits, load_boston
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, mean_squared_error

boston = load_boston()
y = boston['target']
X = boston['data']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X_train, y_train)

predictions = lgb_model.predict(X_test)

print(mean_squared_error(y_test, predictions))
# 9.97