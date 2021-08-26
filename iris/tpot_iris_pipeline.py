import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.990909090909091
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=MLPClassifier(alpha=0.001, learning_rate_init=0.001)),
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=6, min_samples_leaf=18, min_samples_split=9)),
    RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.7000000000000001, min_samples_leaf=3, min_samples_split=10, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
