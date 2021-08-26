from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
   
iris = load_iris()
iris.data[0:5], iris.target 

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    train_size=0.75, test_size=0.25)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

tpot = TPOTClassifier(verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

tpot.export('tpot_iris_pipeline.py')

import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

features = iris.data[0:5]
target = iris.target

training_features, testing_features, training_classes, testing_classes = \
            train_test_split(features, target, random_state=None)

exported_pipeline = make_pipeline(
    RBFSampler(gamma=0.8500000000000001),
    DecisionTreeClassifier(criterion="entropy", max_depth=3, min_samples_leaf=4, min_samples_split=9)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
print(results)