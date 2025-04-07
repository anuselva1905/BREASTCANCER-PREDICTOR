import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# Load the data
tpot_data = pd.read_csv('cleaned_data.csv')
features = tpot_data.drop('Classification', axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['Classification'], random_state=42, test_size=0.2)

# Define the pipeline
exported_pipeline = make_pipeline(
    FastICA(tol=0.05),
    StackingEstimator(estimator=GradientBoostingClassifier(
        learning_rate=0.1, max_depth=7, max_features=0.3,
        min_samples_leaf=14, min_samples_split=16, n_estimators=100, subsample=0.25
    )),
    StackingEstimator(estimator=DecisionTreeClassifier(
        criterion="entropy", max_depth=8, min_samples_leaf=11, min_samples_split=12
    )),
    StackingEstimator(estimator=DecisionTreeClassifier(
        criterion="entropy", max_depth=9, min_samples_leaf=14, min_samples_split=5
    )),
    GradientBoostingClassifier(
        learning_rate=0.01, max_depth=5, max_features=0.55,
        min_samples_leaf=1, min_samples_split=13, n_estimators=100, subsample=0.5
    )
)

# Set consistent random state
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

# Fit the model
exported_pipeline.fit(training_features, training_target)

# Predict and print accuracy
from sklearn.metrics import accuracy_score
results = exported_pipeline.predict(testing_features)
print(accuracy_score(testing_target, results))

# Save model to disk
import joblib
joblib.dump(exported_pipeline, 'model.pkl')
