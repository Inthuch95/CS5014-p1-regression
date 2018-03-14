'''
Created on 14 Mar 2018

@author: it41
'''
from sklearn.linear_model import SGDRegressor
from cross_validation import get_scores, display_scores
from sklearn.externals import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.regression import mean_squared_error

print("Loading data")
X_train = np.loadtxt("../X_train.txt")
y_train = np.loadtxt("../y_train.txt")
X_test = np.loadtxt("../X_test.txt")
y_test = np.loadtxt("../y_test.txt")

# transform the data
print("Transforming data")
full_pipeline = Pipeline([
        ("std_scaler", StandardScaler())
    ])
X_train = full_pipeline.fit_transform(X_train)
X_test = full_pipeline.fit_transform(X_test)

print("Training with SGD ridge")
sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X_train, y_train)
joblib.dump(sgd_reg, "../sgd_reg.pkl")
scores = get_scores(sgd_reg, X_train, y_train)
rmse_scores = np.sqrt(-scores)
display_scores(rmse_scores)

print("")
predictions = sgd_reg.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print("Test set result")
print("MSE: ", mse)
print("RMSE: ", rmse)