'''
Created on Mar 8, 2018

@author: User
'''
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 
from cross_validation import get_scores, display_scores
from sklearn.externals import joblib
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
        ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
        ("std_scaler", StandardScaler())
    ])
X_train = full_pipeline.fit_transform(X_train)
X_test = full_pipeline.fit_transform(X_test)

print("Training with polynomial regression")
linreg = LinearRegression()
linreg.fit(X_train, y_train)
joblib.dump(linreg, "../normal_linreg.pkl")
scores = get_scores(linreg, X_train, y_train)
rmse_scores = np.sqrt(-scores)
display_scores(rmse_scores)

print("")
predictions = linreg.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print("Test set result")
print("MSE: ", mse)
print("RMSE: ", rmse)