'''
Created on 14 Mar 2018
'''
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.externals import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics.regression import mean_squared_error


print("Loading data")
X_train = np.loadtxt("../prepared_data/X_train.txt")
y_train = np.loadtxt("../prepared_data/y_train.txt")
X_test = np.loadtxt("../prepared_data/X_test.txt")
y_test = np.loadtxt("../prepared_data/y_test.txt")

# transform the data
print("Transforming data")
full_pipeline = Pipeline([
        ("poly_features", PolynomialFeatures(degree=2, include_bias=True))
    ])
X_train = full_pipeline.fit_transform(X_train)
X_test = full_pipeline.fit_transform(X_test)

print("Training with Ridge regression")
model = RidgeCV()
model.fit(X_train, y_train)
joblib.dump(model, "../models/poly_ridge_reg.pkl")

# get the error of training set
print("")
train_predictions = model.predict(X_train)
mse = mean_squared_error(y_train, train_predictions)
rmse = np.sqrt(mse)
print("Train set result")
print("MSE: ", mse)
print("RMSE: ", rmse)
print("R^2: ", model.score(X_train, y_train))

# get the error of testing set
print("")
test_predictions = model.predict(X_test)
mse = mean_squared_error(y_test, test_predictions)
rmse = np.sqrt(mse)
print("Test set result")
print("MSE: ", mse)
print("RMSE: ", rmse)
print("R^2: ", model.score(X_test, y_test))