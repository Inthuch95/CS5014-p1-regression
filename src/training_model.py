'''
Created on 16 Mar 2018
'''
from sklearn.pipeline import Pipeline
from sklearn.metrics.regression import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.externals import joblib
import numpy as np
import sys

regression_type = sys.argv[1]
regularization = sys.argv[2]

print("Loading data")
X_train = np.loadtxt("../prepared_data/X_train.txt")
y_train = np.loadtxt("../prepared_data/y_train.txt")
X_test = np.loadtxt("../prepared_data/X_test.txt")
y_test = np.loadtxt("../prepared_data/y_test.txt")

# add polynomial features if selected
if regression_type == "-l":
    X_train = np.c_[np.ones_like(X_train), X_train]
    X_test = np.c_[np.ones_like(X_test), X_test]
elif regression_type == "-p":
    # transform the data
    print("Transforming data")
    full_pipeline = Pipeline([
            ("poly_features", PolynomialFeatures(degree=2, include_bias=True))
        ])
    X_train = full_pipeline.fit_transform(X_train)
    X_test = full_pipeline.fit_transform(X_test)
else:
    print("Invalid parameter")
    quit()

# get the right algorithm    
if regularization == "-n":
    model = LinearRegression(n_jobs=-1)
elif regularization == "-r":
    model = RidgeCV(cv=10)
else:
    print("Invalid model")
    quit()

print("Training model")
model.fit(X_train, y_train)
if regression_type == "-l" and regularization == "-n":
    joblib.dump(model, "../models/linreg.pkl")
elif regression_type == "-l" and regularization == "-r":
    joblib.dump(model, "../models/linear_ridge.pkl")
elif regression_type == "-p" and regularization == "-n":
    joblib.dump(model, "../models/poly_reg.pkl")
elif regression_type == "-p" and regularization == "-r":
    joblib.dump(model, "../models/poly_ridge_reg.pkl")
else:
    print("Invalid parameter")
    quit()
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