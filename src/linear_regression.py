'''
Created on Mar 8, 2018

@author: User
'''
import numpy as np
from sklearn.linear_model import LinearRegression 
from cross_validation import display_scores, get_scores
from sklearn.externals import joblib

X_train = np.loadtxt("../X_train")
y_train = np.loadtxt("../y_train")

linreg = LinearRegression()
linreg.fit(X_train, y_train)
joblib.dump(linreg, "../normal_linreg.pkl")
# predictions = linreg.predict(X_train)
# scores = get_scores(linreg, X_train, y_train)
# rmse_scores = np.sqrt(-scores)
# display_scores(rmse_scores)
# for i, j in zip(predictions, y_train):
#     print("actual: " + str(j), "predicted: " + str(i))
# mse = mean_squared_error(y_train, predictions)
# rmse = np.sqrt(mse)
# print("MSE: ", mse)
# print("RMSE: ", rmse)