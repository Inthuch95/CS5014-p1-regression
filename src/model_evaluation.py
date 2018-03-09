'''
Created on 9 Mar 2018

@author: it41
'''
from cross_validation import display_scores, get_scores
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
import numpy as np

X_train = np.loadtxt("../X_train")
y_train = np.loadtxt("../y_train")
model = joblib.load("../normal_linreg.pkl")
predictions = model.predict(X_train)
scores = get_scores(model, X_train, y_train)
rmse_scores = np.sqrt(-scores)
display_scores(rmse_scores)
# for i, j in zip(predictions, y_train):
#     print("actual: " + str(j), "predicted: " + str(i))
# mse = mean_squared_error(y_train, predictions)
# rmse = np.sqrt(mse)
# print("MSE: ", mse)
# print("RMSE: ", rmse)