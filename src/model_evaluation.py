'''
Created on 9 Mar 2018

@author: it41
'''
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
import numpy as np

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
def get_scores(model, X_train, y_train, scoring_val="neg_mean_squared_error", cv_val=10):
    scores = cross_val_score(model, X_train, y_train, scoring=scoring_val, cv=cv_val)
    return scores

X_train = np.loadtxt("../X_train.txt")
y_train = np.loadtxt("../y_train.txt")
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