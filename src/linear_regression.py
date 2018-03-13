'''
Created on Mar 8, 2018

@author: User
'''
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
def get_scores(model, X_train, y_train, scoring_val="neg_mean_squared_error", cv_val=10):
    scores = cross_val_score(model, X_train, y_train, scoring=scoring_val, cv=cv_val)
    return scores

X_train = np.loadtxt("../X_train.txt")
X_train = np.c_[np.ones_like(X_train), X_train]
y_train = np.loadtxt("../y_train.txt")

linreg = LinearRegression()
linreg.fit(X_train, y_train)
joblib.dump(linreg, "../normal_linreg.pkl")
scores = get_scores(linreg, X_train, y_train)
rmse_scores = np.sqrt(-scores)
display_scores(rmse_scores)