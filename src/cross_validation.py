'''
Created on Mar 8, 2018

@author: User
'''
from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
def get_scores(model, X_train, y_train, scoring_val="neg_mean_squared_error", cv_val=10):
    scores = cross_val_score(model, X_train, y_train, scoring=scoring_val, cv=cv_val)
    return scores