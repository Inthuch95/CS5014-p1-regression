'''
Created on 14 Mar 2018

@author: it41
'''
from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
def get_scores(model, x, y, scoring_val="neg_mean_squared_error", cv_val=10):
    scores = cross_val_score(model, x, y, scoring=scoring_val, cv=cv_val)
    return scores