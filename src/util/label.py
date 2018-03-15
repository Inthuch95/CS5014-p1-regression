'''
Created on Mar 15, 2018
'''
from sklearn.base import TransformerMixin
from sklearn.preprocessing.label import LabelBinarizer

class MyLabelBinarizer(TransformerMixin):
    # make LabelBinarizer with 2 arguments (should replace this class with CategoricalEncoder in newer version of sklearn)
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)