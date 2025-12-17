import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Apply log transformation for numeric variables
        """
        X = X.copy()
        for col in self.columns:
            X[col] = np.log(X[col] + 1)
        return X
