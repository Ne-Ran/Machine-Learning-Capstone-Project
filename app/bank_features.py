"""
Bank Marketing Feature Engineering Module.
Provides scikit-learn compatible transformer for custom feature engineering.
"""
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class BankFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Deterministic Feature Engineering for Bank Marketing dataset.
    
    Engineered features:
    1. pdays_contacted: Binary indicator (1 if pdays != -1, else 0).
       Captures whether the client was contacted in a previous campaign.
    2. balance_negative: Binary indicator (1 if balance < 0, else 0).
       Captures whether the client has an overdrawn account balance.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        # Stateless transformer: fitting requires no parameters
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
        else:
            raise TypeError("BankFeatureEngineering expects a pandas DataFrame.")
            
        if 'pdays' in X_out.columns:
            X_out['pdays_contacted'] = (X_out['pdays'] != -1).astype(int)
        else:
            raise KeyError("Column 'pdays' missing from input DataFrame.")
            
        if 'balance' in X_out.columns:
            X_out['balance_negative'] = (X_out['balance'] < 0).astype(int)
        else:
            raise KeyError("Column 'balance' missing from input DataFrame.")
            
        return X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        features = list(input_features)
        if 'pdays_contacted' not in features:
            features.append('pdays_contacted')
        if 'balance_negative' not in features:
            features.append('balance_negative')
        return np.array(features, dtype=object)
