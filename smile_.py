import numpy as np
import scipy as sp

class SMiLE():
    """SMiLE algorithm for multi label with missing labels
    (Semi-supervised multi-label classification using imcomplete label information)
    

    Parameters
    ----------

    s : float, optional, default : 0.5
        Smoothness parameter for class imbalance
    
    alpha : float, optional, default : 0.35
        Smoothness assumption parameter, ensures similar instances
        having similar predicted output. This parameter balances the
        importance of the two terms of the equation to optimize
    
    k : int, optional, default : 5
        Neighbours parameter for clustering during the algorithm.
        It will indicate the number of clusters we want to create
        for the k nearest neighbor (kNN)

    Attributes
    ----------

    yCorrelation : array, [n_labels, n_labels]
        Correlation matrix between labels
    
    W : array, [n_samples, n_samples]
        Weighted matrix created by kNN for instances
    

    
    """

    def __init__(self, s=0.5, alpha=0.35, k=5):
        
        self.s = s
        self.alpha = alpha
        self.k = k

    def fit(self, X, y):
        """Fits the model

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances.
        y : array-like, shape=(n_samples, n_labels)
            Training labels.
        """
        return self

    def predict(self, X):
        """Predicts using the model

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Test instances.
        
        Returns:
        --------
        predictions : array-like, shape=(n_samples, n_labels)
            Label predictions for the test instances.
        """
        return predictions
