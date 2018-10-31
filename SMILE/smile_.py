import numpy as np
from sklearn.cluster import KMeans


class SMiLE:
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

    y_corr : array, [n_labels, n_labels]
        Correlation matrix between labels
    
    W : array, [n_samples, n_samples]
        Weighted matrix created by kNN for instances
    

    
    """

    def __init__(self, s=0.5, alpha=0.35, k=5):
        """Initialize properties.

        :param s:
        :param alpha:
        :param k:
        """
        self.s = s
        self.alpha = alpha
        self.k = k

        self.W = None
        self.y_corr = None

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
        predictions = np.zeros(1)
        return predictions

    def label_correlation(self, y, s):
        """Correlation between labels in a label matrix

        Parameters
        ----------
        y : array-like (n_samples, n_labels)
            Label matrix

        s : float
            Smoothness parameter

        Returns
        -------
        L : array-like (n_labels, n_labels)
            Label correlation matrix

        """
        L = np.zeros(y.shape[1], y.shape[1])


        return L

    """
    def estimateMissingLabels():


        return estimateMatrix
    """
    def weight_adjacent_matrix(self, X, k):
        """Using the kNN algorithm we will use the clusters to get a weight matrix

        Parameters
        ----------
        X : array-like or sparse matrix (n_samples, n_features)
            Data to classify or in this case to make clusters
        k : int
            Number of clusters we want to make
        
        Returns
        -------
        W : array-like (n_samples, n_samples)
            Weighted matrix created from the predictions of kNN
            wij = 1 if xi is in the same cluster as xj
            wij = 0 other case
        """
        kNN = KMeans(n_clusters=k)
        kNN.fit(X)
        predictions = kNN.predict(X)
        W = np.zeros(shape=[X.shape[0], X.shape[0]], dtype=int)
        for i in range(0, X.shape[0]):
            for j in range(0, X.shape[0]):
                if int(predictions[i]) == int(predictions[j]):
                    W[i,j] = 1
                else:
                    W[i,j] = 0

        return W