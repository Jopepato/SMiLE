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
        L = np.zeros(shape=[y.shape[1], y.shape[1]])

        for i in range(0, y.shape[1]):
            for j in range(0, y.shape[1]):
                coincidence = 0
                yi = sum(y[:,i])
                for k in range(0, y.shape[0]):
                    if y[k,i] == y[k,j]:
                        coincidence += 1
                L[i,j] = (coincidence + s)/(yi + 2*s)

        return L

    
    def estimate_mising_labels(self, y, L):
        """Estimation of the missing labels, using the correlation matrix

        Parameters
        ----------
        y : array-like (n_samples, n_labels)
            Label matrix

        L : array-like (n_labels, n_labels)
            Label correlation matrix

        Returns
        -------
        estimateMatrix : array-like (n_samples, n_labels)
            Label estimation matrix
            y~ic = yiT * L(.,c) if yic == 0
            y~ic = 1 otherwise
        """

        estimateMatrix = np.zeros(shape=[y.shape[0],y.shape[1]])
        for i in range(0, y.shape[0]):
            for j in range(0, y.shape[1]):
                if y[i,j] == 0:
                    estimateMatrix[i,j] = np.dot(np.transpose(y[i,:]), L[:,j])
                else:
                    estimateMatrix[i,j] = 1

        return estimateMatrix

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
    
    def diagonal_matrix_H(self, X, y):
        """Diagonal matrix that indicates if X is labeled

        Parameters
        ----------
        X : array-like or sparse matrix (n_samples, n_features)
            Data to classify
        y : array-like (n_samples, n_labels)
            Labels of the data

        Returns
        -------
        H : array-like (n_samples, n_samples)
            Diagonal matrix indicating if an element of X is labeled or not
        """

        H = np.zeros(shape=[X.shape[0], X.shape[0]])

        for i in range(0, X.shape[0]):
            if np.sum(y[i,:]) != 0:
                H[i,i] = 1

        return H
    
    def diagonal_matrix_lambda(self, W):
        """Diagonal matrix that indicates if X is labeled

        Parameters
        ----------
        W : array-like (n_samples, n_samples)
            Weighted matrix

        Returns
        -------
        diagonal_lambda : array-like (n_samples, n_samples)
            Diagonal matrix having the sum of weights of the weighted matrix
        """
        diagonal_lambda = np.zeros(shape=[W.shape[0], W.shape[1]])
        for i in range(0, W.shape[0]):
            diagonal_lambda[i,i] = np.sum(W[i,:])
        
        return diagonal_lambda


    def graph_laplacian_matrix(self, lambda_matrix, W):
        """Diagonal matrix that indicates if X is labeled

        Parameters
        ----------
        lambda_matrix : array-like (n_samples, n_samples)
            Diagonal matrix having the sum of weights of the weighted matrix
        W : array-like (n_samples, n_samples)
            Weighted matrix

        Returns
        -------
        M : array-like (n_samples, n_samples)
            Graph laplacian matrix
        """
        M = np.zeros(shape=[W.shape[0], W.shape[1]])
        M = np.abs(np.array(lambda_matrix) - np.array(W))
        return M
    
    def diagonal_matrix_Hc(self, H):
        """Diagonal matrix that indicates if X is labeled

        Parameters
        ----------
        H : array-like (n_samples, n_samples)
            Diagonal matrix indicating if an element of X is labeled or not

        Returns
        -------
        Hc : array-like (n_samples, n_samples)
            Hc = H - (H*1*1t*Ht)/(N)
        """
        Hc = np.zeros(shape = [H.shape[0], H.shape[0]])
        ident = np.identity(n=H.shape[0])
        numerator1 = np.dot(H, ident)
        numerator2 = np.dot(np.transpose(ident), np.transpose(H))
        numerator = np.dot(numerator1, numerator2)
        product = numerator/H.shape[0]
        Hc = np.abs(H - product)
        return Hc