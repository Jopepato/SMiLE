from .functions import label_correlation, estimate_mising_labels, weight_adjacent_matrix, diagonal_matrix_H, diagonal_matrix_Hc, diagonal_matrix_lambda, graph_laplacian_matrix, predictive_matrix, label_bias
import numpy as np

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

    L : array, [n_labels, n_labels]
        Correlation matrix between labels
    
    W : array, [n_samples, n_samples]
        Weighted matrix created by kNN for instances
    
    estimate_matrix : array-like (n_samples, n_labels)
        Label estimation matrix
        y~ic = yiT * L(.,c) if yic == 0
        y~ic = 1 otherwise

    H : array-like (n_samples, n_samples)
        Diagonal matrix indicating if an element of X is labeled or not    
    
    diagonal_lambda : array-like (n_samples, n_samples)
        Diagonal matrix having the sum of weights of the weighted matrix

    M : array-like (n_samples, n_samples)
        Graph laplacian matrix
    
    Hc : array-like (n_samples, n_samples)
        Hc = H - (H*1*1t*Ht)/(N)
    
    P : array-like (n_features, n_labels)
        P = (X*Hc*Xt + alpha*X*M*Xt)-1 * X*Hc*YPred
        R = dxc

    b : array-like (n_labels)
        Label bias as the second item of the equation
        b = ((estimate_matrix - Pt*X)*H*1)/N
     
    """

    def __init__(self, s=0.5, alpha=0.35, k=5):
        """Initialize properties.

        :param s:
        :param alpha:
        :param k:
        :param W:
        :param L:
        :param estimate_matrix:
        :param H:
        :param diagonal_lambda:
        :param M:
        :param Hc:
        :param P:
        :param b:
        """
        self.s = s
        self.alpha = alpha
        self.k = k

        self.W = None
        self.L = None
        self.estimate_matrix = None
        self.H = None
        self.diagonal_lambda = None
        self.M = None
        self.Hc = None
        self.P = None
        self.b = None


    def fit(self, X, y):
        """Fits the model

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances.
        y : array-like, shape=(n_samples, n_labels)
            Training labels.
        """
        #TODO Ensure the input format
        self.L = label_correlation(y, self.s)
        self.estimate_matrix = estimate_mising_labels(y, self.L)
        self.H = diagonal_matrix_H(X, y)
        self.Hc = diagonal_matrix_Hc(self.H)
        self.W = weight_adjacent_matrix(X, self.k)
        self.diagonal_lambda = diagonal_matrix_lambda(self.W)
        self.M = graph_laplacian_matrix(self.diagonal_lambda, self.W)
        self.P = predictive_matrix(X, self.Hc, self.M, self.estimate_matrix, self.alpha)
        self.b = label_bias(self.estimate_matrix, self.P, X, self.H)

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
        #TODO Ensure the input and output format
        predictions = np.zeros(shape=[X.shape[0], self.b.shape[0]])
        for i in range(0, X.shape[0]):
            numerator1 = np.matmul(np.transpose(self.P), X[i,:])
            predictions[i] = numerator1 + self.b
        return predictions