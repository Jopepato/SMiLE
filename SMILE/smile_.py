import numpy as np
from sklearn.cluster import KMeans
from numpy.linalg import inv


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
        self.L = self.label_correlation(y, self.s)
        self.estimate_matrix = self.estimate_mising_labels(y, self.L)
        self.H = self.diagonal_matrix_H(X, y)
        self.Hc = self.diagonal_matrix_Hc(self.H)
        self.W = self.weight_adjacent_matrix(X, self.k)
        self.diagonal_lambda = smile.diagonal_matrix_lambda(self.W)
        self.M = self.graph_laplacian_matrix(self.diagonal_lambda, self.W)
        self.P = self.predictive_matrix(X, self.Hc, self.M, self.estimate_matrix)
        self.b = self.label_bias(self.estimate_matrix, self.P, X, self.H)

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
            predictions[i] = np.matmul(np.transpose(self.P), X[i]) + self.b
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
        estimate_matrix : array-like (n_samples, n_labels)
            Label estimation matrix
            y~ic = yiT * L(.,c) if yic == 0
            y~ic = 1 otherwise
        """

        estimate_matrix = np.zeros(shape=[y.shape[0],y.shape[1]])
        for i in range(0, y.shape[0]):
            for j in range(0, y.shape[1]):
                if y[i,j] == 0:
                    estimate_matrix[i,j] = np.matmul(np.transpose(y[i,:]), L[:,j])
                else:
                    estimate_matrix[i,j] = 1
                #Normalize the data
                if np.sum(y[i,:]) != 0:
                    y[i,j] = y[i,j]/(np.sum(y[i,:]))

        return estimate_matrix

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
        """

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
        """

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
        """

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
        numerator1 = np.matmul(H, ident)
        numerator2 = np.matmul(np.transpose(ident), np.transpose(H))
        numerator = np.matmul(numerator1, numerator2)
        product = numerator/H.shape[0]
        Hc = np.abs(H - product)
        return Hc
    
    def predictive_matrix(self, X, Hc, M, estimate_matrix):
        """Predictive matrix that works as the first item of the equation

        Parameters
        ----------
        X : array-like or sparse matrix (n_samples, n_features)
            Data to be classified or trained
        Hc : array-like (n_samples, n_samples)
            Diagonal matrix obtained from H
        M : array-like(n_samples, n_samples)
            Graph laplacian matrix

        Returns
        -------
        P : array-like (n_features, n_labels)
            P = (X*Hc*Xt + alpha*X*M*Xt)-1 * X*Hc*YPred
            R = dxc
        """
        P = np.zeros(shape=[X.shape[1], estimate_matrix.shape[1]])
        numerator1 = np.matmul(np.transpose(X), Hc)
        numerator1 = np.matmul(numerator1, X)
        numerator2 = np.matmul(np.transpose(X), M)
        numerator2 = np.matmul(numerator2, X)
        numerator2 = self.alpha * numerator2
        numerator = numerator1 + numerator2
        numerator = inv(numerator)
        numerator2 = np.matmul(np.transpose(X), Hc)
        numerator2 = np.matmul(numerator2, estimate_matrix)
        P = np.matmul(numerator, numerator2)

        return P

    def label_bias(self, estimate_matrix, P, X, H):
        """Label bias that works as the second item of the equation

        Parameters
        ----------
        estimate_matrix : array-like (n_samples, n_samples)
            Diagonal matrix indicating if an element of X is labeled or not
        P : array-like (n_features, n_labels)
            Predictive item
        X : array-like (n_samples, n_features)
            Data to train or test
        H : array-like (n_samples, n_samples)
            Diagonal matrix indicating if an element of X is labeled or not

        Returns
        -------
        b : array-like (n_labels)
            Label bias as the second item of the equation
            b = ((estimate_matrix - Pt*X)*H*1)/N
        """
        b = np.zeros(estimate_matrix.shape[1])
        ytranspose = np.transpose(estimate_matrix)
        aux = np.matmul(np.transpose(P),np.transpose(X))
        numerator1 = np.abs(ytranspose - aux)
        numerator2 = np.matmul(H, np.identity(H.shape[0]))
        numerator = np.matmul(numerator1, numerator2)
        b = numerator / H.shape[0]
        return b