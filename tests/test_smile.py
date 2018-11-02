import pytest
import numpy as np
import unittest
from sklearn.datasets import load_iris, make_multilabel_classification
import sys
from SMILE.smile_ import SMiLE


class SmileTest(unittest.TestCase):
    def test_weight(self):
        smile = SMiLE()
        X, y = make_multilabel_classification()
        W = smile.weight_adjacent_matrix(X=X, k=5)
        self.assertTrue(np.sum(W) != 0)

    def test_label_correlation(self):
        
        smile = SMiLE()
        X, y = make_multilabel_classification()
        correlation = np.zeros(shape=[y.shape[1], y.shape[1]])
        correlation = smile.label_correlation(y, smile.s)
        self.assertTrue(np.sum(correlation) != 0)
    
    def test_estimate_missing_labels(self):

        smile = SMiLE()
        X, y = make_multilabel_classification()
        correlation = smile.label_correlation(y, smile.s)
        estimate_matrix = np.zeros(shape=[y.shape[0], y.shape[1]])
        estimate_matrix = smile.estimate_mising_labels(y, correlation)
        self.assertTrue(np.sum(estimate_matrix) != 0)
    
    def test_diagonal_matrix_H(self):
        smile = SMiLE()
        X, y = make_multilabel_classification()
        diagonal_matrix = np.zeros(shape=[X.shape[0], X.shape[0]])
        diagonal_matrix = smile.diagonal_matrix_H(X, y)
        self.assertTrue(np.sum(diagonal_matrix) != 0)
    
    def test_diagonal_matrix_lambda(self):
        smile = SMiLE()
        X, y = make_multilabel_classification()
        W = smile.weight_adjacent_matrix(X=X, k=5)
        diagonal_lambda = np.zeros(shape=[X.shape[0], X.shape[0]])
        diagonal_lambda = smile.diagonal_matrix_lambda(W)
        self.assertTrue(np.sum(diagonal_lambda) != 0)

    def test_laplacian_matrix(self):
        smile = SMiLE()
        X, y = make_multilabel_classification()
        W = smile.weight_adjacent_matrix(X=X, k= 5)
        diagonal_lambda = smile.diagonal_matrix_lambda(W)
        M = np.zeros(shape=[X.shape[0], X.shape[0]])
        M = smile.graph_laplacian_matrix(diagonal_lambda, W)

        self.assertTrue(np.sum(M) != 0)

    def test_diagonal_matrix_Hc(self):
        smile = SMiLE()
        X, y = make_multilabel_classification()
        H = smile.diagonal_matrix_H(X, y)
        Hc = np.zeros(shape = [H.shape[0], H.shape[1]])
        Hc = smile.diagonal_matrix_Hc(H)
        self.assertTrue(np.sum(Hc) != 0)

    def test_predective_matrix(self):
        smile = SMiLE()
        X, y = make_multilabel_classification()
        L = smile.label_correlation(y, smile.s)
        estimate_matrix = smile.estimate_mising_labels(y, L)
        H = smile.diagonal_matrix_H(X, y)
        Hc = smile.diagonal_matrix_Hc(H)
        W = smile.weight_adjacent_matrix(X,k=5)
        lambda_matrix = smile.diagonal_matrix_lambda(W)
        M = smile.graph_laplacian_matrix(lambda_matrix, W)
        P = np.zeros(shape=[X.shape[1], y.shape[1]])
        P = smile.predective_matrix(X, Hc, M, estimate_matrix)
        self.assertTrue(np.sum(P) != 0)

    def test_label_bias(self):
        smile = SMiLE()
        X, y = make_multilabel_classification()
        L = smile.label_correlation(y, smile.s)
        estimate_matrix = smile.estimate_mising_labels(y, L)
        H = smile.diagonal_matrix_H(X, y)
        Hc = smile.diagonal_matrix_Hc(H)
        W = smile.weight_adjacent_matrix(X,k=5)
        lambda_matrix = smile.diagonal_matrix_lambda(W)
        M = smile.graph_laplacian_matrix(lambda_matrix, W)
        P = smile.predective_matrix(X, Hc, M, estimate_matrix)
        b = np.zeros(y.shape[1])
        b = smile.label_bias(estimate_matrix, P, X, H)
        self.assertTrue(False)




if __name__ == '__main__':
    unittest.main()
