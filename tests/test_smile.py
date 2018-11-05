import pytest
import numpy as np
import unittest
from sklearn.datasets import load_iris, make_multilabel_classification
import sys
from SMILE.smile_ import SMiLE
from SMILE.functions import *


class SmileTest(unittest.TestCase):
    def test_weight(self):
        X, y = make_multilabel_classification()
        W = weight_adjacent_matrix(X=X, k=5)
        self.assertTrue(np.sum(W) != 0)

    def test_label_correlation(self):
    
        X, y = make_multilabel_classification()
        correlation = np.zeros(shape=[y.shape[1], y.shape[1]])
        correlation = label_correlation(y, s=0.5)
        self.assertTrue(np.sum(correlation) != 0)
    
    def test_estimate_missing_labels(self):

        X, y = make_multilabel_classification()
        correlation = label_correlation(y, s=0.5)
        estimate_matrix = np.zeros(shape=[y.shape[0], y.shape[1]])
        estimate_matrix = estimate_mising_labels(y, correlation)
        self.assertTrue(np.sum(estimate_matrix) != 0)
    
    def test_diagonal_matrix_H(self):
        X, y = make_multilabel_classification()
        diagonal_matrix = np.zeros(shape=[X.shape[0], X.shape[0]])
        diagonal_matrix = diagonal_matrix_H(X, y)
        self.assertTrue(np.sum(diagonal_matrix) != 0)
    
    def test_diagonal_matrix_lambda(self):
        X, y = make_multilabel_classification()
        W = weight_adjacent_matrix(X=X, k=5)
        diagonal_lambda = np.zeros(shape=[X.shape[0], X.shape[0]])
        diagonal_lambda = diagonal_matrix_lambda(W)
        self.assertTrue(np.sum(diagonal_lambda) != 0)

    def test_laplacian_matrix(self):
        X, y = make_multilabel_classification()
        W = weight_adjacent_matrix(X=X, k= 5)
        diagonal_lambda = diagonal_matrix_lambda(W)
        M = np.zeros(shape=[X.shape[0], X.shape[0]])
        M = graph_laplacian_matrix(diagonal_lambda, W)

        self.assertTrue(np.sum(M) != 0)

    def test_diagonal_matrix_Hc(self):
        X, y = make_multilabel_classification()
        H = diagonal_matrix_H(X, y)
        Hc = np.zeros(shape = [H.shape[0], H.shape[1]])
        Hc = diagonal_matrix_Hc(H)
        self.assertTrue(np.sum(Hc) != 0)

    def test_predictive_matrix(self):
        X, y = make_multilabel_classification()
        L = label_correlation(y, s=0.5)
        estimate_matrix = estimate_mising_labels(y, L)
        H = diagonal_matrix_H(X, y)
        Hc = diagonal_matrix_Hc(H)
        W = weight_adjacent_matrix(X,k=5)
        lambda_matrix = diagonal_matrix_lambda(W)
        M = graph_laplacian_matrix(lambda_matrix, W)
        P = np.zeros(shape= [X.shape[1], y.shape[1]])
        P = predictive_matrix(X, Hc, M, estimate_matrix, alpha=0.35)
        self.assertTrue(np.sum(P) != 0)

    def test_label_bias(self):
        X, y = make_multilabel_classification()
        L = label_correlation(y, s=0.5)
        estimate_matrix = estimate_mising_labels(y, L)
        H = diagonal_matrix_H(X, y)
        Hc = diagonal_matrix_Hc(H)
        W = weight_adjacent_matrix(X,k=5)
        lambda_matrix = diagonal_matrix_lambda(W)
        M = graph_laplacian_matrix(lambda_matrix, W)
        P = predictive_matrix(X, Hc, M, estimate_matrix, alpha=0.35)
        b = np.zeros(y.shape[1])
        b = label_bias(estimate_matrix, P, X, H)
        self.assertTrue(np.sum(b) != 0)

    
    def test_fit(self):
        smile = SMiLE()
        X, y = make_multilabel_classification()
        smile.fit(X, y)
        self.assertTrue(np.sum(smile.P) != 0)
        self.assertTrue(np.sum(smile.b) != 0)


    def test_predict(self):
        smile = SMiLE()
        X, y = make_multilabel_classification()
        smile.fit(X,y)
        predictions = np.zeros(shape=[X.shape[0], y.shape[1]])
        predictions = smile.predict(X)
        self.assertTrue(np.sum(predictions) != 0)


if __name__ == '__main__':
    unittest.main()
