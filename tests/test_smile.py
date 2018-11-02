import pytest
import numpy as np
import unittest
from sklearn.datasets import load_iris, make_multilabel_classification
import sys
from SMILE.smile_ import SMiLE


class SmileTest(unittest.TestCase):
    def test_weight(self):
        smile = SMiLE()
        data = load_iris()
        x = data.data
        weighted = smile.weight_adjacent_matrix(X=x, k=5)
        numberOf1s = 0
        for i in range(weighted.shape[0]):
            for j in range(weighted.shape[0]):
                numberOf1s += weighted[i,j]
        self.assertTrue(numberOf1s != 0)

    def test_label_correlation(self):
        
        smile = SMiLE()
        X, y = make_multilabel_classification()
        correlation = np.zeros(shape=[y.shape[1], y.shape[1]])
        correlation = smile.label_correlation(y, smile.s)
        notEmpty = 0
        for i in range(correlation.shape[0]):
            for j in range(correlation.shape[1]):
                notEmpty += correlation[i,j]
        self.assertTrue(notEmpty != 0)
    
    def test_estimate_missing_labels(self):

        smile = SMiLE()
        X, y = make_multilabel_classification()
        correlation = smile.label_correlation(y, smile.s)
        estimate_matrix = np.zeros(shape=[y.shape[0], y.shape[1]])
        estimate_matrix = smile.estimate_mising_labels(y, correlation)
        notEmpty = 0
        for i in range(estimate_matrix.shape[0]):
            for j in range(estimate_matrix.shape[1]):
                notEmpty += estimate_matrix[i,j]
        self.assertTrue(notEmpty != 0)

if __name__ == '__main__':
    unittest.main()
