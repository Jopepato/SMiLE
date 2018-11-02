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
        X, y, p_c, p_w_c = make_multilabel_classification(n_labels=50)
        correlation = np.zeros(y.shape[1], y.shape[1])
        correlation = smile.label_correlation(y, smile.s)
        notEmpty = 0
        for i in range(correlation.shape[0]):
            for j in range(correlation.shape[1]):
                notEmpty += correlation[i,j]
        self.assertTrue(notEmpty != 0)




if __name__ == '__main__':
    unittest.main()
