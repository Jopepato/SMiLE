import pytest
import numpy as np
import unittest
from sklearn.datasets import load_iris
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
        self.assertFalse(numberOf1s == 0)


if __name__ == '__main__':
    unittest.main()
