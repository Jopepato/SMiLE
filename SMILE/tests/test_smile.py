import pytest
import numpy as np
import unittest
from sklearn.datasets import load_iris
import sys
from SMILE.smile_ import SMiLE


def weightAdjacentMatrixTest(unittest):
    smile = SMiLE()
    data = load_iris()
    x = data.data
    weighted = smile.weightAdjacentMatrix(X=x, k=5)
    numberOf1s = 0
    for i in range(weighted.shape[0]):
        for j in range(weighted.shape[0]):
            numberOf1s += weighted[i,j]
    
    assert(numberOf1s != 0)

