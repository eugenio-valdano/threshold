"""
Tests the library.
"""

from pandas import read_csv
import numpy as np
import threshold.threshold as thr
import threshold.threshold_util as thu

# read sample networks
df = read_csv('tests/net.csv', sep=',')

# get list of csr matrices
lA = thu.DataFrame_to_lA(df, weight='weight')

# threshold object
Z = thr.threshold(lA, weighted=False)

# compute threshold
lc = Z.compute(0.5)

# TESTS
import unittest
class myTest(unittest.TestCase):

    def test_upload(self):
        self.assertTrue(Z.N==200)

    def test_comput(self):
        print 'threshold: {:.6f}'.format(lc)
        self.assertTrue(np.abs(lc-0.0456174793566)<1e-3)



if __name__ == '__main__':
    unittest.main()
