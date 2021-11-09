import unittest

import ars
import numpy as np

# Not very meaningfull...
class TestHyperParameters(unittest.TestCase):
    def test_init(self):
        hp = HyperParameters()
        self.assertEqual(hp.number_of_steps, 1024)

# test with vector of 5
ni = 5

class TestNormalizer(unittest.TestCase):
    def test_init(self):
        np.allclose(norm.n, np.zeros(ni))

    # one observation...
    def test_obs1(self):
        norm = Normalizer(ni)
        norm.observe(np.array([1,2,3,4,5]))
        # Check if n and mean are correct for one obeservation
        self.assertTrue(np.allclose(norm.n,         np.array([1,1,1,1,1])))
        self.assertTrue(np.allclose(norm.mean,      np.array([1,2,3,4,5])))
        self.assertTrue(np.allclose(norm.mean_diff, np.array([0,0,0,0,0])))
        # dont forget the divede by zero prevention 
        self.assertTrue(np.allclose(norm.var,       np.array([1e-2,1e-2,1e-2,1e-2,1e-2])))
        
    # two observations...
    def test_obs2(self):
        norm = Normalizer(ni)
        norm.observe(np.array([1,2,3,4,5]))
        norm.observe(np.array([6,7,8,9,10]))
        # (6 - 1) * (6 - 3.5)
        # (7 - 2) * (7 - 4.5) ...
        self.assertTrue(np.allclose(norm.n,         np.array([ 2   , 2   , 2   , 2   , 2   ])))
        self.assertTrue(np.allclose(norm.mean,      np.array([ 3.5 , 4.5 , 5.5 , 6.5 , 7.5 ])))
        self.assertTrue(np.allclose(norm.mean_diff, np.array([12.5 ,12.5 ,12.5 ,12.5 ,12.5 ])))
        self.assertTrue(np.allclose(norm.var,       np.array([ 6.25+1e-2,6.25+1e-2,6.25+1e-2,6.25+1e-2,6.25+1e-2])))

    def test_normalize(self):
        norm = Normalizer(ni)
        norm.observe(np.array([1,2,3,4,5]))
        norm.observe(np.array([6,7,8,9,10]))
        # 1 - 3.5 / sqrt 6.26 = -0.9992009587217894232990894329571
        # 8 - 4.5 / sqrt 6.26 =  1.3988813422105051926187252061399
        # 2 - 5.5 ..          = -1.3988813422105051926187252061399
        # 7 - 6.5 ..          =  0.19984019174435788465981788659142
        # 4 - 7.5 ..          = -1.3988813422105051926187252061399
        a = norm.normalize(np.array([1,8,2,7,4]))
        self.assertTrue(np.allclose(a, np.array([ -0.999200958, 1.398881342, -1.398881342, 0.199840191, -1.398881342 ])))
        

if __name__ == '__main__':
    unittest.main()