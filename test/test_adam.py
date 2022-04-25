import unittest
from adam import *

class TestAdam(unittest.TestCase):

    def test_default_initialisation(self):
        adam = Adam()
        self.assertEqual(adam.alpha, 1e-3)
        self.assertEqual(adam.beta_1, 0.9)
        self.assertEqual(adam.beta_2, 0.999)
        self.assertEqual(adam.eps, 1e-8)
        self.assertEqual(adam.max_iter, 10)
        self.assertEqual(adam.xt, 0)
        self.assertEqual(adam.mt, 0)
        self.assertEqual(adam.vt, 0)
        self.assertEqual(adam.t, 0)
        self.assertEqual(adam.f, 0)