import unittest
from adam import *

class TestAdam(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng()

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

    def test_init_from_config(self):
        config = {}
        config["alpha"] = 1e-2*self.rng.random()
        config["beta_1"] = self.rng.random()
        config["beta_2"] = self.rng.random()
        config["eps"] = 1-4*self.rng.random()
        config["max_iter"] = self.rng.integers(0,300)
        adam = Adam(config)
        self.assertEqual(adam.alpha, config["alpha"])
        self.assertEqual(adam.beta_1, config["beta_1"])
        self.assertEqual(adam.beta_2, config["beta_2"])
        self.assertEqual(adam.eps, config["eps"])
        self.assertEqual(adam.max_iter, config["max_iter"])
        self.assertEqual(adam.xt, 0)
        self.assertEqual(adam.mt, 0)
        self.assertEqual(adam.vt, 0)
        self.assertEqual(adam.t, 0)
        self.assertEqual(adam.f, 0)