import unittest
import numpy.testing as test
from vortexwake.adam import *


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
        config["alpha"] = 1e-2 * self.rng.random()
        config["beta_1"] = self.rng.random()
        config["beta_2"] = self.rng.random()
        config["eps"] = 1 - 4 * self.rng.random()
        config["max_iter"] = self.rng.integers(0, 300)
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

    def test_optimisation(self):
        def f(x, q):
            return np.sum(x ** 2), 2 * x

        x0 = self.rng.random(10)
        q0 = 0
        y0 = f(x0, q0)[0]
        adam = Adam({"alpha": 1e-2})
        x_opt = adam.minimise(f, x0, q0)
        y_opt = f(x_opt, q0)[0]

        test.assert_array_less(y_opt, y0)
        test.assert_array_less(np.abs(x_opt), np.abs(x0))

    def test_nan_in_solution(self):
        def f(x, q):
            return np.sum(x ** 2), 2 * x

        x0 = self.rng.random(10)
        x0[self.rng.integers(0, 10)] = np.nan
        q0 = 0
        adam = Adam({"alpha": 1e-2})
        self.assertRaises(ValueError, adam.minimise, f, x0, q0)
