import unittest
import vortexwake as vw
import numpy as np


# class MyTest(unittest.TestCase):
#     def test_state_vector_conversions(self):
#         rng = np.random.default_rng()
#
#         fvw2 = vw.VortexWake("../config/base_2d.json")
#         q = rng.random((fvw2.num_states, 1))
#         self.assertEqual(q, fvw2.state_vector_from_states(*fvw2.states_from_state_vector(q)))
#         # fvw3 = vw.VortexWake("base_2d.json")


def test_state_vector_conversions_2d():
    rng = np.random.default_rng()
    fvw = vw.VortexWake("../config/base_2d.json")
    q = rng.random((fvw.num_states, 1))
    states = fvw.states_from_state_vector(q)
    np.testing.assert_allclose(q, fvw.state_vector_from_states(*states))


def test_state_vector_conversions_3d():
    rng = np.random.default_rng()
    fvw = vw.VortexWake("../config/base_3d.json")
    q = rng.random((fvw.num_states, 1))
    states = fvw.states_from_state_vector(q)
    np.testing.assert_allclose(q, fvw.state_vector_from_states(*states))
