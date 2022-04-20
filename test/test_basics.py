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


def test_state_vector_conversions():
    rng = np.random.default_rng()

    fvw2 = vw.VortexWake("../config/base_2d.json")
    q = rng.random((fvw2.num_states, 1))
    np.testing.assert_allclose(q, fvw2.state_vector_from_states(*fvw2.states_from_state_vector(q)))
    # fvw3 = vw.VortexWake("base_2d.json")
