import unittest
import vortexwake as vw
import numpy as np
import numpy.testing as test
import json


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
    test.assert_allclose(q, fvw.state_vector_from_states(*states))


def test_state_vector_conversions_3d():
    rng = np.random.default_rng()
    fvw = vw.VortexWake("../config/base_3d.json")
    q = rng.random((fvw.num_states, 1))
    states = fvw.states_from_state_vector(q)
    test.assert_allclose(q, fvw.state_vector_from_states(*states))

def test_state_unpack():
    rng = np.random.default_rng()
    fvw = vw.VortexWake("../config/base_3d.json")
    q = rng.random((fvw.num_states, 1))
    states = fvw.states_from_state_vector(q)
    test.assert_equal(len(states), 4)

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()
    if isinstance(object, np.ndarray):
        return object.tolist()


def generate_random_config(dim=3):
    rng = np.random.default_rng()
    config = {}
    config["dimension"] = dim
    config["time_step"] = rng.random()
    config["num_rings"] = rng.integers(3,100)
    #todo: catch zero turbine definition?
    config["num_elements"] = rng.integers(3,50)
    config["num_turbines"] = rng.integers(1,5)
    config["turbine_positions"] = rng.random((config["num_turbines"],config["dimension"]))
    config_name = "rng_config_{:d}d.json".format(dim)
    with open(config_name,"w") as f:
        json.dump(config, f, default=np_encoder, separators=(", ", ': '), indent=4)
    return config, config_name


def test_config():
    for dim in [2,3]:
        config, config_name = generate_random_config(dim)
        fvw = vw.VortexWake(config_name)
        test.assert_equal(fvw.dim, config["dimension"])
        test.assert_equal(fvw.time_step, config["time_step"])
        test.assert_equal(fvw.num_rings, config["num_rings"])
        test.assert_equal(fvw.num_elements, config["num_elements"])
        test.assert_equal(fvw.num_turbines, config["num_turbines"])
    # test.assert_equal(fvw., config["dimension"])


