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
    config["num_rings"] = rng.integers(3, 100)
    # todo: catch zero turbine definition?
    config["num_elements"] = rng.integers(3, 50)
    config["num_turbines"] = rng.integers(1, 5)
    config["turbine_positions"] = rng.random((config["num_turbines"], config["dimension"]))
    config_name = "rng_config_{:d}d.json".format(dim)
    with open(config_name, "w") as f:
        json.dump(config, f, default=np_encoder, separators=(", ", ': '), indent=4)
    return config, config_name


def test_config():
    for dim in [2, 3]:
        config, config_name = generate_random_config(dim)
        fvw = vw.VortexWake(config_name)
        test.assert_equal(fvw.dim, config["dimension"])
        test.assert_equal(fvw.time_step, config["time_step"])
        test.assert_equal(fvw.num_rings, config["num_rings"])
        if dim==2:
            test.assert_equal(fvw.num_elements, 1)
        elif dim==3:
            test.assert_equal(fvw.num_elements, config["num_elements"])
        test.assert_equal(fvw.num_turbines, config["num_turbines"])
    # test.assert_equal(fvw., config["dimension"])


def test_rotation():
    # clockwise positive rotation
    unit_vector_x = np.array([1, 0, 0])
    unit_vector_y = np.array([0, 1, 0])
    unit_vector_z = np.array([0, 0, 1])
    theta = 0.
    test.assert_almost_equal(unit_vector_x @ vw.rot_z_3d(theta).T, unit_vector_x)
    theta = 360.
    test.assert_almost_equal(unit_vector_x @ vw.rot_z_3d(theta).T, unit_vector_x)
    theta = -180.
    test.assert_almost_equal(unit_vector_x @ vw.rot_z_3d(theta).T, -unit_vector_x)

    theta = 90.
    test.assert_almost_equal(unit_vector_x @ vw.rot_z_3d(theta).T, -unit_vector_y)
    theta = -90.
    test.assert_almost_equal(unit_vector_x @ vw.rot_z_3d(theta).T, unit_vector_y)

    rng = np.random.default_rng()
    for theta in 360 * rng.standard_normal(10):
        test.assert_almost_equal(unit_vector_z @ vw.rot_z_3d(theta).T, unit_vector_z)


def test_rotation_derivative():
    unit_vector_x = np.array([1, 0, 0])
    unit_vector_y = np.array([0, 1, 0])
    unit_vector_z = np.array([0, 0, 1])
    theta = 0.
    test.assert_almost_equal(unit_vector_x @ vw.drot_z_dpsi_3d(theta).T, -unit_vector_y * (np.pi / 180.))
    theta = 360.
    test.assert_almost_equal(unit_vector_x @ vw.drot_z_dpsi_3d(theta).T, -unit_vector_y * (np.pi / 180.))
    theta = -180.
    test.assert_almost_equal(unit_vector_x @ vw.drot_z_dpsi_3d(theta).T, unit_vector_y * (np.pi / 180.))

    theta = 90.
    test.assert_almost_equal(unit_vector_x @ vw.drot_z_dpsi_3d(theta).T, -unit_vector_x * (np.pi / 180.))
    theta = -90.
    test.assert_almost_equal(unit_vector_x @ vw.drot_z_dpsi_3d(theta).T, unit_vector_x * (np.pi / 180.))

    rng = np.random.default_rng()
    for theta in 360 * rng.standard_normal(10):
        test.assert_almost_equal(unit_vector_z @ vw.drot_z_dpsi_3d(theta).T, np.zeros(3))

def test_rotation_2d():
    rng = np.random.default_rng()
    for theta in 360 * rng.standard_normal(20):
        test.assert_allclose(vw.rot_z_2d(theta), vw.rot_z_3d(theta)[0:2, 0:2])
        test.assert_allclose(vw.drot_z_dpsi_2d(theta), vw.drot_z_dpsi_3d(theta)[0:2,0:2])

def test_new_rings_3d():
    rng = np.random.default_rng()
    fvw = vw.VortexWake("../config/base_3d.json")
    q = rng.random(fvw.num_states)
    m = rng.random(fvw.total_controls)
    u = rng.random(fvw.dim)
    new_ring_states_no_tangent, new_ring_derivatives = fvw.new_rings(q, m, u, with_tangent=False)
    X0, G0, U0, M0 = new_ring_states_no_tangent
    (dX0_dq,dX0_dm), (dG0_dq, dG0_dm), (dU0_dq, dU0_dm), (dM0_dq, dM0_dm) = new_ring_derivatives

    fvw.new_rings(q, m, u, with_tangent=True)
    new_ring_states, new_ring_derivatives = fvw.new_rings(q, m, u, with_tangent=False)
    X0, G0, U0, M0 = new_ring_states
    (dX0_dq,dX0_dm), (dG0_dq, dG0_dm), (dU0_dq, dU0_dm), (dM0_dq, dM0_dm) = new_ring_derivatives
    for a,b in zip(new_ring_states, new_ring_states_no_tangent):
        test.assert_allclose(a,b)

    #todo: test actual position of points

def test_new_rings_2d():
    rng = np.random.default_rng()
    # fvw = vw.VortexWake("../config/base_2d.json")
    config, config_name = generate_random_config(2)
    fvw = vw.VortexWake(config_name)
    q = rng.random(fvw.num_states)
    m = rng.random(fvw.total_controls)
    u = rng.random(fvw.dim)
    new_ring_states_no_tangent, new_ring_derivatives = fvw.new_rings(q, m, u, with_tangent=False)
    X0, G0, U0, M0 = new_ring_states_no_tangent
    (dX0_dq, dX0_dm), (dG0_dq, dG0_dm), (dU0_dq, dU0_dm), (dM0_dq, dM0_dm) = new_ring_derivatives

    fvw.new_rings(q, m, u, with_tangent=True)
    new_ring_states, new_ring_derivatives = fvw.new_rings(q, m, u, with_tangent=False)
    X0, G0, U0, M0 = new_ring_states
    (dX0_dq, dX0_dm), (dG0_dq, dG0_dm), (dU0_dq, dU0_dm), (dM0_dq, dM0_dm) = new_ring_derivatives
    for a, b in zip(new_ring_states, new_ring_states_no_tangent):
        test.assert_allclose(a, b)

    for wt in range(fvw.num_turbines):
        test.assert_almost_equal(np.linalg.norm(X0[wt,0]-X0[wt,1]),1) # unit diameter
        test.assert_almost_equal(X0[wt,0]+X0[wt,1] - 2*fvw.turbine_positions[wt], np.zeros(2)) # opposing points and position


def test_disc_velocity():
    rng = np.random.default_rng()
    fvw = vw.VortexWake("../config/base_3d.json")
    q = rng.random(fvw.num_states)
    m = rng.random(fvw.num_controls)
    u = rng.random(fvw.dim)
    ur, dur_dq, dur_dm = fvw.disc_velocity(q,m,with_tangent=True)
    print(ur.shape, dur_dq.shape, dur_dm.shape)

def test_initialise_states():
    for dim in [2, 3]:
        config, config_name = generate_random_config(dim)
        fvw = vw.VortexWake(config_name)
        X,G,U,M = fvw.initialise_states()
        q = fvw.state_vector_from_states(X,G,U,M)


#todo: generalise set up
#todo: test magnitude of vortex strength
#todo: robust derivative testing?
