import unittest
import vortexwake as vw
import numpy as np
import numpy.testing as test
import json


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
    config["vortex_core_size"] = rng.random()
    config_name = "rng_config_{:d}d.json".format(dim)
    with open(config_name, "w") as f:
        json.dump(config, f, default=np_encoder, separators=(", ", ': '), indent=4)
    return config, config_name


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
        test.assert_allclose(vw.drot_z_dpsi_2d(theta), vw.drot_z_dpsi_3d(theta)[0:2, 0:2])


class TestVortexWake(unittest.TestCase):

    def setUp(self):
        self.skipTest("Testing delegated to subclasses")
        self.rng = None  # np.random.default_rng()
        self.vw = None
        self.fvw = None
        self.dimension = None
        self.config = None

    def test_config(self):
        test.assert_equal(self.fvw.dim, self.config["dimension"])
        test.assert_equal(self.fvw.time_step, self.config["time_step"])
        test.assert_equal(self.fvw.num_rings, self.config["num_rings"])
        test.assert_equal(self.fvw.num_turbines, self.config["num_turbines"])
        test.assert_equal(self.fvw.vortex_core_size, self.config["vortex_core_size"])

    def test_state_vector_conversions(self):
        q = self.rng.random((self.fvw.num_states, 1))
        states = self.fvw.states_from_state_vector(q)
        test.assert_allclose(q, self.fvw.state_vector_from_states(*states))

    def test_state_unpack(self):
        q = self.rng.random((self.fvw.num_states, 1))
        states = self.fvw.states_from_state_vector(q)
        test.assert_equal(len(states), 4)

    def test_initialise_states(self):
        X, G, U, M = self.fvw.initialise_states()
        q = self.fvw.state_vector_from_states(X, G, U, M)
        self.assertEqual(q.shape[0], self.fvw.num_states)

    def random_state_vector(self):
        return self.rng.random((self.fvw.num_states, 1))

    def random_control_vector(self):
        return self.rng.random(self.fvw.total_controls)

    def random_inflow_vector(self):
        return self.rng.random(self.dimension)

    def test_new_rings(self):
        fvw = self.vw("../config/base_{:d}d.json".format(self.dimension))
        q = self.rng.random((fvw.num_states, 1))
        m = self.rng.random(fvw.total_controls)
        u = self.random_inflow_vector()

        new_ring_states_no_tangent, new_ring_derivatives = fvw.new_rings(q, m, u, with_tangent=False)
        X0, G0, U0, M0 = new_ring_states_no_tangent
        (dX0_dq, dX0_dm), (dG0_dq, dG0_dm), (dU0_dq, dU0_dm), (dM0_dq, dM0_dm) = new_ring_derivatives

        new_ring_states, new_ring_derivatives = fvw.new_rings(q, m, u, with_tangent=True)
        X0, G0, U0, M0 = new_ring_states
        (dX0_dq, dX0_dm), (dG0_dq, dG0_dm), (dU0_dq, dU0_dm), (dM0_dq, dM0_dm) = new_ring_derivatives
        for a, b in zip(new_ring_states, new_ring_states_no_tangent):
            test.assert_allclose(a, b)

    # def test_disc_velocity():
    #     rng = np.random.default_rng()
    #     fvw = vw.VortexWake("../config/base_3d.json")
    #     q = rng.random(fvw.num_states)
    #     m = rng.random(fvw.total_controls)
    #     u = rng.random(fvw.dim)
    #     ur, dur_dq, dur_dm = fvw.disc_velocity(q, m, with_tangent=True)
    #     print(ur.shape, dur_dq.shape, dur_dm.shape)
    #
    #
    def test_velocity(self):
        fvw = self.vw("../config/base_{:d}d.json".format(self.dimension))
        number_of_points = self.rng.integers(1, 20)
        p =self.rng.standard_normal((number_of_points, self.dimension))
        q = self.random_state_vector()
        m = self.random_control_vector()
        u_no_tangent, du_dq, du_dm = fvw.velocity(q, m, p, with_tangent=False)
        u, du_dq, du_dm = fvw.velocity(q, m, p, with_tangent=True)
        test.assert_almost_equal(u_no_tangent, u)

    def test_disc_velocity(self):
        fvw = self.vw("../config/base_{:d}d.json".format(self.dimension))
        q = self.random_state_vector()
        m = self.random_control_vector()
        ur_no_tangent, dur_dq, dur_dm = fvw.disc_velocity(q, m, with_tangent=False)
        ur, dur_dq, dur_dm = fvw.disc_velocity(q, m, with_tangent=True)
        test.assert_almost_equal(ur_no_tangent, ur)


class TestVortexWake3D(TestVortexWake):

    def setUp(self):
        self.rng = np.random.default_rng()
        self.config, config_name = generate_random_config(3)
        self.vw = vw.VortexWake3D
        self.fvw = vw.VortexWake3D(config_name)
        self.dimension = 3

    def test_class_dimension_error(self):
        config, config_name = generate_random_config(3)
        self.assertRaises(ValueError, vw.VortexWake2D, config_name)

    # todo: def test_new_ring_positions(self):

    def test_config_3d(self):
        test.assert_equal(self.fvw.num_elements, self.config["num_elements"])


class TestVortexWake2D(TestVortexWake):
    def setUp(self):
        self.rng = np.random.default_rng()
        self.config, config_name = generate_random_config(2)
        self.vw = vw.VortexWake2D
        self.fvw = vw.VortexWake2D(config_name)
        self.dimension = 2

    def test_class_dimension_error(self):
        config, config_name = generate_random_config(2)
        self.assertRaises(ValueError, vw.VortexWake3D, config_name)

    def test_new_ring_positions(self):
        q = self.random_state_vector()
        m = self.random_control_vector()
        u = self.random_inflow_vector()

        new_ring_states, new_ring_derivatives = self.fvw.new_rings(q, m, u, with_tangent=True)
        X0, G0, U0, M0 = new_ring_states
        for wt in range(self.fvw.num_turbines):
            test.assert_almost_equal(np.linalg.norm(X0[wt, 0] - X0[wt, 1]), 1)  # unit diameter
            test.assert_almost_equal(X0[wt, 0] + X0[wt, 1] - 2 * self.fvw.turbine_positions[wt],
                                     np.zeros(2))  # opposing points and position

    def test_config_2d(self):
        test.assert_equal(self.fvw.num_elements, 1)

    # @unittest.skip("not implemented yet")
    def test_disc_velocity(self):
        super().test_disc_velocity()

    @unittest.skip("not implemented yet")
    def test_velocity(self):
        super().test_velocity()

# todo: generalise set up
# todo: test magnitude of vortex strength
# todo: robust derivative testing?
if __name__ == '__main__':
    unittest.main()