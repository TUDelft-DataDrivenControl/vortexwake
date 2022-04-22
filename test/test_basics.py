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
    config["num_virtual_turbines"] = rng.integers(1, 5)
    config["turbine_positions"] = rng.random(
        (config["num_turbines"] + config["num_virtual_turbines"], config["dimension"]))
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
        test.assert_equal(self.fvw.total_turbines, self.fvw.num_turbines + self.fvw.num_virtual_turbines)

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
        p = self.rng.standard_normal((number_of_points, self.dimension))
        q = self.rng.random((fvw.num_states, 1))
        m = self.rng.random(fvw.total_controls)
        u_no_tangent, du_dq, du_dm = fvw.velocity(q, m, p, with_tangent=False)
        u, du_dq, du_dm = fvw.velocity(q, m, p, with_tangent=True)
        test.assert_almost_equal(u_no_tangent, u)

    def test_disc_velocity(self):
        fvw = self.vw("../config/base_{:d}d.json".format(self.dimension))
        q = self.rng.random((fvw.num_states, 1))
        m = self.rng.random(fvw.total_controls)
        ur_no_tangent, dur_dq, dur_dm = fvw.disc_velocity(q, m, with_tangent=False)
        ur, dur_dq, dur_dm = fvw.disc_velocity(q, m, with_tangent=True)
        test.assert_almost_equal(ur_no_tangent, ur)

    def test_update_state(self):
        fvw = self.vw("../config/base_{:d}d.json".format(self.dimension))
        q = self.rng.random((fvw.num_states, 1))
        m = self.rng.random(fvw.total_controls)
        u = self.random_inflow_vector()
        qk_no_tangent, dqn_dq, dqn_dm = fvw.update_state(q.copy(), m, u, with_tangent=False)
        qk, dqn_dq, dqn_dm = fvw.update_state(q.copy(), m, u, with_tangent=True)
        test.assert_almost_equal(qk_no_tangent, qk)  # this does not always pass.
        test.assert_equal(dqn_dq.shape, (fvw.num_states, fvw.num_states))
        test.assert_equal(dqn_dm.shape, (fvw.num_states, fvw.total_controls))
        test.assert_equal(qk.shape, q.shape)

        # todo: test position update - how to?
        X, G, U, M = fvw.states_from_state_vector(q)
        Xk, Gk, Uk, Mk = fvw.states_from_state_vector(qk)

        test.assert_equal(Gk[1:, :], G[:-1, :])
        test.assert_equal(Uk[1:, :], U[:-1, :])
        test.assert_equal(Mk[:, 0], m)

    def test_new_rings_in_update_state(self):
        fvw = self.vw("../config/base_{:d}d.json".format(self.dimension))
        q = self.rng.random((fvw.num_states, 1))
        m = self.rng.random(fvw.total_controls)
        u = self.random_inflow_vector()
        qk, dqn_dq, dqn_dm = fvw.update_state(q.copy(), m, u, with_tangent=False)
        Xk, Gk, Uk, Mk = fvw.states_from_state_vector(qk)

        (X0, G0, U0, M0), derivatives = fvw.new_rings(q.copy(), m, u, with_tangent=False)
        for wt in range(fvw.num_turbines):
            test.assert_equal(Xk[wt * fvw.num_rings], X0[wt])
            test.assert_equal(Gk[wt * fvw.num_rings], G0[wt])
            test.assert_equal(Uk[wt * fvw.num_rings], U0[wt])
            test.assert_equal(Mk[wt * fvw.num_rings], M0[wt])

    def test_calculate_power(self):
        fvw = self.vw("../config/base_{:d}d.json".format(self.dimension))
        q = self.rng.random((fvw.num_states, 1))
        m = self.rng.random(fvw.total_controls)
        p_no_tangent, dp_dq, dp_dm = fvw.calculate_power(q, m, with_tangent=True)
        p, dp_dq, dp_dm = fvw.calculate_power(q, m, with_tangent=True)
        test.assert_almost_equal(p_no_tangent, p)
        test.assert_equal(p.shape, (fvw.total_turbines,))
        test.assert_equal(dp_dq.shape, (fvw.total_turbines, fvw.num_states))
        test.assert_equal(dp_dm.shape, (fvw.total_turbines, fvw.total_controls))

    def test_evaluate_objective_function(self):
        fvw = self.vw("../config/base_{:d}d.json".format(self.dimension))
        n = self.rng.integers(10, 30)
        q = self.rng.random((n, fvw.num_states))
        # dq_dq = self.rng.random((n, fvw.num_states, fvw.num_states))
        # dq_dm = self.rng.random((n, fvw.num_states, fvw.num_states))
        m = self.rng.random((n, fvw.total_controls))
        Q = self.rng.random((n, 1, fvw.total_turbines))
        R = self.rng.random((n, fvw.total_controls, fvw.total_controls))
        phi_no_tangent, dphi_dq, dphi_dm = fvw.evaluate_objective_function(q, m, Q, R, with_tangent=False)
        phi, dphi_dq, dphi_dm = fvw.evaluate_objective_function(q, m, Q, R, with_tangent=True)
        test.assert_equal(phi_no_tangent, phi)


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
        test.assert_equal(self.fvw.num_elements, 2)


    # @unittest.skip("not implemented yet")
    def test_update_state(self):
        super().test_update_state()

    @unittest.skip("not implemented yet")
    def test_new_rings_in_update_state(self):
        super().test_new_rings_in_update_state()

    @unittest.skip("not implemented yet")
    def test_calculate_power(self):
        super().test_calculate_power()

    @unittest.skip("not implemented yet")
    def test_evaluate_objective_function(self):
        super().test_evaluate_objective_function()


# todo: generalise set up
# todo: test magnitude of vortex strength
# todo: perform physical sensibility checks
    # i.e. velocity in/out of wake
    # generate power curves
    #
# todo: robust derivative testing?
if __name__ == '__main__':
    unittest.main()
