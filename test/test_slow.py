import unittest
import vortexwake as vw
import numpy as np
import numpy.testing as test
import json


class TestVortexWakeSlow(unittest.TestCase):

    def setUp(self):
        self.skipTest("Testing delegated to subclasses")
        self.rng = None  # np.random.default_rng()
        self.vw = None
        self.fvw = None
        self.dimension = None
        self.config = None

    def test_run_forward(self):
        fvw = self.vw("../config/base_{:d}d.json".format(self.dimension))
        q0 = self.rng.random((fvw.num_states, 1))
        n = self.rng.integers(5, 10)
        m = np.zeros((n, fvw.total_controls))
        m[:, fvw.induction_idx::fvw.num_controls] = 0.5 * self.rng.random((n, fvw.num_turbines))
        m[:, fvw.yaw_idx::fvw.num_controls] = 30. * self.rng.random((n, fvw.num_turbines))
        u = 0.1 * self.rng.standard_normal((n, self.dimension)) + fvw.unit_vector_x
        state_history_A, dqn_dq_history, dqn_dm_history = fvw.run_forward(initial_state=q0, control_series=m,
                                                                          inflow_series=u, num_steps=n,
                                                                          with_tangent=False)
        state_history_B, dqn_dq_history, dqn_dm_history = fvw.run_forward(initial_state=q0, control_series=m,
                                                                          inflow_series=u, num_steps=n,
                                                                          with_tangent=True)
        test.assert_equal(state_history_A[0], q0[:, 0])
        test.assert_equal(state_history_B[0], q0[:, 0])
        test.assert_allclose(state_history_A, state_history_B)
        test.assert_equal(state_history_B.shape, (n + 1, fvw.num_states))
        test.assert_equal(dqn_dq_history.shape, (n, fvw.num_states, fvw.num_states))
        test.assert_equal(dqn_dm_history.shape, (n, fvw.num_states, fvw.total_controls))

    def test_construct_gradient(self):
        fvw = self.vw("../config/base_{:d}d.json".format(self.dimension))
        q0 = self.rng.random((fvw.num_states, 1))
        n = self.rng.integers(5, 10)
        m = np.zeros((n+1, fvw.total_controls))
        m[:, fvw.induction_idx::fvw.num_controls] = 0.5 * self.rng.random((n+1, fvw.total_turbines))
        m[:, fvw.yaw_idx::fvw.num_controls] = 30. * self.rng.random((n+1, fvw.total_turbines))
        u = 0.1 * self.rng.standard_normal((n, self.dimension)) + fvw.unit_vector_x
        states, dqn_dq, dqn_dm = fvw.run_forward(initial_state=q0, control_series=m,
                                                 inflow_series=u, num_steps=n,
                                                 with_tangent=True)
        Q = self.rng.random((n+1, 1, fvw.total_turbines))
        R = self.rng.random((n+1, fvw.total_controls, fvw.total_controls))
        phi, dphi_dq, dphi_dm = fvw.evaluate_objective_function(states, m, Q, R, with_tangent=True)
        gradient = vw.construct_gradient(dqn_dq, dqn_dm, dphi_dq, dphi_dm)
        # test.assert_equal(gradient.shape, (n+1, fvw.total_controls))

    # todo: verification of partial derivatives and total gradient

class TestVortexWake3DSlow(TestVortexWakeSlow):

    def setUp(self):
        self.rng = np.random.default_rng()
        # self.config, config_name = generate_random_config(3)
        self.vw = vw.VortexWake3D
        # self.fvw = vw.VortexWake3D(config_name)
        self.dimension = 3


class TestVortexWake2DSlow(TestVortexWakeSlow):

    def setUp(self):
        # self.skipTest("slow methods not implemented for 2D yet")

        self.rng = np.random.default_rng()
        # self.config, config_name = generate_random_config(3)
        self.vw = vw.VortexWake2D
        # self.fvw = vw.VortexWake3D(config_name)
        self.dimension = 2
