import unittest
from vortexwake import vortexwake as vw
import numpy as np
import numpy.testing as test
import matplotlib.pyplot as plt
from functools import partial


def get_colours(n):
    return plt.cm.viridis(np.linspace(0.1, 0.9, n))


def mean_absolute_error(A, B):
    return np.sum(np.abs(A - B)) / (np.product(A.shape))  # / np.sum(np.abs(A))


def construct_jacobian_fd(f, q, m, dq=1e-4, dm=1e-4):
    q0 = q.copy()
    m0 = m.copy().ravel()
    y0 = f(q0, m0)
    df_dq = np.zeros((len(y0), len(q0)))
    df_dm = np.zeros((len(y0), len(m0)))

    for n in range(len(q0)):
        dq0 = np.zeros_like(q0)
        dq0[n] = dq

        y1 = f(q0.copy() - dq0, np.reshape(m0.copy(), m.shape))
        y2 = f(q0.copy() + dq0, np.reshape(m0.copy(), m.shape))

        df_dq[:, n:n + 1] = (y2 - y1) / (2 * dq)

    for n in range(len(m0)):
        dm0 = np.zeros_like(m0)
        dm0[n] = dm

        y1 = f(q0.copy(), np.reshape(m0.copy() - dm0, m.shape))
        y2 = f(q0.copy(), np.reshape(m0.copy() + dm0, m.shape))

        df_dm[:, n:n + 1] = (y2 - y1) / (2 * dm)

    df_dq = np.where(np.isnan(df_dq), 0, df_dq)
    df_dm = np.where(np.isnan(df_dm), 0, df_dm)

    return df_dq, df_dm


class TestDerivatives(unittest.TestCase):

    def setUp(self):
        self.skipTest("derivative tests not implemented")
        self.rng = np.random.default_rng()
        self.vw = None
        self.fvw = None
        self.dimension = None
        self.config = None
        self.threshold = None

    def run_transient(self, num_steps=100):
        states = self.fvw.initialise_states()
        q0 = self.fvw.state_vector_from_states(*states)
        n = num_steps
        m = np.zeros((n, self.fvw.total_controls))
        m[:, self.fvw.induction_idx::self.fvw.num_controls] = 0.25 + 0.02 * self.rng.random((n, self.fvw.num_controls))
        m[:, self.fvw.yaw_idx::self.fvw.num_controls] = 20 + 2 * self.rng.random((n, self.fvw.num_controls))
        u = np.zeros((n, self.fvw.dim)) + self.fvw.unit_vector_x
        qh, _, _ = self.fvw.run_forward(q0, m, u, n, with_tangent=False)
        return qh[-1].copy(), m[-1].copy()

    def print_graphical_derivative_comparison(self, function, state, controls, name):
        def f(q, m):
            return function(q.copy(), m.copy(), with_tangent=False)[0].reshape(-1, 1)

        df_dq_A, df_dm_A = construct_jacobian_fd(f, state.copy(), controls.copy())
        df_dq_B, df_dm_B = function(state.copy(), controls.copy(), with_tangent=True)[1:3]

        df_dq_B = df_dq_B.reshape(-1, self.fvw.num_states)
        df_dm_B = df_dm_B.reshape(-1, self.fvw.total_controls)

        self.print_difference_plots(df_dq_A, df_dq_B, name + "_dq")
        self.print_difference_plots(df_dm_A, df_dm_B, name + "_dm")

        return (df_dq_A, df_dm_A), (df_dq_B, df_dm_B)

    def print_difference_plots(self, A, B, name):
        fig, ax = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(12, 6))
        vmax = 1e-3

        if A.shape[1] < 5 or A.shape[0] < 5:
            if A.shape[0] < A.shape[1]:
                A = A.T
                B = B.T
            num_lines = A.shape[1]
            colours = get_colours(num_lines + 1)
            for n in range(num_lines):
                ax[0].plot(B[:, n], c=colours[n])
                ax[1].plot(A[:, n], c=colours[n])
                ax[2].plot((A - B)[:, n], c=colours[n])
            for a in ax:
                a.grid(True)
        else:
            ax[0].imshow(B, cmap="RdBu", vmin=-vmax, vmax=vmax)
            ax[1].imshow(A, cmap="RdBu", vmin=-vmax, vmax=vmax)
            ax[2].imshow(A - B, cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax[0].set_title("analytical")
        ax[1].set_title("central diff")
        ax[2].set_title("MAE: {:1.2e}".format(mean_absolute_error(A, B)))
        fig.savefig("./figures/{:d}d/derivative_difference_{:s}.png".format(self.dimension, name), format="png",
                    dpi=600)

    @unittest.skip
    def test_new_rings(self):
        assert False

    # @unittest.skip
    def test_disc_velocity(self):
        # def disc_velocity(self, states, controls, with_tangent, all_turbines=False):
        function = partial(self.fvw.disc_velocity, all_turbines=True)
        (df_dq_A, df_dm_A), (df_dq_B, df_dm_B) = self.print_graphical_derivative_comparison(function, self.q0, self.m0,
                                                                                            "disc_velocity")
        threshold = 5e-4
        self.assertLess(mean_absolute_error(df_dq_A, df_dq_B), threshold)
        self.assertLess(mean_absolute_error(df_dm_A, df_dm_B), threshold)

    # @unittest.skip
    def test_velocity(self):
        n = self.rng.integers(5, 10)
        points = self.rng.standard_normal((n, self.dimension))
        points[:, 0] += np.arange(n)
        function = partial(self.fvw.velocity, points=points)
        (df_dq_A, df_dm_A), (df_dq_B, df_dm_B) = self.print_graphical_derivative_comparison(function, self.q0, self.m0,
                                                                                            "velocity")

        threshold = 1e-6
        self.assertLess(mean_absolute_error(df_dq_A, df_dq_B), threshold)
        self.assertLess(mean_absolute_error(df_dm_A, df_dm_B), threshold)

    def test_power(self):
        function = self.fvw.calculate_power  # (q, m, with_tangent=True)
        (df_dq_A, df_dm_A), (df_dq_B, df_dm_B) = self.print_graphical_derivative_comparison(function, self.q0, self.m0,
                                                                                            "power")
        threshold = 1e-6
        self.assertLess(mean_absolute_error(df_dq_A, df_dq_B), threshold)
        self.assertLess(mean_absolute_error(df_dm_A, df_dm_B), threshold)

    # @unittest.skip
    def test_update_state(self):
        # inflow = self.fvw.unit_vector_x
        function = partial(self.fvw.update_state, inflow=self.fvw.unit_vector_x)
        (df_dq_A, df_dm_A), (df_dq_B, df_dm_B) = self.print_graphical_derivative_comparison(function, self.q0, self.m0,
                                                                                            "update_state")

        indices = [self.fvw.X_index_start, self.fvw.G_index_start, self.fvw.U_index_start, self.fvw.M_index_start, -1]
        for row in range(len(indices) - 1):
            ida0 = indices[row]
            ida1 = indices[row + 1]
            for col in range(len(indices) - 1):
                idb0 = indices[col]
                idb1 = indices[col + 1]
                if row < 1 and col < 2:
                    threshold = 1e-3
                else:
                    threshold = 1e-5
                # print(row,col)
                self.assertLess(mean_absolute_error(df_dq_A[ida0:ida1, idb0:idb1], df_dq_B[ida0:ida1, idb0:idb1]),
                                threshold)
        threshold = 1e-5
        self.assertLess(mean_absolute_error(df_dm_A, df_dm_B), threshold)

    # @unittest.skip
    def test_full_gradient(self):
        num_steps = self.rng.integers(5, 10)
        print(num_steps)
        n = num_steps
        m = np.zeros((n, self.fvw.total_controls))
        m[:, self.fvw.induction_idx::self.fvw.num_controls] = 0.25 + 0.02 * self.rng.random(
            (n, self.fvw.num_controls))
        m[:, self.fvw.yaw_idx::self.fvw.num_controls] = 20 + 2 * self.rng.random((n, self.fvw.num_controls))
        u = np.zeros((n, self.fvw.dim)) + self.fvw.unit_vector_x
        qh, dqn_dq, dqn_dm = self.fvw.run_forward(self.q0, m, u, n, with_tangent=True)
        Q = self.rng.random((n+1, 1, self.fvw.total_turbines))
        R = self.rng.random((n+1, self.fvw.total_controls, self.fvw.total_controls))
        for k in range(n+1):
            # todo: test for diagonality of R weight matrix
            R[k] = 1e-3 * np.diag(self.rng.random(self.fvw.total_controls))  # R needs to be diagonal
            # this works
            # Q[k] = np.ones_like(Q[k])
            # R[k] = 0
            # this works
            # Q[k] = np.ones_like(Q[k])
            # R[k] = np.eye(self.fvw.total_controls)
            # this works
            # Q[k] = 0
            # R[k] = np.eye(self.fvw.total_controls)
        phi, dphi_dq, dphi_dm = self.fvw.evaluate_objective_function(qh, m, Q, R, with_tangent=True)
        print(phi.shape, dphi_dq.shape, dphi_dm.shape)
        gradient_B = vw.construct_gradient(dqn_dq, dqn_dm, dphi_dq, dphi_dm)

        y0 = np.sum(phi)

        y1 = np.zeros((num_steps, self.fvw.total_controls))
        y2 = np.zeros((num_steps, self.fvw.total_controls))

        m0 = m.copy()
        dm = 1e-4
        for c in range(self.fvw.total_controls):
            for s in range(num_steps):
                m = m0.copy()
                m[s, c] -= dm
                qh, dqn_dq, dqn_dm = self.fvw.run_forward(self.q0, m, u, n, with_tangent=False)
                y1[s, c] = np.sum(self.fvw.evaluate_objective_function(qh, m, Q, R, with_tangent=False)[0])

                m = m0.copy()
                m[s, c] += dm
                qh, dqn_dq, dqn_dm = self.fvw.run_forward(self.q0, m, u, n, with_tangent=False)
                y2[s, c] = np.sum(self.fvw.evaluate_objective_function(qh, m, Q, R, with_tangent=False)[0])

        gradient_A = (y2 - y1) / (2 * dm)

        # print(y0, y1, y2)
        test.assert_allclose(y1,y0, 10*dm)
        test.assert_allclose(y2, y0, 10 * dm)

        self.print_difference_plots(gradient_A, gradient_B, "full_gradient")
        threshold = 5e-4
        self.assertLess(mean_absolute_error(gradient_A[:, self.fvw.induction_idx::self.fvw.num_controls],
                                            gradient_B[:, self.fvw.induction_idx::self.fvw.num_controls]), threshold)
        self.assertLess(mean_absolute_error(gradient_A[:, self.fvw.yaw_idx::self.fvw.num_controls],
                                            gradient_B[:, self.fvw.yaw_idx::self.fvw.num_controls]), threshold)
        self.assertLess(mean_absolute_error(gradient_A, gradient_B), threshold)

    def test_full_gradient_taylor_expansion(self):
        num_steps = self.rng.integers(5, 10)
        n = num_steps
        m = np.zeros((n, self.fvw.total_controls))
        m[:, self.fvw.induction_idx::self.fvw.num_controls] = 0.25 + 0.02 * self.rng.random(
            (n, self.fvw.num_controls))
        m[:, self.fvw.yaw_idx::self.fvw.num_controls] = 20 + 2 * self.rng.random((n, self.fvw.num_controls))
        u = np.zeros((n, self.fvw.dim)) + self.fvw.unit_vector_x
        qh, dqn_dq, dqn_dm = self.fvw.run_forward(self.q0, m, u, n, with_tangent=True)
        Q = self.rng.random((n+1, 1, self.fvw.total_turbines))
        R = self.rng.random((n+1, self.fvw.total_controls, self.fvw.total_controls))
        for k in range(n):
            # todo: test for diagonality of R weight matrix
            R[k] = 1e-5 * np.diag(self.rng.random(self.fvw.total_controls))  # R needs to be diagonal
        phi, dphi_dq, dphi_dm = self.fvw.evaluate_objective_function(qh, m, Q, R, with_tangent=True)
        gradient = vw.construct_gradient(dqn_dq, dqn_dm, dphi_dq, dphi_dm)
        y0 = np.sum(phi)
        h = 1e-1 / (2 ** np.arange(5))
        dm = np.ones_like(m)
        yv = np.zeros_like(h)
        taylor_remainder = np.zeros_like(h)
        # print
        for k in range(len(h)):
            qh, dqn_dq, dqn_dm = self.fvw.run_forward(self.q0, m + h[k] * dm, u, n, with_tangent=False)
            yv[k] = np.sum(self.fvw.evaluate_objective_function(qh, m, Q, R, with_tangent=False)[0])
            taylor_remainder[k] = yv[k] - y0 - h[k] * (gradient * dm).sum()

        taylor_convergence = taylor_remainder[:-1] / taylor_remainder[1:]
        print(taylor_convergence)
        test.assert_array_less(np.abs(taylor_convergence - 4), 0.4)


class TestDerivatives3D(TestDerivatives):

    def setUp(self):
        self.rng = np.random.default_rng()
        self.vw = vw.VortexWake3D
        self.dimension = 3
        self.fvw = self.vw("../config/base_3d.json")
        self.q0, self.m0 = self.run_transient()
        self.threshold = 5e-4

    # @unittest.skip
    def test_velocity(self):
        super().test_velocity()

    def test_disc_velocity(self):
        super().test_disc_velocity()

    def test_update_state(self):
        super().test_update_state()

    def test_power(self):
        super().test_power()

    def test_full_gradient(self):
        super().test_full_gradient()

    def test_full_gradient_taylor_expansion(self):
        super().test_full_gradient_taylor_expansion()


class TestDerivatives2D(TestDerivatives):

    def setUp(self):
        self.rng = np.random.default_rng()
        self.vw = vw.VortexWake2D
        self.dimension = 2
        self.fvw = self.vw("../config/base_2d.json")
        self.q0, self.m0 = self.run_transient(200)
        self.threshold = 1e-6

    def test_velocity(self):
        super().test_velocity()

    def test_disc_velocity(self):
        super().test_disc_velocity()

    def test_update_state(self):
        super().test_update_state()

    def test_power(self):
        super().test_power()

    def test_full_gradient(self):
        super().test_full_gradient()

    def test_full_gradient_taylor_expansion(self):
        super().test_full_gradient_taylor_expansion()
