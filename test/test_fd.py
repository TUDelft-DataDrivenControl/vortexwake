import unittest
import vortexwake as vw
import numpy as np
import numpy.testing as test
import json
import matplotlib.pyplot as plt
from functools import partial


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

        fig, ax = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(12, 6))
        vmax = 1e-3
        ax[0].imshow(df_dq_B, cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax[1].imshow(df_dq_A, cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax[2].imshow(df_dq_A - df_dq_B, cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax[0].set_title("analytical")
        ax[1].set_title("central diff")
        ax[2].set_title("difference")
        fig.savefig("./figures/adj_fd_{:s}_dq_{:d}.png".format(name, self.dimension), format="png", dpi=600)

        fig, ax = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(12, 6))
        vmax = 1e-3
        ax[0].plot(df_dm_B)
        ax[1].plot(df_dm_A)
        ax[2].plot(df_dm_A - df_dm_B)
        ax[0].set_title("analytical")
        ax[1].set_title("central diff")
        ax[2].set_title("difference")
        fig.savefig("./figures/adj_fd_{:s}_dm_{:d}.png".format(name, self.dimension), format="png", dpi=600)

    @unittest.skip
    def test_new_rings(self):
        assert False

    # @unittest.skip
    def test_disc_velocity(self):
        # def disc_velocity(self, states, controls, with_tangent, all_turbines=False):
        function = partial(self.fvw.disc_velocity, all_turbines=True)
        self.print_graphical_derivative_comparison(function, self.q0, self.m0, "disc_velocity")

    # @unittest.skip
    def test_velocity(self):
        n = self.rng.integers(5, 10)
        points = self.rng.standard_normal((n, self.dimension))
        points[:, 0] += np.arange(n)
        function = partial(self.fvw.velocity, points=points)
        self.print_graphical_derivative_comparison(function, self.q0, self.m0, "velocity")
        # assert False

    def test_power(self):
        function = self.fvw.calculate_power  # (q, m, with_tangent=True)
        self.print_graphical_derivative_comparison(function, self.q0, self.m0, "power")

    # @unittest.skip
    def test_update_state(self):
        # inflow = self.fvw.unit_vector_x
        function = partial(self.fvw.update_state, inflow=self.fvw.unit_vector_x)
        self.print_graphical_derivative_comparison(function, self.q0, self.m0, "update_state")

    @unittest.skip
    def test_full_gradient(self):
        assert False


class TestDerivatives3D(TestDerivatives):

    def setUp(self):
        self.rng = np.random.default_rng()
        self.vw = vw.VortexWake3D
        self.dimension = 3
        self.fvw = self.vw("../config/base_3d.json")
        self.q0, self.m0 = self.run_transient()

    # @unittest.skip
    def test_velocity(self):
        super().test_velocity()

    def test_disc_velocity(self):
        super().test_disc_velocity()

    def test_update_state(self):
        super().test_update_state()


class TestDerivatives2D(TestDerivatives):

    def setUp(self):
        self.rng = np.random.default_rng()
        self.vw = vw.VortexWake2D
        self.dimension = 2
        self.fvw = self.vw("../config/base_2d.json")
        self.q0, self.m0 = self.run_transient(200)

    def test_velocity(self):
        super().test_velocity()

    def test_disc_velocity(self):
        super().test_disc_velocity()

    def test_update_state(self):
        super().test_update_state()

    def test_power(self):
        super().test_power()
