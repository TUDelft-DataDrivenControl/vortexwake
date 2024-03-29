import numpy as np

# todo: use 'with' to cover only necessary blocks of code
np.seterr(divide='ignore', invalid='ignore')
import json
from time import time


class VortexWake:
    """Base class for wake simulation with Free-Vortex method.
    2D or 3D, with discrete adjoint for gradients.

    Initialise either `VortexWake2D` or `VortexWake3D` to run.

    Configuration can be specified as a Python dictionary or as a path to a `.json` file.

    Parameters to be configured are:

        - `dimension`, 2 or 3
        - `time_step`, discrete time-step for simulation
        - `num_rings`, number of rings used to model the wake
        - `num_elements`, number of elements per ring - two for 2D, ring discretisation in 3D
        - `vortex_core_size`, parameter for Gaussian smoothing of vortex singularity
        - `num_turbines`, number of turbines to model with free-vortex wake
        - `num_virtual_turbines`, number of turbines to evaluate based on flow velocity
        - `turbine_positions`, list of positions of FVW and virtual turbines

    """

    def __init__(self, config):
        if type(config) is str:
            with open(config, "r") as cf:
                config = json.load(cf)
        if config["dimension"] != self.dim:
            raise ValueError("Trying to instantiate 2D FVW with 3D configuration")
        # self.dim = config["dimension"]
        self.num_elements = config.get("num_elements", 2)
        if self.dim == 2:
            self.num_elements = 2

        self.num_rings = config["num_rings"]
        self.num_points = self.num_elements + 1 if self.dim == 3 else self.num_elements
        self.num_controls = 2
        self.num_turbines = config.get("num_turbines", 1)
        self.num_virtual_turbines = config.get("num_virtual_turbines", 0)
        self.total_turbines = self.num_turbines + self.num_virtual_turbines

        self.turbine_positions = np.array(config.get("turbine_positions", [[0., 0., 0.]]))

        self.total_rings = self.num_rings * self.num_turbines
        self.total_points = self.num_points * self.total_rings
        self.total_elements = self.num_elements * self.total_rings
        self.total_controls = self.num_controls * self.total_turbines
        self.num_states = (self.dim * self.total_points * 2) + self.total_elements + self.total_controls

        self.time_step = config["time_step"]
        self.vortex_core_size = config["vortex_core_size"]
        self.radius = 0.5

        self.X_index_start = 0
        self.X_index_end = self.dim * self.total_points
        self.G_index_start = self.X_index_end
        # todo: reformulate such that single Gamma per ring
        self.G_index_end = self.G_index_start + self.total_elements
        self.U_index_start = self.G_index_end
        self.U_index_end = self.U_index_start + self.dim * self.total_points
        self.M_index_start = self.U_index_end
        self.M_index_end = self.M_index_start + self.total_controls

        self.unit_vector_x = np.zeros(self.dim)
        self.unit_vector_x[0] = 1
        # structure of control vector
        # todo: check number of controls
        self.induction_idx = 0
        self.yaw_idx = 1

    def states_from_state_vector(self, q):
        """Unpack state column vector into state arrays

        :param q: full state vector of size [num_states]
        :return: tuple of state arrays (X,G,U,M)
        """
        X = q[self.X_index_start: self.X_index_end].reshape(self.total_rings, self.num_points, self.dim)
        G = q[self.G_index_start: self.G_index_end].reshape(self.total_rings, self.num_elements, 1)
        U = q[self.U_index_start: self.U_index_end].reshape(self.total_rings, self.num_points, self.dim)
        M = q[self.M_index_start: self.M_index_end].reshape(self.total_controls, 1)
        return X, G, U, M

    def state_vector_from_states(self, X, G, U, M):
        """Pack state arrays into a single state vector

        :param X: vortex element positions [num_rings x num_points x dim]
        :param G: vortex element strengths [num_rings x num_elements]
        :param U: stored inflow velocity [num_rings x num_points x dim]
        :param M: stored controls [total_controls]
        :return: full state vector of size [num_states]
        """
        q = np.zeros((self.num_states, 1))
        q[self.X_index_start:self.X_index_end:self.dim, 0] = X[:, :, 0].ravel()
        q[self.X_index_start + 1:self.X_index_end:self.dim, 0] = X[:, :, 1].ravel()
        if self.dim == 3:
            q[self.X_index_start + 2:self.X_index_end:self.dim, 0] = X[:, :, 2].ravel()

        q[self.G_index_start:self.G_index_end, 0] = G.ravel()

        q[self.U_index_start:self.U_index_end:self.dim, 0] = U[:, :, 0].ravel()
        q[self.U_index_start + 1:self.U_index_end:self.dim, 0] = U[:, :, 1].ravel()
        if self.dim == 3:
            q[self.U_index_start + 2:self.U_index_end:self.dim, 0] = U[:, :, 2].ravel()

        q[self.M_index_start:self.M_index_end, 0] = M.ravel()
        return q

    def initialise_states(self):
        """Initialise states for start of numerical simulation

        :return: tuple of state arrays (X,G,U,M)
        """
        X, G, U, M = self.states_from_state_vector(np.zeros((self.num_states, 1)))
        (X0, G0, U0, M0), derivatives = self.new_rings(np.zeros((self.num_states, 1)), np.zeros(self.total_controls),
                                                       self.unit_vector_x)
        X[::self.num_rings] = X0
        G[::self.num_rings] = G0
        for wt in range(self.num_turbines):
            U[wt * self.num_rings:(wt + 1) * self.num_rings] = U0[wt]
        M[:] = M0
        return X, G, U, M

    def new_rings(self, states, controls, inflow, with_tangent=False):
        """Generate values for new rings to be initialised.

        :param states: state vector [num_states]
        :param controls: control vector [total_controls]
        :param inflow: inflow vector [dim] or [num_points x dim]
        :param with_tangent: boolean, calculate partial derivatives if `True`
        :return: tuple of new ring states (X0, G0, U0, M0),
             tuple of new ring state derivatives
        """
        X0 = np.zeros((self.num_turbines, self.num_points, self.dim))
        G0 = np.zeros((self.num_turbines, self.num_elements, 1))
        U0 = np.zeros((self.num_turbines, self.num_points, self.dim))
        M0 = np.zeros((self.total_controls, 1))

        dX0_dq = None
        dX0_dm = np.zeros((self.num_turbines, self.dim * self.num_points, self.total_controls))

        dG0_dq = np.zeros((self.num_turbines, self.num_states))
        dG0_dm = np.zeros((self.num_turbines, self.total_controls))

        dU0_dq = None
        dU0_dm = None

        dM0_dq = None
        dM0_dm = np.eye(self.total_controls)

        # todo: generalise for flexibility
        a = controls[self.induction_idx::self.num_controls]
        psi = controls[self.yaw_idx::self.num_controls]
        h = self.time_step

        X0[:, :, 1] = self.y0
        if self.dim == 3:
            X0[:, :, 2] = self.z0

        U0[:] = inflow

        M0[:, 0] = controls

        ur, dur_dq, dur_dm = self.disc_velocity(states, controls, with_tangent)
        # print(ur)
        for wt in range(self.num_turbines):
            X0[wt, :] = X0[wt, :] @ self.rot_z(psi[wt]).T
            X0[wt] += self.turbine_positions[wt]

            # thrust_coefficient = (1+a[wt]**2) *  4 * a[wt] / (1 - a[wt])
            # adjustment from Burton Wind Energy Handbook
            ct1 = 2.3
            # ct1 = 2.
            at = 1 - 0.5 * np.sqrt(ct1)
            if a[wt] > at:
                thrust_coefficient = ct1 / (1 - a[wt]) ** 2 - 4 * (np.sqrt(ct1) - 1) / (1 - a[wt])
            else:
                thrust_coefficient = 4 * a[wt] / (1 - a[wt])

            n = self.unit_vector_x @ self.rot_z(psi[wt]).T
            if self.dim == 2:
                G0[wt] = self.time_step * thrust_coefficient * (1 / 2) * (ur[wt].T @ n) ** 2 * np.array([[1], [-1]])
            elif self.dim == 3:
                G0[wt] = self.time_step * thrust_coefficient * (1 / 2) * (ur[wt].T @ n) ** 2

            if with_tangent:
                dn_dpsi = self.unit_vector_x @ self.drot_z_dpsi(psi[wt]).T

                dX0_dm[wt, :, self.yaw_idx + self.num_controls * wt] = np.reshape(
                    X0[wt, :] @ self.drot_z_dpsi(psi[wt]).T,
                    (self.num_points * self.dim,))

                dG0_dur = h * thrust_coefficient * n * (ur[wt].T @ n)
                dG0_dq[wt] = dG0_dur @ dur_dq[wt]

                if a[wt] > at:
                    dct_da = 2 * ct1 / (1 - a[wt]) ** 3 - 4 * (np.sqrt(ct1) - 1) / (1 - a[wt]) ** 2
                else:
                    dct_da = (4 / (1 - a[wt]) ** 2)
                dG0_da = h * (1 / 2) * (ur[wt].T @ n) ** 2 * dct_da
                # dG0_da = h * (1 / 2) * (ur[wt].T @ n) ** 2 * ((4 / (1 - a[wt]) ** 2) * (1+a[wt]**2) +  (4* 2 * a[wt]**2 / (1 - a[wt])))

                dG0_dpsi = h * thrust_coefficient * (ur[wt].T @ n) * (
                        ur[wt].T @ dn_dpsi + n.T @ dur_dm[wt][:, self.yaw_idx + self.num_controls * wt])
                # print(dG0_da, dG0_dpsi)
                dG0_dm[wt, self.induction_idx + self.num_controls * wt] = dG0_da

                # print(self.induction_idx + self.num_controls * wt)
                dG0_dm[wt, self.yaw_idx + self.num_controls * wt] = dG0_dpsi

                # dG0_dq[wt, self.M_index_start + self.induction_idx + self.num_controls * wt] += dG0_da
                # dG0_dq[wt, self.M_index_start + self.yaw_idx + self.num_controls * wt] += dG0_dpsi

        return (X0, G0, U0, M0), ((dX0_dq, dX0_dm), (dG0_dq, dG0_dm), (dU0_dq, dU0_dm), (dM0_dq, dM0_dm))

    def disc_velocity(self, states, controls, with_tangent=False, all_turbines=False):
        """Calculate disc-averaged velocity for the (virtual) turbines in the simulation.

        :param states: full state vector [num_states]
        :param controls: control vector [total_controls]
        :param with_tangent:  boolean, calculate partial derivatives if `True`
        :param all_turbines:  boolean, include virtual turbines if `True`
        :return: disc-averaged velocity [num_turbines x dim],
            partial derivatives
        """
        nt = self.total_turbines if all_turbines else self.num_turbines
        pt = np.zeros((nt,) + self.rotor_disc_points.shape)
        ur = np.zeros((nt, self.dim))
        dur_dq = np.zeros((nt, self.dim, self.num_states))
        dur_dm = np.zeros((nt, self.dim, self.total_controls))
        dur_dp = np.zeros((nt, self.rotor_disc_points.shape[0] * self.dim))
        for wt in range(nt):
            psi = states[self.M_index_start + self.yaw_idx + wt * self.num_controls].squeeze()
            pt[wt] = self.rotor_disc_points @ self.rot_z(psi).T + self.turbine_positions[wt]
            u, du_dq, du_dm, du_dpt = self.velocity(states, controls, pt[wt], with_tangent)
            ur[wt] = np.sum(u * self.rotor_disc_weights, axis=0) / np.sum(self.rotor_disc_weights)

            if with_tangent:
                dur_dq[wt] = (self.rotor_disc_weights_tiled @ du_dq) / np.sum(self.rotor_disc_weights)
                dpt_dpsi = self.rotor_disc_points @ self.drot_z_dpsi(psi).T
                dpt_dpsi = dpt_dpsi.reshape(-1, 1)
                dur_dq[wt, :, self.M_index_start + self.yaw_idx + wt * self.num_controls] += (
                        self.rotor_disc_weights_tiled @ du_dpt @ dpt_dpsi).squeeze()
        return ur, dur_dq, dur_dm

    def run_forward(self, initial_state, control_series, inflow_series, num_steps, with_tangent):
        """Run forward simulation of the Free-Vortex Wake model for a given initial condition, set of controls, inflow
        over time, and number of simulation steps.

        :param initial_state: state vector for start of simulation [num_states]
        :param control_series: control signals over time [num_steps x total_controls]
        :param inflow_series: inflow vector over time [num_steps x dim] or [num_steps x num_points x dim]
        :param num_steps: number of discrete time steps to simulate
        :param with_tangent:  boolean, calculate partial derivatives if `True`
        :return: simulated state trajectory [num_steps+1 x num_states],
            partial derivatives
        """
        state_history = np.zeros((num_steps + 1, self.num_states))
        q = initial_state.copy()
        state_history[0] = q.T

        if with_tangent:  # avoid memory issues for running large problems without tangent
            #todo: add a similar switch to other functions
            dqn_dq_history = np.zeros((num_steps, self.num_states, self.num_states))
            dqn_dm_history = np.zeros((num_steps, self.num_states, self.total_controls))
            for k in range(0, num_steps):
                # t1 = time()
                q, dqn_dq_history[k], dqn_dm_history[k] = self.update_state(q,
                                                                            control_series[k],
                                                                            inflow_series[k],
                                                                            with_tangent)
                state_history[k + 1] = q.T
        else:
            dqn_dq_history = None
            dqn_dm_history = None
            for k in range(0, num_steps):
                q = self.update_state(q,
                                      control_series[k],
                                      inflow_series[k],
                                      with_tangent)[0]
                state_history[k + 1] = q.T
                # q, dqn_dq_history[idx], dqn_dm_history[idx] = update_state_with_tangent(q, controls[idx], inflow[idx])
            # t2 = time()
            # print("Step {:d} out of {:d} in {:.2f}.".format(k, num_steps, t2 - t1))

        return state_history, dqn_dq_history, dqn_dm_history

    def calculate_power(self, states, controls, with_tangent):
        """ Calculate power from turbines simulated with free-vortex wake and virtual turbines

        :param states: full state vector [num_states]
        :param controls: control vector [total_controls]
        :param with_tangent:  boolean, calculate partial derivatives if `True`
        :return: power [total_turbines] and partial derivatives
        """
        X, G, U, M = self.states_from_state_vector(states)
        a = M[self.induction_idx::self.num_controls, 0]
        psi = M[self.yaw_idx::self.num_controls, 0]
        ur, dur_dq, dur_dm = self.disc_velocity(states, controls, with_tangent, all_turbines=True)
        p = np.zeros(self.total_turbines)

        dp_dq = np.zeros((self.total_turbines, self.num_states))
        dp_dm = np.zeros((self.total_turbines, self.total_controls))

        dM_dq = np.zeros((self.total_controls, self.num_states))
        dM_dq[:, self.M_index_start:self.M_index_end] = np.eye(self.total_controls)

        for wt in range(self.total_turbines):
            n = (self.unit_vector_x) @ self.rot_z(psi[wt]).T
            # todo: check power coefficient adjustment
            power_coefficient = np.pi * self.radius ** 2 * (4 * a[wt]) / (1 - a[wt])
            if wt >= self.num_turbines:
                power_coefficient *= (1 - a[wt]) ** 3

            p[wt] = power_coefficient * (1 / 2) * (ur[wt].T @ n) ** 3

            if with_tangent:
                dn_dpsi = (self.unit_vector_x) @ self.drot_z_dpsi(psi[wt]).T
                # dcp_da = 4 * np.pi * self.radius ** 2
                # if wt >= self.num_turbines:
                #     dcp_da = 4 * np.pi * self.radius ** 2 * (1 - a[wt]) ** 3 \
                #              - 4 * a[wt] * np.pi * self.radius ** 2 * 3 * (1 - a[wt]) ** 2
                # else:
                #     dcp_da = 4 * np.pi * self.radius ** 2
                # dcp_da = 4 * np.pi * self.radius ** 2 / (1-a[wt])**2
                if wt >= self.num_turbines:
                    dcp_da = np.pi * self.radius ** 2 * (-1 * 4 * a[wt] * 2 * (1 - a[wt]) + 4 * (1 - a[wt]) ** 2)
                else:
                    dcp_da = np.pi * self.radius ** 2 * (4 / (1 - a[wt]) ** 2)

                da_dq = dM_dq[self.induction_idx + wt * self.num_controls]
                dpsi_dq = dM_dq[self.yaw_idx + wt * self.num_controls]

                dp_dq[wt] = (1 / 2) * (ur[wt].T @ n) ** 3 * dcp_da * da_dq + \
                            power_coefficient * (3 / 2) * (ur[wt].T @ n) ** 2 * \
                            (n @ dur_dq[wt] + (ur[wt].T @ dn_dpsi) * dpsi_dq)

        return p, dp_dq, dp_dm

    def evaluate_objective_function(self, states, controls, Q, R, with_tangent):
        """Evaluate objective function for optimisation

        :param states: state trajectory [num_steps+1 x num_states]
        :param controls: control series [num_steps x total_controls]
        :param Q: output weight [num_steps+1 x total_turbines]
        :param R: input weight [num_steps+1 x total_controls x total_controls]
        :param with_tangent:  boolean, calculate partial derivatives if `True`
        :return: objective function evaluated per simulation step [num_steps] and partial derivatives
        """
        num_steps = states.shape[0] - 1

        m = np.zeros((num_steps + 1, self.total_controls))
        m[:-1] = controls
        # evaluate the change in control value
        dm = np.zeros((num_steps + 1, self.total_controls))
        dm[:-1] = controls - states[:-1, self.M_index_start:self.M_index_end]
        ddm_dq = np.zeros((self.total_controls, self.num_states))
        ddm_dq[:, self.M_index_start: self.M_index_end] = -1 * np.eye(self.total_controls)
        ddm_dm = np.eye(self.total_controls)

        phi = np.zeros(num_steps + 1)
        dphi_dq = np.zeros((num_steps + 1, 1, self.num_states))
        dphi_dm = np.zeros((num_steps + 1, 1, self.total_controls))

        for k in range(num_steps + 1):
            # todo: evaluate power series and evaluate objective without for loop?
            p, dp_dq, dp_dm = self.calculate_power(states[k], m[k], with_tangent)
            phi[k] = Q[k] @ p.T + dm[k].T @ R[k] @ dm[k]

            if with_tangent:
                dphi_dq[k] = Q[k] @ dp_dq + 2 * dm[k].T @ R[k] @ ddm_dq
                dphi_dm[k] = Q[k] @ dp_dm + 2 * dm[k].T @ R[k] @ ddm_dm

        return phi, dphi_dq, dphi_dm


class VortexWake2D(VortexWake):
    """2D Free-Vortex Wake implementation.

    """
    def __init__(self, config_file):
        self.dim = 2

        VortexWake.__init__(self, config_file)
        # self.num_elements = 2
        self.rot_z = rot_z_2d
        self.drot_z_dpsi = drot_z_dpsi_2d

        azimuthal_angles = np.array([0, np.pi])
        self.y0 = self.radius * np.cos(azimuthal_angles)

        r = np.linspace(-0.45, 0.45, 9)
        p = np.zeros((len(r), 2))
        w = np.zeros((len(r), 1))
        p[:, 1] = r
        w[:] = (r[1] - r[0])
        self.rotor_disc_points = p
        self.rotor_disc_weights = w
        self.rotor_disc_weights_tiled = (w @ np.eye(self.dim)[:, None, :]).reshape(self.dim, self.dim * len(w))

    def velocity(self, states, controls, points, with_tangent):
        """Calculation of velocity at specified spatial coordinates.

        :param states: full state vector [num_states]
        :param controls: control vector [total_controls]
        :param points: spatial coordinates to evaluate [? x 2]
        :param with_tangent:  boolean, calculate partial derivatives if `True`
        :return: velocity at each coordinate [? x 2] and partial derivatives
        """
        elements, vortex_strengths, free_flow, saved_controls = self.states_from_state_vector(states)
        inflow_vector = np.reshape(free_flow, (-1, 2))

        ex = np.reshape(elements[:, :, 0], (-1, 1))
        ey = np.reshape(elements[:, :, 1], (-1, 1))
        # ez = np.reshape(elements[:, :, 2], (-1, 1))

        p = np.reshape(points, (-1, 2))
        px = p[:, 0].T
        py = p[:, 1].T
        # pz = p[:, 2].T

        rx = ex - px
        ry = ey - py
        #     rz = np.reshape(np.ravel(e1z),(-1,1)) - pz

        sigmak = self.vortex_core_size
        Gammak = np.reshape(np.ravel(vortex_strengths), (-1, 1))

        r_norm_sq = rx ** 2 + ry ** 2
        r_norm_sq_sq = r_norm_sq ** 2

        u0 = (Gammak / (2 * np.pi * r_norm_sq))
        u0 = np.where(np.isnan(u0), 0, u0)
        u0 = np.where(np.isinf(u0), 0, u0)
        u0x = -ry * u0
        u0y = rx * u0

        u2 = 1 - np.exp(- r_norm_sq / sigmak ** 2)
        u2 = np.where(np.isnan(u2), 0, u2)
        u2 = np.where(np.isinf(u2), 0, u2)
        ui_x = u0x * u2
        ui_y = u0y * u2
        # ui_z = ui_z * p6

        result = np.zeros(p.shape)

        ui_x = np.where(np.isnan(ui_x), 0, ui_x)
        ui_y = np.where(np.isnan(ui_y), 0, ui_y)
        # ui_z = np.where(np.isnan(ui_z), 0, ui_z)
        #

        r_norm = (rx ** 2 + ry ** 2)
        weights = np.exp(-r_norm * 10)
        normalised_weights = weights / weights.sum(axis=0)

        inflow_vector_x = (inflow_vector[:, 0:1] * normalised_weights).sum(axis=0)
        inflow_vector_y = (inflow_vector[:, 1:2] * normalised_weights).sum(axis=0)

        result[:, 0] = ui_x.sum(axis=0) + inflow_vector_x
        result[:, 1] = ui_y.sum(axis=0) + inflow_vector_y
        result = np.reshape(result, points.shape)

        # todo: tangent!
        # P1 = num_points * 2 * num_rings
        # E1 = num_elements * num_rings
        # P = num_turbines * P1
        # E = num_turbines * E1
        n_p = p.shape[0]
        du_dq = np.zeros((n_p * self.dim, self.num_states))
        du_dp = np.zeros((n_p * self.dim, n_p * self.dim))
        if with_tangent:
            ## select subset of full jacobian
            du_dX = du_dq[:, self.X_index_start:self.X_index_end]
            du_dGamma = du_dq[:, self.G_index_start:self.G_index_end]
            du_dU = du_dq[:, self.U_index_start:self.U_index_end]
            du_dM = du_dq[:, self.M_index_start:self.M_index_end]
            ##   dX_dX

            pi = np.pi
            const0 = (-Gammak / (pi * r_norm_sq_sq))
            du0_dr1_x = const0 * rx
            du0_dr1_y = const0 * ry

            const2 = (2 / sigmak ** 2) * np.exp(-r_norm_sq / sigmak ** 2)
            du2_dr1_x = const2 * rx
            du2_dr1_y = const2 * ry

            u02 = u0 * u2
            du_x_dr1_x = 0 - ry * (du0_dr1_x * u2 + du2_dr1_x * u0)
            du_x_dr1_y = -1 * u02 - ry * (du0_dr1_y * u2 + du2_dr1_y * u0)

            du_y_dr1_x = u02 + rx * (du0_dr1_x * u2 + du2_dr1_x * u0)
            du_y_dr1_y = 0 + rx * (du0_dr1_y * u2 + du2_dr1_y * u0)

            du_x_dx0_x = -du_x_dr1_x
            du_x_dx0_y = -du_x_dr1_y

            du_y_dx0_x = -du_y_dr1_x
            du_y_dx0_y = -du_y_dr1_y

            du_x_dx1_x = du_x_dr1_x
            du_x_dx1_y = du_x_dr1_y

            du_y_dx1_x = du_y_dr1_x
            du_y_dx1_y = du_y_dr1_y

            du_x_dx1_x = np.where(np.isnan(du_x_dx1_x), 0, du_x_dx1_x)
            du_x_dx1_y = np.where(np.isnan(du_x_dx1_y), 0, du_x_dx1_y)

            du_y_dx1_x = np.where(np.isnan(du_y_dx1_x), 0, du_y_dx1_x)
            du_y_dx1_y = np.where(np.isnan(du_y_dx1_y), 0, du_y_dx1_y)

            # todo:
            for idx in range(self.num_rings * self.num_turbines):
                du_dX_sub = du_dX[:, idx * self.num_points * 2:(idx + 1) * self.num_points * 2]

                du_dX_sub[0::2, 0:-1:2] += du_x_dx1_x[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                du_dX_sub[1::2, 0:-1:2] += du_y_dx1_x[idx * self.num_elements:(idx + 1) * self.num_elements, :].T

                du_dX_sub[0::2, 1::2] += du_x_dx1_y[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                du_dX_sub[1::2, 1::2] += du_y_dx1_y[idx * self.num_elements:(idx + 1) * self.num_elements, :].T

            nw = normalised_weights
            w = weights
            sw = weights.sum(axis=0)

            # dw_dx_1 =  20* rx *nw
            dw_dx_2 = (nw / sw)[:, :, None] * (-20 * (rx * w).T)  # / (sw[:,None,None]**2)
            nw_sq = 20 * (nw / sw)[:, :, None]

            dw_dx = nw_sq * (rx * w).T
            dw_dy = nw_sq * (ry * w).T
            I = np.eye(self.total_points)
            dw_dx -= np.swapaxes(np.sum(dw_dx[:, :, :], axis=0)[:, :, None] * I, 1, 0)
            dw_dy -= np.swapaxes(np.sum(dw_dy[:, :, :], axis=0)[:, :, None] * I, 1, 0)

            inf_x = inflow_vector[:, 0:1]
            inf_y = inflow_vector[:, 1:2]

            dinf_x_dx = dw_dx.T @ inf_x
            dinf_x_dy = dw_dy.T @ inf_x

            dinf_y_dx = dw_dx.T @ inf_y
            dinf_y_dy = dw_dy.T @ inf_y

            du_dX[0::2, 0::2] += dinf_x_dx.T.squeeze()
            du_dX[0::2, 1::2] += dinf_x_dy.T.squeeze()

            du_dX[1::2, 0::2] += dinf_y_dx.T.squeeze()
            du_dX[1::2, 1::2] += dinf_y_dy.T.squeeze()

            du_dU[0::2, 0::2] = normalised_weights.T
            du_dU[1::2, 1::2] = normalised_weights.T

            ##   du_dGamma
            du_dGamma[0::2, :] = ((-ry / Gammak) * u02).T
            du_dGamma[1::2, :] = ((rx / Gammak) * u02).T

            ## du_dU

            ##   du_dsigma#
            du_dq = np.where(np.isnan(du_dq), 0, du_dq)

            du_dp[0::2, 0::2] = np.diag(np.sum(du_x_dx0_x, axis=0))
            du_dp[0::2, 1::2] = np.diag(np.sum(du_x_dx0_y, axis=0))

            du_dp[1::2, 0::2] = np.diag(np.sum(du_y_dx0_x, axis=0))
            du_dp[1::2, 1::2] = np.diag(np.sum(du_y_dx0_y, axis=0))

        du_dm = np.zeros((n_p * 2, len(controls)))

        return result, du_dq, du_dm, du_dp

    def update_state(self, states, controls, inflow, with_tangent):
        """Update the state vector one discrete time step with given controls and inflow.

        :param states: full state vector [num_states]
        :param controls: control vector [total_controls]
        :param inflow: inflow vector [dim] or [num_points x dim]
        :param with_tangent:  boolean, calculate partial derivatives if `True`
        :return: updated state [num_states] and partial derivatives
        """
        states = states.copy()
        new_states, new_state_derivatives = self.new_rings(states, controls, inflow, with_tangent)
        elements, vortex_strengths, free_flow, saved_controls = self.states_from_state_vector(states)
        inflow_vector = np.reshape(free_flow, (-1, 2))
        points = elements

        ex = np.reshape(elements[:, :, 0], (-1, 1))
        ey = np.reshape(elements[:, :, 1], (-1, 1))
        # ez = np.reshape(elements[:, :, 2], (-1, 1))

        p = np.reshape(points, (-1, 2))
        px = p[:, 0].T
        py = p[:, 1].T
        # pz = p[:, 2].T

        rx = ex - px
        ry = ey - py
        #     rz = np.reshape(np.ravel(e1z),(-1,1)) - pz

        sigmak = self.vortex_core_size
        Gammak = np.reshape(np.ravel(vortex_strengths), (-1, 1))

        r_norm_sq = rx ** 2 + ry ** 2
        r_norm_sq_sq = r_norm_sq ** 2

        u0 = (Gammak / (2 * np.pi * r_norm_sq))
        u0 = np.where(np.isnan(u0), 0, u0)
        u0 = np.where(np.isinf(u0), 0, u0)
        u0x = -ry * u0
        u0y = rx * u0

        u2 = 1 - np.exp(- r_norm_sq / sigmak ** 2)
        u2 = np.where(np.isnan(u2), 0, u2)
        u2 = np.where(np.isinf(u2), 0, u2)
        ui_x = u0x * u2
        ui_y = u0y * u2
        # ui_z = ui_z * p6

        result = np.zeros((self.total_points, 2))

        ui_x = np.where(np.isnan(ui_x), 0, ui_x)
        ui_y = np.where(np.isnan(ui_y), 0, ui_y)
        # ui_z = np.where(np.isnan(ui_z), 0, ui_z)
        #
        normalised_weights = np.eye(self.total_points)
        # print(normalised_weights[:,0])
        inflow_vector_x = (inflow_vector[:, 0:1] * normalised_weights).sum(axis=0)
        inflow_vector_y = (inflow_vector[:, 1:2] * normalised_weights).sum(axis=0)

        result[:, 0] = ui_x.sum(axis=0) + inflow_vector_x
        result[:, 1] = ui_y.sum(axis=0) + inflow_vector_y
        result = np.reshape(result, points.shape)

        points[1:] = points[:-1] + result[:-1] * self.time_step

        (dX0_dq, dX0_dm), (dGamma0_dq, dGamma0_dm), (dU0_dq, dU0_dm), (dM0_dq, dM0_dm) = new_state_derivatives
        (X0, Gamma0, U0, M0) = new_states

        points[0::self.num_rings] = X0

        vortex_strengths[1:] = vortex_strengths[:-1]
        vortex_strengths[0::self.num_rings] = Gamma0

        free_flow[1:] = free_flow[:-1]
        free_flow[0::self.num_rings] = U0

        new_controls = M0

        # todo: tangent!
        P1 = self.num_points * 2 * self.num_rings
        E1 = self.num_elements * self.num_rings
        P = self.num_turbines * P1
        E = self.num_turbines * E1

        dqn_dq = np.zeros((self.num_states, self.num_states))
        if with_tangent:
            ## select subset of full jacobian
            dX_dX = dqn_dq[self.X_index_start:self.X_index_end, self.X_index_start:self.X_index_end]
            dGamma_dX = dqn_dq[self.G_index_start:self.G_index_end, self.X_index_start:self.X_index_end]

            dX_dGamma = dqn_dq[self.X_index_start:self.X_index_end, self.G_index_start:self.G_index_end]
            dGamma_dGamma = dqn_dq[self.G_index_start:self.G_index_end, self.G_index_start:self.G_index_end]
            #     dsigma_dGamma = dqn_dq[P+E:P+2*E,P:P+E]
            dX_dU = dqn_dq[self.X_index_start:self.X_index_end, self.U_index_start:self.U_index_end]
            dU_dX = dqn_dq[self.U_index_start:self.U_index_end, self.X_index_start:self.X_index_end]
            dU_dGamma = dqn_dq[self.U_index_start:self.U_index_end, self.U_index_start:self.U_index_end]
            dU_dU = dqn_dq[self.U_index_start:self.U_index_end, self.U_index_start:self.U_index_end]

            ##   dX_dX
            pi = np.pi
            const0 = (-Gammak / (pi * r_norm_sq_sq))
            du0_dr1_x = const0 * rx
            du0_dr1_y = const0 * ry

            const2 = (2 / sigmak ** 2) * np.exp(-r_norm_sq / sigmak ** 2)
            du2_dr1_x = const2 * rx
            du2_dr1_y = const2 * ry

            u02 = u0 * u2
            du_x_dr1_x = 0 - ry * (du0_dr1_x * u2 + du2_dr1_x * u0)
            du_x_dr1_y = -1 * u02 - ry * (du0_dr1_y * u2 + du2_dr1_y * u0)

            du_y_dr1_x = u02 + rx * (du0_dr1_x * u2 + du2_dr1_x * u0)
            du_y_dr1_y = 0 + rx * (du0_dr1_y * u2 + du2_dr1_y * u0)

            du_x_dx0_x = -du_x_dr1_x
            du_x_dx0_y = -du_x_dr1_y

            du_y_dx0_x = -du_y_dr1_x
            du_y_dx0_y = -du_y_dr1_y

            du_x_dx1_x = du_x_dr1_x
            du_x_dx1_y = du_x_dr1_y

            du_y_dx1_x = du_y_dr1_x
            du_y_dx1_y = du_y_dr1_y

            du_x_dx1_x = np.where(np.isnan(du_x_dx1_x), 0, du_x_dx1_x)
            du_x_dx1_y = np.where(np.isnan(du_x_dx1_y), 0, du_x_dx1_y)

            du_y_dx1_x = np.where(np.isnan(du_y_dx1_x), 0, du_y_dx1_x)
            du_y_dx1_y = np.where(np.isnan(du_y_dx1_y), 0, du_y_dx1_y)

            for idx in range(self.num_rings * self.num_turbines):
                dX_dX_sub = dX_dX[:, idx * self.num_points * 2:(idx + 1) * self.num_points * 2]

                dX_dX_sub[0::2, 0:-1:2] += du_x_dx1_x[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                dX_dX_sub[1::2, 0:-1:2] += du_y_dx1_x[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                #
                dX_dX_sub[0::2, 1::2] += du_x_dx1_y[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                dX_dX_sub[1::2, 1::2] += du_y_dx1_y[idx * self.num_elements:(idx + 1) * self.num_elements, :].T

            du_x_dx0_x = np.where(np.isnan(du_x_dx0_x), 0, du_x_dx0_x)
            du_x_dx0_y = np.where(np.isnan(du_x_dx0_y), 0, du_x_dx0_y)

            du_y_dx0_x = np.where(np.isnan(du_y_dx0_x), 0, du_y_dx0_x)
            du_y_dx0_y = np.where(np.isnan(du_y_dx0_y), 0, du_y_dx0_y)

            dX_dX[0::2, 0::2] += np.diag(np.sum(du_x_dx0_x, 0))
            dX_dX[0::2, 1::2] += np.diag(np.sum(du_x_dx0_y, 0))

            dX_dX[1::2, 0::2] += np.diag(np.sum(du_y_dx0_x, 0))
            dX_dX[1::2, 1::2] += np.diag(np.sum(du_y_dx0_y, 0))
            #
            h = self.time_step
            dX_dX *= h
            dX_dX += np.eye(P)

            dX_dX[2 * self.num_points:] = dX_dX[:-2 * self.num_points]

            for wt in range(self.num_turbines):
                dX_dX[wt * P1: wt * P1 + 2 * self.num_points] = 0
            #
            # ## dX_dGamma
            dX_dGamma[0::2, :] = ((-ry / Gammak) * u02).T
            dX_dGamma[1::2, :] = ((rx / Gammak) * u02).T

            # zero ring does not move, shift dependency down
            dX_dGamma[2 * self.num_points:, :] = h * dX_dGamma[:-2 * self.num_points, :]
            for idx in range(self.num_turbines):
                dX_dGamma[idx * P1:idx * P1 + 2 * self.num_points, :] = 0
            #

            # ## dX_dM
            #

            ## dGamma_dX
            ## dGamma_dGamma
            dGamma_dGamma[:] = np.diag(np.ones(E - self.num_elements), -self.num_elements)
            # off-diagonal, zero ring already zero
            for wt in range(self.num_turbines):
                dGamma_dGamma[wt * E1:wt * E1 + self.num_elements] = 0
            ## dGamma_dsigma
            ## dGamma_dM

            ## dX_dU
            dX_dU[0::2, 0::2] = h * normalised_weights.T
            dX_dU[1::2, 1::2] = h * normalised_weights.T

            dX_dU[2 * self.num_points:] = dX_dU[:-2 * self.num_points]
            for wt in range(self.num_turbines):
                dX_dU[wt * P1:wt * P1 + 2 * self.num_points] = 0

            dU_dU[:] = np.diag(np.ones(P - 2 * self.num_points), -2 * self.num_points)
            ## dM_dX = 0
            ## dM_dGamma = 0
            ## dM_dsigma = 0
            ## dM_dM =0

            # todo
            ## dX0_dq
            ## dGamma0_dq
            for wt in range(self.num_turbines):
                dqn_dq[P + wt * E1 + 0] = dGamma0_dq[wt]
                dqn_dq[P + wt * E1 + 1] = -dGamma0_dq[wt]

            dqn_dq = np.where(np.isnan(dqn_dq), 0, dqn_dq)
        ## dqn_dm
        # dqn_dm[P + wt * E1 + 1] = dGamma0_dm[wt]
        dqn_dm = np.zeros((self.num_states, self.total_controls))
        # for wt in range(num_turbines):
        #     dqn_dm[P + wt * E1 + 0] = -dGamma0_dm[wt]
        #     dqn_dm[wt * P1: wt * P1 + 2 * num_points] = dX0_dm[wt]

        dqn_dm[self.M_index_start:self.M_index_end] = dM0_dm

        dqn_dm[self.G_index_start:self.G_index_end:2 * self.num_rings] = dGamma0_dm
        dqn_dm[self.G_index_start + 1:self.G_index_end:2 * self.num_rings] = dGamma0_dm * -1

        # for wt in range(num_turbines):
        #     dqn_dm[P + wt * E1: P + wt * E1 + num_elements] = dGamma0_dm[wt]
        #     dqn_dm[wt * P1: wt * P1 + 3 * num_points] = dX0_dm[wt]
        #
        # dqn_dm[P + 3 * E:P + 3 * E + num_controls] = dM0_dm

        return self.state_vector_from_states(points, vortex_strengths, free_flow, new_controls), dqn_dq, dqn_dm


class VortexWake3D(VortexWake):
    def __init__(self, config_file):
        self.dim = 3
        VortexWake.__init__(self, config_file)

        self.rot_z = rot_z_3d
        self.drot_z_dpsi = drot_z_dpsi_3d

        azimuthal_angles = np.arange(0., self.num_points) * (2 * np.pi) / self.num_elements
        self.y0 = self.radius * np.cos(azimuthal_angles)
        self.z0 = self.radius * np.sin(azimuthal_angles)

        # radial points
        # define points and weights for rotor averaged velocity
        # theta = np.arange(0, 2 * np.pi, np.pi / 3)
        # r = np.linspace(0.05, 0.45, 3)
        # p = np.zeros((len(r) * len(theta), 3))
        # w = np.zeros((len(r) * len(theta), 1))
        # for idx in range(len(r)):
        #     p[idx * len(theta):(idx + 1) * len(theta), 1] = r[idx] * np.cos(theta)
        #     p[idx * len(theta):(idx + 1) * len(theta), 2] = r[idx] * np.sin(theta)
        #     w[idx * len(theta):(idx + 1) * len(theta)] = r[idx] * (np.pi / 6) * (r[1] - r[0])
        # isocell points
        N1 = 3
        nrings = 3
        i = np.arange(nrings) + 1
        Ni = (2 * i - 1) * N1
        p = np.zeros((Ni.sum(), 3))
        w = np.ones((Ni.sum(),1))
        Ri = (i - 0.5) * (0.5 / nrings)
        N0 = 0
        for r in range(nrings):
            Thi = np.arange(0, 2 * np.pi, 2 * np.pi / Ni[r])
            p[N0:N0 + Ni[r], 1] = Ri[r] * np.cos(Thi)
            p[N0:N0 + Ni[r], 2] = Ri[r] * np.sin(Thi)
            N0 += Ni[r]
        self.rotor_disc_points = p
        self.rotor_disc_weights = w
        self.rotor_disc_weights_tiled = (w @ np.eye(self.dim)[:, None, :]).reshape(self.dim, self.dim * len(w))

    def velocity(self, states, controls, points, with_tangent):
        """Calculation of velocity at specified spatial coordinates.

        :param states: full state vector [num_states]
        :param controls: control vector [total_controls]
        :param points: spatial coordinates to evaluate [? x 3]
        :param with_tangent:  boolean, calculate partial derivatives if `True`
        :return: velocity at each coordinate [? x 3] and partial derivatives
        """
        elements, vortex_strengths, free_flow, saved_controls = self.states_from_state_vector(
            states)

        inflow_vector = np.reshape(free_flow, (-1, 3))

        ex = elements[:, :, 0]
        ey = elements[:, :, 1]
        ez = elements[:, :, 2]

        e1x = ex[:, :-1]
        e1y = ey[:, :-1]
        e1z = ez[:, :-1]

        e2x = ex[:, 1:]
        e2y = ey[:, 1:]
        e2z = ez[:, 1:]

        r0x = e2x - e1x
        r0y = e2y - e1y
        r0z = e2z - e1z

        r0x = np.reshape(r0x, (-1, 1))
        r0y = np.reshape(r0y, (-1, 1))
        r0z = np.reshape(r0z, (-1, 1))

        p = np.reshape(points, (-1, 3))
        px = p[:, 0].T
        py = p[:, 1].T
        pz = p[:, 2].T

        rx = np.reshape(np.ravel(ex), (-1, 1)) - px
        ry = np.reshape(np.ravel(ey), (-1, 1)) - py
        rz = np.reshape(np.ravel(ez), (-1, 1)) - pz

        r1x = np.reshape(np.ravel(e1x), (-1, 1)) - px
        r1y = np.reshape(np.ravel(e1y), (-1, 1)) - py
        r1z = np.reshape(np.ravel(e1z), (-1, 1)) - pz

        r2x = np.reshape(np.ravel(e2x), (-1, 1)) - px
        r2y = np.reshape(np.ravel(e2y), (-1, 1)) - py
        r2z = np.reshape(np.ravel(e2z), (-1, 1)) - pz

        r1x = np.where(np.abs(r1x) < 1e-9, 0, r1x)
        r1y = np.where(np.abs(r1y) < 1e-9, 0, r1y)
        r1z = np.where(np.abs(r1z) < 1e-9, 0, r1z)

        r2x = np.where(np.abs(r2x) < 1e-9, 0, r2x)
        r2y = np.where(np.abs(r2y) < 1e-9, 0, r2y)
        r2z = np.where(np.abs(r2z) < 1e-9, 0, r2z)

        cross_r1_r2_x = r1y * r2z - r1z * r2y
        cross_r1_r2_y = r1z * r2x - r1x * r2z
        cross_r1_r2_z = r1x * r2y - r1y * r2x
        cross_r1_r2_sq_x = cross_r1_r2_x ** 2
        cross_r1_r2_sq_y = cross_r1_r2_y ** 2
        cross_r1_r2_sq_z = cross_r1_r2_z ** 2

        sum_cross_r1_r2_sq = cross_r1_r2_sq_x + cross_r1_r2_sq_y + cross_r1_r2_z
        sum_cross_r1_r2_sq_sq = sum_cross_r1_r2_sq ** 2

        cross_r1_r2_norm_sq = cross_r1_r2_x ** 2 + cross_r1_r2_y ** 2 + cross_r1_r2_z ** 2
        cross_r1_r2_norm = np.sqrt(cross_r1_r2_norm_sq)

        Gammak = np.reshape(np.ravel(vortex_strengths), (-1, 1))
        #     Gammak = q[P:P+E]
        u0x = (Gammak / (4 * np.pi)) * cross_r1_r2_x / cross_r1_r2_norm_sq
        u0y = (Gammak / (4 * np.pi)) * cross_r1_r2_y / cross_r1_r2_norm_sq
        u0z = (Gammak / (4 * np.pi)) * cross_r1_r2_z / cross_r1_r2_norm_sq
        u0x = np.where(np.isnan(u0x), 0, u0x)
        u0y = np.where(np.isnan(u0y), 0, u0y)
        u0z = np.where(np.isnan(u0z), 0, u0z)

        r1_norm_sq = r1x ** 2 + r1y ** 2 + r1z ** 2
        r1_norm = np.sqrt(r1_norm_sq)
        r2_norm_sq = r2x ** 2 + r2y ** 2 + r2z ** 2
        r2_norm = np.sqrt(r2_norm_sq)
        p2 = (r0x * r1x + r0y * r1y + r0z * r1z) / r1_norm
        p3 = (r0x * r2x + r0y * r2y + r0z * r2z) / r2_norm
        p2 = np.where(np.isnan(p2), 0, p2)
        p3 = np.where(np.isnan(p3), 0, p3)
        u1 = p2 - p3
        u1 = np.where(np.isnan(u1), 0, u1)

        ui_x = u0x * u1
        ui_y = u0y * u1
        ui_z = u0z * u1

        r0_norm_sq = r0x ** 2 + r0y ** 2 + r0z ** 2
        r0_norm = np.sqrt(r0_norm_sq)

        # sigmak = np.reshape(np.ravel(vortex_core_sizes), (-1, 1))
        sigmak = self.vortex_core_size
        #     sigmak = q[P+E:P+2*E]
        u2 = 1 - np.exp(- cross_r1_r2_norm_sq / (sigmak ** 2 * r0_norm_sq))
        u2 = np.where(np.isnan(u2), 0, u2)

        ui_x = ui_x * u2
        ui_y = ui_y * u2
        ui_z = ui_z * u2

        result = np.zeros(p.shape)

        ui_x = np.where(np.isnan(ui_x), 0, ui_x)
        ui_y = np.where(np.isnan(ui_y), 0, ui_y)
        ui_z = np.where(np.isnan(ui_z), 0, ui_z)

        # todo: check
        r_norm = (rx ** 2 + ry ** 2 + rz ** 2)
        weights = np.exp(-r_norm * 10)
        normalised_weights = weights / weights.sum(axis=0)

        inflow_vector_x = (inflow_vector[:, 0:1] * normalised_weights).sum(axis=0)
        inflow_vector_y = (inflow_vector[:, 1:2] * normalised_weights).sum(axis=0)
        inflow_vector_z = (inflow_vector[:, 2:] * normalised_weights).sum(axis=0)

        result[:, 0] = ui_x.sum(axis=0) + inflow_vector_x
        result[:, 1] = ui_y.sum(axis=0) + inflow_vector_y
        result[:, 2] = ui_z.sum(axis=0) + inflow_vector_z

        n_p = p.shape[0]
        du_dq = np.zeros((n_p * 3, self.num_states))
        du_dp = np.zeros((n_p * 3, n_p * 3))
        if with_tangent:
            ## select subset of full jacobian
            du_dX = du_dq[:, self.X_index_start:self.X_index_end]
            du_dGamma = du_dq[:, self.G_index_start:self.G_index_end]
            # du_dsigma = du_dq[:, P + E:P + 2 * E]
            du_dU = du_dq[:, self.U_index_start:self.U_index_end]
            du_dM = du_dq[:, self.M_index_start:self.M_index_end]

            # du0_dr0 = 0
            pi = np.pi
            cross_r1_r2_x = r1y * r2z - r1z * r2y

            cross_r1_r2_y = r1z * r2x - r1x * r2z

            cross_r1_r2_z = r1x * r2y - r1y * r2x

            cross_r2_r1_x = r1x * r2z - r1z * r2x

            cross_r1_r2_x_sq = (cross_r1_r2_x) ** 2

            cross_r1_r2_y_sq = (cross_r1_r2_y) ** 2

            cross_r1_r2_z_sq = (cross_r1_r2_z) ** 2

            cross_r2_r1_x_sq = (cross_r2_r1_x) ** 2

            sum_cross_r1_r2_sq_sq = (cross_r1_r2_z_sq + cross_r2_r1_x_sq + cross_r1_r2_x_sq) ** 2

            const = Gammak / (4 * pi * sum_cross_r1_r2_sq_sq)

            diag = cross_r1_r2_norm_sq

            du0_x_dr1_x = const * (-cross_r1_r2_z_sq - cross_r2_r1_x_sq - cross_r1_r2_x_sq - 2 * (cross_r1_r2_x) * (
                    r2y * (cross_r1_r2_z) + r2z * (cross_r2_r1_x)) + diag)

            du0_x_dr1_y = const * (2 * r2x * (cross_r1_r2_z) * (cross_r1_r2_x) + r2z * (
                    cross_r1_r2_z_sq + cross_r2_r1_x_sq - cross_r1_r2_x_sq))

            du0_x_dr1_z = const * (2 * r2x * (cross_r2_r1_x) * (cross_r1_r2_x) - r2y * (
                    cross_r1_r2_z_sq + cross_r2_r1_x_sq - cross_r1_r2_x_sq))

            du0_y_dr1_x = const * (2 * r2y * (cross_r1_r2_z) * (cross_r2_r1_x) - r2z * (
                    cross_r1_r2_z_sq - cross_r2_r1_x_sq + cross_r1_r2_x_sq))

            du0_y_dr1_y = const * (-cross_r1_r2_z_sq - cross_r2_r1_x_sq + 2 * (cross_r2_r1_x) * (
                    -r2x * (cross_r1_r2_z) + r2z * (cross_r1_r2_x)) - cross_r1_r2_x_sq + diag)

            du0_y_dr1_z = const * (
                    r2x * (cross_r1_r2_z_sq - cross_r2_r1_x_sq + cross_r1_r2_x_sq) - 2 * r2y * (cross_r2_r1_x) * (
                cross_r1_r2_x))

            du0_z_dr1_x = const * (
                    r2y * (-cross_r1_r2_z_sq + cross_r2_r1_x_sq + cross_r1_r2_x_sq) - 2 * r2z * (cross_r1_r2_z) * (
                cross_r2_r1_x))

            du0_z_dr1_y = const * (
                    -r2x * (-cross_r1_r2_z_sq + cross_r2_r1_x_sq + cross_r1_r2_x_sq) - 2 * r2z * (cross_r1_r2_z) * (
                cross_r1_r2_x))

            du0_z_dr1_z = const * (-cross_r1_r2_z_sq + 2 * (cross_r1_r2_z) * (
                    r2x * (cross_r2_r1_x) + r2y * (cross_r1_r2_x)) - cross_r2_r1_x_sq - cross_r1_r2_x_sq + diag)

            du0_x_dr2_x = const * (-cross_r1_r2_z_sq - cross_r2_r1_x_sq - cross_r1_r2_x_sq + 2 * (cross_r1_r2_x) * (
                    r1y * (cross_r1_r2_z) + r1z * (cross_r2_r1_x)) + diag)

            du0_x_dr2_y = const * (-2 * r1x * (cross_r1_r2_z) * (cross_r1_r2_x) - r1z * (
                    cross_r1_r2_z_sq + cross_r2_r1_x_sq - cross_r1_r2_x_sq))

            du0_x_dr2_z = const * (-2 * r1x * (cross_r2_r1_x) * (cross_r1_r2_x) + r1y * (
                    cross_r1_r2_z_sq + cross_r2_r1_x_sq - cross_r1_r2_x_sq))

            du0_y_dr2_x = const * (-2 * r1y * (cross_r1_r2_z) * (cross_r2_r1_x) + r1z * (
                    cross_r1_r2_z_sq - cross_r2_r1_x_sq + cross_r1_r2_x_sq))

            du0_y_dr2_y = const * (-cross_r1_r2_z_sq - cross_r2_r1_x_sq + 2 * (cross_r2_r1_x) * (
                    r1x * (cross_r1_r2_z) - r1z * (cross_r1_r2_x)) - cross_r1_r2_x_sq + diag)

            du0_y_dr2_z = const * (
                    -r1x * (cross_r1_r2_z_sq - cross_r2_r1_x_sq + cross_r1_r2_x_sq) + 2 * r1y * (cross_r2_r1_x) * (
                cross_r1_r2_x))

            du0_z_dr2_x = const * (
                    -r1y * (-cross_r1_r2_z_sq + cross_r2_r1_x_sq + cross_r1_r2_x_sq) + 2 * r1z * (cross_r1_r2_z) * (
                cross_r2_r1_x))

            du0_z_dr2_y = const * (
                    r1x * (-cross_r1_r2_z_sq + cross_r2_r1_x_sq + cross_r1_r2_x_sq) + 2 * r1z * (cross_r1_r2_z) * (
                cross_r1_r2_x))

            du0_z_dr2_z = const * (-cross_r1_r2_z_sq - 2 * (cross_r1_r2_z) * (
                    r1x * (cross_r2_r1_x) + r1y * (cross_r1_r2_x)) - cross_r2_r1_x_sq - cross_r1_r2_x_sq + diag)

            du0_x_dr1_x = np.where(np.isnan(du0_x_dr1_x), 0, du0_x_dr1_x)
            du0_x_dr1_y = np.where(np.isnan(du0_x_dr1_y), 0, du0_x_dr1_y)
            du0_x_dr1_z = np.where(np.isnan(du0_x_dr1_z), 0, du0_x_dr1_z)

            du0_y_dr1_x = np.where(np.isnan(du0_y_dr1_x), 0, du0_y_dr1_x)
            du0_y_dr1_y = np.where(np.isnan(du0_y_dr1_y), 0, du0_y_dr1_y)
            du0_y_dr1_z = np.where(np.isnan(du0_y_dr1_z), 0, du0_y_dr1_z)

            du0_z_dr1_x = np.where(np.isnan(du0_z_dr1_x), 0, du0_z_dr1_x)
            du0_z_dr1_y = np.where(np.isnan(du0_z_dr1_y), 0, du0_z_dr1_y)
            du0_z_dr1_z = np.where(np.isnan(du0_z_dr1_z), 0, du0_z_dr1_z)

            du0_x_dr2_x = np.where(np.isnan(du0_x_dr2_x), 0, du0_x_dr2_x)
            du0_x_dr2_y = np.where(np.isnan(du0_x_dr2_y), 0, du0_x_dr2_y)
            du0_x_dr2_z = np.where(np.isnan(du0_x_dr2_z), 0, du0_x_dr2_z)

            du0_y_dr2_x = np.where(np.isnan(du0_y_dr2_x), 0, du0_y_dr2_x)
            du0_y_dr2_y = np.where(np.isnan(du0_y_dr2_y), 0, du0_y_dr2_y)
            du0_y_dr2_z = np.where(np.isnan(du0_y_dr2_z), 0, du0_y_dr2_z)

            du0_z_dr2_x = np.where(np.isnan(du0_z_dr2_x), 0, du0_z_dr2_x)
            du0_z_dr2_y = np.where(np.isnan(du0_z_dr2_y), 0, du0_z_dr2_y)
            du0_z_dr2_z = np.where(np.isnan(du0_z_dr2_z), 0, du0_z_dr2_z)

            du1_dr0_x = r1x / r1_norm - r2x / r2_norm
            du1_dr0_y = r1y / r1_norm - r2y / r2_norm
            du1_dr0_z = r1z / r1_norm - r2z / r2_norm

            dot_r0_r1 = r0x * r1x + r0y * r1y + r0z * r1z
            r1_norm_cube = r1_norm ** 3
            du1_dr1_x = (r1_norm_sq * r0x - dot_r0_r1 * r1x) / r1_norm_cube
            du1_dr1_y = (r1_norm_sq * r0y - dot_r0_r1 * r1y) / r1_norm_cube
            du1_dr1_z = (r1_norm_sq * r0z - dot_r0_r1 * r1z) / r1_norm_cube

            dot_r0_r2 = r0x * r2x + r0y * r2y + r0z * r2z
            r2_norm_cube = r2_norm ** 3
            du1_dr2_x = -(r2_norm_sq * r0x - dot_r0_r2 * r2x) / r2_norm_cube
            du1_dr2_y = -(r2_norm_sq * r0y - dot_r0_r2 * r2y) / r2_norm_cube
            du1_dr2_z = -(r2_norm_sq * r0z - dot_r0_r2 * r2z) / r2_norm_cube

            du1_dr0_x = np.where(np.isnan(du1_dr0_x), 0, du1_dr0_x)
            du1_dr0_y = np.where(np.isnan(du1_dr0_y), 0, du1_dr0_y)
            du1_dr0_z = np.where(np.isnan(du1_dr0_z), 0, du1_dr0_z)

            du1_dr1_x = np.where(np.isnan(du1_dr1_x), 0, du1_dr1_x)
            du1_dr1_y = np.where(np.isnan(du1_dr1_y), 0, du1_dr1_y)
            du1_dr1_z = np.where(np.isnan(du1_dr1_z), 0, du1_dr1_z)

            du1_dr2_x = np.where(np.isnan(du1_dr2_x), 0, du1_dr2_x)
            du1_dr2_y = np.where(np.isnan(du1_dr2_y), 0, du1_dr2_y)
            du1_dr2_z = np.where(np.isnan(du1_dr2_z), 0, du1_dr2_z)

            du2_dr0_c = (-np.exp(-cross_r1_r2_norm_sq / (sigmak ** 2 * r0_norm_sq)) *
                         (2 * cross_r1_r2_norm_sq / (sigmak ** 2 * r0_norm_sq ** 2)))
            du2_dr0_c = np.where(np.isnan(du2_dr0_c), 0, du2_dr0_c)

            du2_dr0_x = du2_dr0_c * r0x
            du2_dr0_y = du2_dr0_c * r0y
            du2_dr0_z = du2_dr0_c * r0z

            du2_dr12_c = (np.exp(-cross_r1_r2_norm_sq / (sigmak ** 2 * r0_norm_sq)) *
                          (2 / (sigmak ** 2 * r0_norm_sq)))
            du2_dr12_c = np.where(np.isnan(du2_dr12_c), 0, du2_dr12_c)
            #     du2_dr1_x = du2_dr12_c * (r2y * cross_r1_r2_z - cross_r1_r2_y * r2z)
            #     du2_dr1_y = du2_dr12_c * (r2z * cross_r1_r2_x - cross_r1_r2_z * r2x)
            #     du2_dr1_z = du2_dr12_c * (r2x * cross_r1_r2_y - cross_r1_r2_x * r2y)

            #     du2_dr2_x = du2_dr12_c * (r1z * cross_r1_r2_y - cross_r1_r2_z * r1y)
            #     du2_dr2_y = du2_dr12_c * (r1x * cross_r1_r2_z - cross_r1_r2_x * r1z)
            #     du2_dr2_z = du2_dr12_c * (r1y * cross_r1_r2_x - cross_r1_r2_y * r1x)
            du2_dr1_x = du2_dr12_c * (r2y * (cross_r1_r2_z) + r2z * (-cross_r1_r2_y))
            du2_dr1_y = du2_dr12_c * (-r2x * (cross_r1_r2_z) + r2z * (cross_r1_r2_x))
            du2_dr1_z = du2_dr12_c * (-r2x * (-cross_r1_r2_y) - r2y * (cross_r1_r2_x))

            du2_dr2_x = du2_dr12_c * (-r1y * (cross_r1_r2_z) - r1z * (-cross_r1_r2_y))
            du2_dr2_y = du2_dr12_c * (r1x * (cross_r1_r2_z) - r1z * (cross_r1_r2_x))
            du2_dr2_z = du2_dr12_c * (r1x * (-cross_r1_r2_y) + r1y * (cross_r1_r2_x))

            du_x_dr0_x = u0x * du1_dr0_x * u2 + u0x * u1 * du2_dr0_x
            du_x_dr0_y = u0x * du1_dr0_y * u2 + u0x * u1 * du2_dr0_y
            du_x_dr0_z = u0x * du1_dr0_z * u2 + u0x * u1 * du2_dr0_z

            du_y_dr0_x = u0y * du1_dr0_x * u2 + u0y * u1 * du2_dr0_x
            du_y_dr0_y = u0y * du1_dr0_y * u2 + u0y * u1 * du2_dr0_y
            du_y_dr0_z = u0y * du1_dr0_z * u2 + u0y * u1 * du2_dr0_z

            du_z_dr0_x = u0z * du1_dr0_x * u2 + u0z * u1 * du2_dr0_x
            du_z_dr0_y = u0z * du1_dr0_y * u2 + u0z * u1 * du2_dr0_y
            du_z_dr0_z = u0z * du1_dr0_z * u2 + u0z * u1 * du2_dr0_z

            du_x_dr1_x = du0_x_dr1_x * u1 * u2 + u0x * du1_dr1_x * u2 + u0x * u1 * du2_dr0_x
            du_x_dr1_y = du0_x_dr1_y * u1 * u2 + u0x * du1_dr1_y * u2 + u0x * u1 * du2_dr0_y
            du_x_dr1_z = du0_x_dr1_z * u1 * u2 + u0x * du1_dr1_z * u2 + u0x * u1 * du2_dr0_z

            du_y_dr1_x = du0_y_dr1_x * u1 * u2 + u0y * du1_dr1_x * u2 + u0y * u1 * du2_dr1_x
            du_y_dr1_y = du0_y_dr1_y * u1 * u2 + u0y * du1_dr1_y * u2 + u0y * u1 * du2_dr1_y
            du_y_dr1_z = du0_y_dr1_z * u1 * u2 + u0y * du1_dr1_z * u2 + u0y * u1 * du2_dr1_z

            du_z_dr1_x = du0_z_dr1_x * u1 * u2 + u0z * du1_dr1_x * u2 + u0z * u1 * du2_dr1_x
            du_z_dr1_y = du0_z_dr1_y * u1 * u2 + u0z * du1_dr1_y * u2 + u0z * u1 * du2_dr1_y
            du_z_dr1_z = du0_z_dr1_z * u1 * u2 + u0z * du1_dr1_z * u2 + u0z * u1 * du2_dr1_z

            du_x_dr2_x = du0_x_dr2_x * u1 * u2 + u0x * du1_dr2_x * u2 + u0x * u1 * du2_dr2_x
            du_x_dr2_y = du0_x_dr2_y * u1 * u2 + u0x * du1_dr2_y * u2 + u0x * u1 * du2_dr2_y
            du_x_dr2_z = du0_x_dr2_z * u1 * u2 + u0x * du1_dr2_z * u2 + u0x * u1 * du2_dr2_z

            du_y_dr2_x = du0_y_dr2_x * u1 * u2 + u0y * du1_dr2_x * u2 + u0y * u1 * du2_dr2_x
            du_y_dr2_y = du0_y_dr2_y * u1 * u2 + u0y * du1_dr2_y * u2 + u0y * u1 * du2_dr2_y
            du_y_dr2_z = du0_y_dr2_z * u1 * u2 + u0y * du1_dr2_z * u2 + u0y * u1 * du2_dr2_z

            du_z_dr2_x = du0_z_dr2_x * u1 * u2 + u0z * du1_dr2_x * u2 + u0z * u1 * du2_dr2_x
            du_z_dr2_y = du0_z_dr2_y * u1 * u2 + u0z * du1_dr2_y * u2 + u0z * u1 * du2_dr2_y
            du_z_dr2_z = du0_z_dr2_z * u1 * u2 + u0z * du1_dr2_z * u2 + u0z * u1 * du2_dr2_z

            du_x_dx0_x = -du_x_dr1_x - du_x_dr2_x
            du_x_dx0_y = -du_x_dr1_y - du_x_dr2_y
            du_x_dx0_z = -du_x_dr1_z - du_x_dr2_z

            du_y_dx0_x = -du_y_dr1_x - du_y_dr2_x
            du_y_dx0_y = -du_y_dr1_y - du_y_dr2_y
            du_y_dx0_z = -du_y_dr1_z - du_y_dr2_z

            du_z_dx0_x = -du_z_dr1_x - du_z_dr2_x
            du_z_dx0_y = -du_z_dr1_y - du_z_dr2_y
            du_z_dx0_z = -du_z_dr1_z - du_z_dr2_z

            du_x_dx1_x = du_x_dr1_x - du_x_dr0_x
            du_x_dx1_y = du_x_dr1_y - du_x_dr0_y
            du_x_dx1_z = du_x_dr1_z - du_x_dr0_z

            du_y_dx1_x = du_y_dr1_x - du_y_dr0_x
            du_y_dx1_y = du_y_dr1_y - du_y_dr0_y
            du_y_dx1_z = du_y_dr1_z - du_y_dr0_z

            du_z_dx1_x = du_z_dr1_x - du_z_dr0_x
            du_z_dx1_y = du_z_dr1_y - du_z_dr0_y
            du_z_dx1_z = du_z_dr1_z - du_z_dr0_z

            du_x_dx2_x = du_x_dr2_x + du_x_dr0_x
            du_x_dx2_y = du_x_dr2_y + du_x_dr0_y
            du_x_dx2_z = du_x_dr2_z + du_x_dr0_z

            du_y_dx2_x = du_y_dr2_x + du_y_dr0_x
            du_y_dx2_y = du_y_dr2_y + du_y_dr0_y
            du_y_dx2_z = du_y_dr2_z + du_y_dr0_z

            du_z_dx2_x = du_z_dr2_x + du_z_dr0_x
            du_z_dx2_y = du_z_dr2_y + du_z_dr0_y
            du_z_dx2_z = du_z_dr2_z + du_z_dr0_z

            du_x_dx1_x = np.where(np.isnan(du_x_dx1_x), 0, du_x_dx1_x)
            du_x_dx1_y = np.where(np.isnan(du_x_dx1_y), 0, du_x_dx1_y)
            du_x_dx1_z = np.where(np.isnan(du_x_dx1_z), 0, du_x_dx1_z)

            du_y_dx1_x = np.where(np.isnan(du_y_dx1_x), 0, du_y_dx1_x)
            du_y_dx1_y = np.where(np.isnan(du_y_dx1_y), 0, du_y_dx1_y)
            du_y_dx1_z = np.where(np.isnan(du_y_dx1_z), 0, du_y_dx1_z)

            du_z_dx1_x = np.where(np.isnan(du_z_dx1_x), 0, du_z_dx1_x)
            du_z_dx1_y = np.where(np.isnan(du_z_dx1_y), 0, du_z_dx1_y)
            du_z_dx1_z = np.where(np.isnan(du_z_dx1_z), 0, du_z_dx1_z)

            du_x_dx2_x = np.where(np.isnan(du_x_dx2_x), 0, du_x_dx2_x)
            du_x_dx2_y = np.where(np.isnan(du_x_dx2_y), 0, du_x_dx2_y)
            du_x_dx2_z = np.where(np.isnan(du_x_dx2_z), 0, du_x_dx2_z)

            du_y_dx2_x = np.where(np.isnan(du_y_dx2_x), 0, du_y_dx2_x)
            du_y_dx2_y = np.where(np.isnan(du_y_dx2_y), 0, du_y_dx2_y)
            du_y_dx2_z = np.where(np.isnan(du_y_dx2_z), 0, du_y_dx2_z)

            du_z_dx2_x = np.where(np.isnan(du_z_dx2_x), 0, du_z_dx2_x)
            du_z_dx2_y = np.where(np.isnan(du_z_dx2_y), 0, du_z_dx2_y)
            du_z_dx2_z = np.where(np.isnan(du_z_dx2_z), 0, du_z_dx2_z)

            for idx in range(self.num_rings * self.num_turbines):
                du_dX_sub = du_dX[:, idx * self.num_points * 3:(idx + 1) * self.num_points * 3]

                du_dX_sub[0::3, 0:-3:3] += du_x_dx1_x[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                du_dX_sub[1::3, 0:-3:3] += du_y_dx1_x[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                du_dX_sub[2::3, 0:-3:3] += du_z_dx1_x[idx * self.num_elements:(idx + 1) * self.num_elements, :].T

                du_dX_sub[0::3, 1:-2:3] += du_x_dx1_y[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                du_dX_sub[1::3, 1:-2:3] += du_y_dx1_y[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                du_dX_sub[2::3, 1:-2:3] += du_z_dx1_y[idx * self.num_elements:(idx + 1) * self.num_elements, :].T

                du_dX_sub[0::3, 2:-1:3] += du_x_dx1_z[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                du_dX_sub[1::3, 2:-1:3] += du_y_dx1_z[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                du_dX_sub[2::3, 2:-1:3] += du_z_dx1_z[idx * self.num_elements:(idx + 1) * self.num_elements, :].T

                du_dX_sub[0::3, 3::3] += du_x_dx2_x[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                du_dX_sub[1::3, 3::3] += du_y_dx2_x[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                du_dX_sub[2::3, 3::3] += du_z_dx2_x[idx * self.num_elements:(idx + 1) * self.num_elements, :].T

                du_dX_sub[0::3, 4::3] += du_x_dx2_y[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                du_dX_sub[1::3, 4::3] += du_y_dx2_y[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                du_dX_sub[2::3, 4::3] += du_z_dx2_y[idx * self.num_elements:(idx + 1) * self.num_elements, :].T

                du_dX_sub[0::3, 5::3] += du_x_dx2_z[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                du_dX_sub[1::3, 5::3] += du_y_dx2_z[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                du_dX_sub[2::3, 5::3] += du_z_dx2_z[idx * self.num_elements:(idx + 1) * self.num_elements, :].T

            # du_dX
            nw = normalised_weights
            w = weights
            sw = weights.sum(axis=0)

            # dw_dx_1 =  20* rx *nw
            dw_dx_2 = (nw / sw)[:, :, None] * (-20 * (rx * w).T)  # / (sw[:,None,None]**2)
            nw_sq = 20 * (nw / sw)[:, :, None]

            dw_dx = nw_sq * (rx * w).T
            dw_dy = nw_sq * (ry * w).T
            dw_dz = nw_sq * (rz * w).T
            I = np.eye(self.total_points)
            dw_dx -= np.swapaxes(np.sum(dw_dx[:, :, :], axis=0)[:, :, None] * I, 1, 0)
            dw_dy -= np.swapaxes(np.sum(dw_dy[:, :, :], axis=0)[:, :, None] * I, 1, 0)
            dw_dz -= np.swapaxes(np.sum(dw_dz[:, :, :], axis=0)[:, :, None] * I, 1, 0)

            inf_x = inflow_vector[:, 0:1]
            inf_y = inflow_vector[:, 1:2]
            inf_z = inflow_vector[:, 2:3]

            dinf_x_dx = dw_dx.T @ inf_x
            dinf_x_dy = dw_dy.T @ inf_x
            dinf_x_dz = dw_dz.T @ inf_x

            dinf_y_dx = dw_dx.T @ inf_y
            dinf_y_dy = dw_dy.T @ inf_y
            dinf_y_dz = dw_dz.T @ inf_y

            dinf_z_dx = dw_dx.T @ inf_z
            dinf_z_dy = dw_dy.T @ inf_z
            dinf_z_dz = dw_dz.T @ inf_z

            du_dX[0::3, 0::3] += dinf_x_dx.T.squeeze()
            du_dX[0::3, 1::3] += dinf_x_dy.T.squeeze()
            du_dX[0::3, 2::3] += dinf_x_dz.T.squeeze()

            du_dX[1::3, 0::3] += dinf_y_dx.T.squeeze()
            du_dX[1::3, 1::3] += dinf_y_dy.T.squeeze()
            du_dX[1::3, 2::3] += dinf_y_dz.T.squeeze()

            du_dX[2::3, 0::3] += dinf_z_dx.T.squeeze()
            du_dX[2::3, 1::3] += dinf_z_dy.T.squeeze()
            du_dX[2::3, 2::3] += dinf_z_dz.T.squeeze()

            ##   du_dGamma
            du_dGamma[0::3, :] = ((1 / Gammak) * u0x * u1 * u2).T
            du_dGamma[1::3, :] = ((1 / Gammak) * u0y * u1 * u2).T
            du_dGamma[2::3, :] = ((1 / Gammak) * u0z * u1 * u2).T

            ##   du_dsigma - core size variation has been removed
            # du2_dsigma = -(np.exp(-1 * cross_r1_r2_norm_sq / (sigmak ** 2 * r0_norm_sq)) *
            #                (2 * cross_r1_r2_norm_sq / (sigmak ** 3 * r0_norm_sq)))
            # du_dsigma[0::3, :] = (u0x * u1 * du2_dsigma).T
            # du_dsigma[1::3, :] = (u0y * u1 * du2_dsigma).T
            # du_dsigma[2::3, :] = (u0z * u1 * du2_dsigma).T

            du_dU[0::3, 0::3] = normalised_weights.T
            du_dU[1::3, 1::3] = normalised_weights.T
            du_dU[2::3, 2::3] = normalised_weights.T

            du_dq = np.where(np.isnan(du_dq), 0, du_dq)

            du_dp[0::3, 0::3] = np.diag(np.sum(du_x_dx0_x, axis=0))
            du_dp[0::3, 1::3] = np.diag(np.sum(du_x_dx0_y, axis=0))
            du_dp[0::3, 2::3] = np.diag(np.sum(du_x_dx0_z, axis=0))

            du_dp[1::3, 0::3] = np.diag(np.sum(du_y_dx0_x, axis=0))
            du_dp[1::3, 1::3] = np.diag(np.sum(du_y_dx0_y, axis=0))
            du_dp[1::3, 2::3] = np.diag(np.sum(du_y_dx0_z, axis=0))

            du_dp[2::3, 0::3] = np.diag(np.sum(du_z_dx0_x, axis=0))
            du_dp[2::3, 1::3] = np.diag(np.sum(du_z_dx0_y, axis=0))
            du_dp[2::3, 2::3] = np.diag(np.sum(du_z_dx0_z, axis=0))

        du_dm = np.zeros((n_p * 3, len(controls)))

        return result, du_dq, du_dm, du_dp

    def update_state(self, states, controls, inflow, with_tangent):
        """Update the state vector one discrete time step with given controls and inflow.

        :param states: full state vector [num_states]
        :param controls: control vector [total_controls]
        :param inflow: inflow vector [3] or [num_points x 3]
        :param with_tangent:  boolean, calculate partial derivatives if `True`
        :return: updated state [num_states] and partial derivatives
        """
        states = states.copy()
        elements, vortex_strengths, free_flow, saved_controls = self.states_from_state_vector(states)
        nr_states, nr_derivatives = self.new_rings(states, controls, inflow, with_tangent)

        inflow_vector = np.reshape(free_flow, (-1, 3))
        points = elements

        ex = elements[:, :, 0]
        ey = elements[:, :, 1]
        ez = elements[:, :, 2]

        e1x = ex[:, :-1]
        e1y = ey[:, :-1]
        e1z = ez[:, :-1]

        e2x = ex[:, 1:]
        e2y = ey[:, 1:]
        e2z = ez[:, 1:]

        r0x = e2x - e1x
        r0y = e2y - e1y
        r0z = e2z - e1z

        r0x = np.reshape(r0x, (-1, 1))
        r0y = np.reshape(r0y, (-1, 1))
        r0z = np.reshape(r0z, (-1, 1))

        p = np.reshape(points, (-1, 3))
        px = p[:, 0].T
        py = p[:, 1].T
        pz = p[:, 2].T

        rx = np.reshape(np.ravel(ex), (-1, 1)) - px
        ry = np.reshape(np.ravel(ey), (-1, 1)) - py
        rz = np.reshape(np.ravel(ez), (-1, 1)) - pz

        r1x = np.reshape(np.ravel(e1x), (-1, 1)) - px
        r1y = np.reshape(np.ravel(e1y), (-1, 1)) - py
        r1z = np.reshape(np.ravel(e1z), (-1, 1)) - pz

        r2x = np.reshape(np.ravel(e2x), (-1, 1)) - px
        r2y = np.reshape(np.ravel(e2y), (-1, 1)) - py
        r2z = np.reshape(np.ravel(e2z), (-1, 1)) - pz

        r1x = np.where(np.abs(r1x) < 1e-9, 0, r1x)
        r1y = np.where(np.abs(r1y) < 1e-9, 0, r1y)
        r1z = np.where(np.abs(r1z) < 1e-9, 0, r1z)

        r2x = np.where(np.abs(r2x) < 1e-9, 0, r2x)
        r2y = np.where(np.abs(r2y) < 1e-9, 0, r2y)
        r2z = np.where(np.abs(r2z) < 1e-9, 0, r2z)

        cross_r1_r2_x = r1y * r2z - r1z * r2y
        cross_r1_r2_y = r1z * r2x - r1x * r2z
        cross_r1_r2_z = r1x * r2y - r1y * r2x

        cross_r1_r2_sq_x = cross_r1_r2_x ** 2
        cross_r1_r2_sq_y = cross_r1_r2_y ** 2
        cross_r1_r2_sq_z = cross_r1_r2_z ** 2

        cross_r1_r2_norm_sq = cross_r1_r2_x ** 2 + cross_r1_r2_y ** 2 + cross_r1_r2_z ** 2
        cross_r1_r2_norm = np.sqrt(cross_r1_r2_norm_sq)

        Gammak = np.reshape(np.ravel(vortex_strengths), (-1, 1))
        #     Gammak = q[P:P+E]
        u0x = (Gammak / (4 * np.pi)) * cross_r1_r2_x / cross_r1_r2_norm_sq
        u0y = (Gammak / (4 * np.pi)) * cross_r1_r2_y / cross_r1_r2_norm_sq
        u0z = (Gammak / (4 * np.pi)) * cross_r1_r2_z / cross_r1_r2_norm_sq
        u0x = np.where(np.isnan(u0x), 0, u0x)
        u0y = np.where(np.isnan(u0y), 0, u0y)
        u0z = np.where(np.isnan(u0z), 0, u0z)

        r1_norm_sq = r1x ** 2 + r1y ** 2 + r1z ** 2
        r1_norm = np.sqrt(r1_norm_sq)
        r2_norm_sq = r2x ** 2 + r2y ** 2 + r2z ** 2
        r2_norm = np.sqrt(r2_norm_sq)

        p2 = (r0x * r1x + r0y * r1y + r0z * r1z) / r1_norm
        p3 = (r0x * r2x + r0y * r2y + r0z * r2z) / r2_norm
        u1 = p2 - p3
        u1 = np.where(np.isnan(u1), 0, u1)

        ui_x = u0x * u1
        ui_y = u0y * u1
        ui_z = u0z * u1

        r0_norm_sq = r0x ** 2 + r0y ** 2 + r0z ** 2
        r0_norm = np.sqrt(r0_norm_sq)

        sigmak = self.vortex_core_size
        u2 = 1 - np.exp(- cross_r1_r2_norm_sq / (sigmak ** 2 * r0_norm_sq))
        u2 = np.where(np.isnan(u2), 0, u2)

        ui_x = ui_x * u2
        ui_y = ui_y * u2
        ui_z = ui_z * u2

        result = np.zeros((self.num_points * self.num_rings * self.num_turbines, 3))

        ui_x = np.where(np.isnan(ui_x), 0, ui_x)
        ui_y = np.where(np.isnan(ui_y), 0, ui_y)
        ui_z = np.where(np.isnan(ui_z), 0, ui_z)

        # todo: check
        # r_norm = (rx ** 2 + ry ** 2 + rz ** 2)
        # weights = np.exp(-r_norm*10)
        # normalised_weights = weights / weights.sum(axis=0)
        normalised_weights = np.eye(self.num_points * self.num_rings * self.num_turbines)
        # print(normalised_weights[:,0])
        inflow_vector_x = (inflow_vector[:, 0:1] * normalised_weights).sum(axis=0)
        inflow_vector_y = (inflow_vector[:, 1:2] * normalised_weights).sum(axis=0)
        inflow_vector_z = (inflow_vector[:, 2:] * normalised_weights).sum(axis=0)

        result[:, 0] = ui_x.sum(axis=0) + inflow_vector_x
        result[:, 1] = ui_y.sum(axis=0) + inflow_vector_y
        result[:, 2] = ui_z.sum(axis=0) + inflow_vector_z
        # result[:, 0] = ui_x.sum(axis=0) + inflow_vector[:,0]
        # result[:, 1] = ui_y.sum(axis=0) + inflow_vector[:,1]
        # result[:, 2] = ui_z.sum(axis=0) + inflow_vector[:,2]

        result = np.reshape(result, points.shape)

        points[1:] = points[:-1] + result[:-1] * self.time_step
        # X0, Gamma0, sigma0, L0 = new_rings(states, controls)
        # print(controls)
        # nr_states, nr_derivatives = self.new_rings(states, controls, inflow, with_tangent)
        # print(nr_states[1])
        X0, Gamma0, U0, M0 = nr_states

        # todo: include dM0 derivatives
        (dX0_dq, dX0_dm), (dGamma0_dq, dGamma0_dm), (dU0_dq, dU0_dm), (dM0_dq, dM0_dm) = nr_derivatives
        # points[0] = X0
        # points[self.num_rings] = X0
        points[0::self.num_rings] = X0

        free_flow[1:] = free_flow[:-1]
        free_flow[0::self.num_rings] = U0
        # Gamma update
        vortex_strengths[1:] = vortex_strengths[:-1]
        # dGamma_dt = thrust_coefficient * (1 / 2) * inflow_velocity_magnitude ** 2 * np.cos(np.deg2rad(initial_yaw)) ** 2
        vortex_strengths[0::self.num_rings] = Gamma0

        # l update
        # l = r0_norm.copy()
        # l[self.num_elements:] = l[:-self.num_elements]
        # l[:self.num_elements] = L0

        # sigma update
        sigmak = self.vortex_core_size

        new_controls = M0

        P1 = self.num_points * 3 * self.num_rings
        E1 = self.num_elements * self.num_rings
        P = self.num_turbines * P1
        E = self.num_turbines * E1
        C = self.total_controls
        dqn_dq = np.zeros((self.num_states, self.num_states))

        if with_tangent:
            ## select subset of full jacobian
            dX_dX = dqn_dq[self.X_index_start:self.X_index_end, self.X_index_start:self.X_index_end]
            #     dGamma_dX = dqn_dq[P:P+E,0:P]

            dX_dGamma = dqn_dq[self.X_index_start:self.X_index_end, self.G_index_start:self.G_index_end]
            dGamma_dGamma = dqn_dq[self.G_index_start:self.G_index_end, self.G_index_start:self.G_index_end]

            dX_dU = dqn_dq[self.X_index_start:self.X_index_end, self.U_index_start: self.U_index_end]
            dU_dU = dqn_dq[self.U_index_start:self.U_index_end, self.U_index_start: self.U_index_end]

            du0_dr0 = 0
            # cross_r1_r2_x = r1y * r2z - r1z * r2y
            # cross_r1_r2_y = r1z * r2x - r1x * r2z
            # cross_r1_r2_z = r1x * r2y - r1y * r2x
            # cross_r1_r2_sq_x = cross_r1_r2_x ** 2
            # cross_r1_r2_sq_y = cross_r1_r2_y ** 2
            # cross_r1_r2_sq_z = cross_r1_r2_z ** 2
            pi = np.pi
            cross_r1_r2_x = r1y * r2z - r1z * r2y

            cross_r1_r2_y = r1z * r2x - r1x * r2z

            cross_r1_r2_z = r1x * r2y - r1y * r2x

            cross_r2_r1_x = r1x * r2z - r1z * r2x

            cross_r1_r2_x_sq = (cross_r1_r2_x) ** 2

            cross_r1_r2_y_sq = (cross_r1_r2_y) ** 2

            cross_r1_r2_z_sq = (cross_r1_r2_z) ** 2

            cross_r2_r1_x_sq = (cross_r2_r1_x) ** 2

            sum_cross_r1_r2_sq_sq = (cross_r1_r2_z_sq + cross_r2_r1_x_sq + cross_r1_r2_x_sq) ** 2

            const = Gammak / (4 * pi * sum_cross_r1_r2_sq_sq)

            diag = cross_r1_r2_norm_sq

            du0_x_dr1_x = const * (-cross_r1_r2_z_sq - cross_r2_r1_x_sq - cross_r1_r2_x_sq - 2 * (cross_r1_r2_x) * (
                    r2y * (cross_r1_r2_z) + r2z * (cross_r2_r1_x)) + diag)

            du0_x_dr1_y = const * (2 * r2x * (cross_r1_r2_z) * (cross_r1_r2_x) + r2z * (
                    cross_r1_r2_z_sq + cross_r2_r1_x_sq - cross_r1_r2_x_sq))

            du0_x_dr1_z = const * (2 * r2x * (cross_r2_r1_x) * (cross_r1_r2_x) - r2y * (
                    cross_r1_r2_z_sq + cross_r2_r1_x_sq - cross_r1_r2_x_sq))

            du0_y_dr1_x = const * (2 * r2y * (cross_r1_r2_z) * (cross_r2_r1_x) - r2z * (
                    cross_r1_r2_z_sq - cross_r2_r1_x_sq + cross_r1_r2_x_sq))

            du0_y_dr1_y = const * (-cross_r1_r2_z_sq - cross_r2_r1_x_sq + 2 * (cross_r2_r1_x) * (
                    -r2x * (cross_r1_r2_z) + r2z * (cross_r1_r2_x)) - cross_r1_r2_x_sq + diag)

            du0_y_dr1_z = const * (
                    r2x * (cross_r1_r2_z_sq - cross_r2_r1_x_sq + cross_r1_r2_x_sq) - 2 * r2y * (cross_r2_r1_x) * (
                cross_r1_r2_x))

            du0_z_dr1_x = const * (
                    r2y * (-cross_r1_r2_z_sq + cross_r2_r1_x_sq + cross_r1_r2_x_sq) - 2 * r2z * (cross_r1_r2_z) * (
                cross_r2_r1_x))

            du0_z_dr1_y = const * (
                    -r2x * (-cross_r1_r2_z_sq + cross_r2_r1_x_sq + cross_r1_r2_x_sq) - 2 * r2z * (cross_r1_r2_z) * (
                cross_r1_r2_x))

            du0_z_dr1_z = const * (-cross_r1_r2_z_sq + 2 * (cross_r1_r2_z) * (
                    r2x * (cross_r2_r1_x) + r2y * (cross_r1_r2_x)) - cross_r2_r1_x_sq - cross_r1_r2_x_sq + diag)

            du0_x_dr2_x = const * (-cross_r1_r2_z_sq - cross_r2_r1_x_sq - cross_r1_r2_x_sq + 2 * (cross_r1_r2_x) * (
                    r1y * (cross_r1_r2_z) + r1z * (cross_r2_r1_x)) + diag)

            du0_x_dr2_y = const * (-2 * r1x * (cross_r1_r2_z) * (cross_r1_r2_x) - r1z * (
                    cross_r1_r2_z_sq + cross_r2_r1_x_sq - cross_r1_r2_x_sq))

            du0_x_dr2_z = const * (-2 * r1x * (cross_r2_r1_x) * (cross_r1_r2_x) + r1y * (
                    cross_r1_r2_z_sq + cross_r2_r1_x_sq - cross_r1_r2_x_sq))

            du0_y_dr2_x = const * (-2 * r1y * (cross_r1_r2_z) * (cross_r2_r1_x) + r1z * (
                    cross_r1_r2_z_sq - cross_r2_r1_x_sq + cross_r1_r2_x_sq))

            du0_y_dr2_y = const * (-cross_r1_r2_z_sq - cross_r2_r1_x_sq + 2 * (cross_r2_r1_x) * (
                    r1x * (cross_r1_r2_z) - r1z * (cross_r1_r2_x)) - cross_r1_r2_x_sq + diag)

            du0_y_dr2_z = const * (
                    -r1x * (cross_r1_r2_z_sq - cross_r2_r1_x_sq + cross_r1_r2_x_sq) + 2 * r1y * (cross_r2_r1_x) * (
                cross_r1_r2_x))

            du0_z_dr2_x = const * (
                    -r1y * (-cross_r1_r2_z_sq + cross_r2_r1_x_sq + cross_r1_r2_x_sq) + 2 * r1z * (cross_r1_r2_z) * (
                cross_r2_r1_x))

            du0_z_dr2_y = const * (
                    r1x * (-cross_r1_r2_z_sq + cross_r2_r1_x_sq + cross_r1_r2_x_sq) + 2 * r1z * (cross_r1_r2_z) * (
                cross_r1_r2_x))

            du0_z_dr2_z = const * (-cross_r1_r2_z_sq - 2 * (cross_r1_r2_z) * (
                    r1x * (cross_r2_r1_x) + r1y * (cross_r1_r2_x)) - cross_r2_r1_x_sq - cross_r1_r2_x_sq + diag)

            du0_x_dr1_x = np.where(np.isnan(du0_x_dr1_x), 0, du0_x_dr1_x)
            du0_x_dr1_y = np.where(np.isnan(du0_x_dr1_y), 0, du0_x_dr1_y)
            du0_x_dr1_z = np.where(np.isnan(du0_x_dr1_z), 0, du0_x_dr1_z)

            du0_y_dr1_x = np.where(np.isnan(du0_y_dr1_x), 0, du0_y_dr1_x)
            du0_y_dr1_y = np.where(np.isnan(du0_y_dr1_y), 0, du0_y_dr1_y)
            du0_y_dr1_z = np.where(np.isnan(du0_y_dr1_z), 0, du0_y_dr1_z)

            du0_z_dr1_x = np.where(np.isnan(du0_z_dr1_x), 0, du0_z_dr1_x)
            du0_z_dr1_y = np.where(np.isnan(du0_z_dr1_y), 0, du0_z_dr1_y)
            du0_z_dr1_z = np.where(np.isnan(du0_z_dr1_z), 0, du0_z_dr1_z)

            du0_x_dr2_x = np.where(np.isnan(du0_x_dr2_x), 0, du0_x_dr2_x)
            du0_x_dr2_y = np.where(np.isnan(du0_x_dr2_y), 0, du0_x_dr2_y)
            du0_x_dr2_z = np.where(np.isnan(du0_x_dr2_z), 0, du0_x_dr2_z)

            du0_y_dr2_x = np.where(np.isnan(du0_y_dr2_x), 0, du0_y_dr2_x)
            du0_y_dr2_y = np.where(np.isnan(du0_y_dr2_y), 0, du0_y_dr2_y)
            du0_y_dr2_z = np.where(np.isnan(du0_y_dr2_z), 0, du0_y_dr2_z)

            du0_z_dr2_x = np.where(np.isnan(du0_z_dr2_x), 0, du0_z_dr2_x)
            du0_z_dr2_y = np.where(np.isnan(du0_z_dr2_y), 0, du0_z_dr2_y)
            du0_z_dr2_z = np.where(np.isnan(du0_z_dr2_z), 0, du0_z_dr2_z)

            du1_dr0_x = r1x / r1_norm - r2x / r2_norm
            du1_dr0_y = r1y / r1_norm - r2y / r2_norm
            du1_dr0_z = r1z / r1_norm - r2z / r2_norm

            dot_r0_r1 = r0x * r1x + r0y * r1y + r0z * r1z
            r1_norm_cube = r1_norm ** 3
            du1_dr1_x = (r1_norm_sq * r0x - dot_r0_r1 * r1x) / r1_norm_cube
            du1_dr1_y = (r1_norm_sq * r0y - dot_r0_r1 * r1y) / r1_norm_cube
            du1_dr1_z = (r1_norm_sq * r0z - dot_r0_r1 * r1z) / r1_norm_cube

            dot_r0_r2 = r0x * r2x + r0y * r2y + r0z * r2z
            r2_norm_cube = r2_norm ** 3
            du1_dr2_x = -(r2_norm_sq * r0x - dot_r0_r2 * r2x) / r2_norm_cube
            du1_dr2_y = -(r2_norm_sq * r0y - dot_r0_r2 * r2y) / r2_norm_cube
            du1_dr2_z = -(r2_norm_sq * r0z - dot_r0_r2 * r2z) / r2_norm_cube

            du1_dr0_x = np.where(np.isnan(du1_dr0_x), 0, du1_dr0_x)
            du1_dr0_y = np.where(np.isnan(du1_dr0_y), 0, du1_dr0_y)
            du1_dr0_z = np.where(np.isnan(du1_dr0_z), 0, du1_dr0_z)

            du1_dr1_x = np.where(np.isnan(du1_dr1_x), 0, du1_dr1_x)
            du1_dr1_y = np.where(np.isnan(du1_dr1_y), 0, du1_dr1_y)
            du1_dr1_z = np.where(np.isnan(du1_dr1_z), 0, du1_dr1_z)

            du1_dr2_x = np.where(np.isnan(du1_dr2_x), 0, du1_dr2_x)
            du1_dr2_y = np.where(np.isnan(du1_dr2_y), 0, du1_dr2_y)
            du1_dr2_z = np.where(np.isnan(du1_dr2_z), 0, du1_dr2_z)

            du2_dr0_c = (-np.exp(-cross_r1_r2_norm_sq / (sigmak ** 2 * r0_norm_sq)) *
                         (2 * cross_r1_r2_norm_sq / (sigmak ** 2 * r0_norm_sq ** 2)))
            du2_dr0_c = np.where(np.isnan(du2_dr0_c), 0, du2_dr0_c)

            du2_dr0_x = du2_dr0_c * r0x
            du2_dr0_y = du2_dr0_c * r0y
            du2_dr0_z = du2_dr0_c * r0z

            du2_dr12_c = (np.exp(-cross_r1_r2_norm_sq / (sigmak ** 2 * r0_norm_sq)) *
                          (2 / (sigmak ** 2 * r0_norm_sq)))
            du2_dr12_c = np.where(np.isnan(du2_dr12_c), 0, du2_dr12_c)
            #     du2_dr1_x = du2_dr12_c * (r2y * cross_r1_r2_z - cross_r1_r2_y * r2z)
            #     du2_dr1_y = du2_dr12_c * (r2z * cross_r1_r2_x - cross_r1_r2_z * r2x)
            #     du2_dr1_z = du2_dr12_c * (r2x * cross_r1_r2_y - cross_r1_r2_x * r2y)

            #     du2_dr2_x = du2_dr12_c * (r1z * cross_r1_r2_y - cross_r1_r2_z * r1y)
            #     du2_dr2_y = du2_dr12_c * (r1x * cross_r1_r2_z - cross_r1_r2_x * r1z)
            #     du2_dr2_z = du2_dr12_c * (r1y * cross_r1_r2_x - cross_r1_r2_y * r1x)
            du2_dr1_x = du2_dr12_c * (r2y * (cross_r1_r2_z) + r2z * (-cross_r1_r2_y))
            du2_dr1_y = du2_dr12_c * (-r2x * (cross_r1_r2_z) + r2z * (cross_r1_r2_x))
            du2_dr1_z = du2_dr12_c * (-r2x * (-cross_r1_r2_y) - r2y * (cross_r1_r2_x))

            du2_dr2_x = du2_dr12_c * (-r1y * (cross_r1_r2_z) - r1z * (-cross_r1_r2_y))
            du2_dr2_y = du2_dr12_c * (r1x * (cross_r1_r2_z) - r1z * (cross_r1_r2_x))
            du2_dr2_z = du2_dr12_c * (r1x * (-cross_r1_r2_y) + r1y * (cross_r1_r2_x))

            du_x_dr0_x = u0x * du1_dr0_x * u2 + u0x * u1 * du2_dr0_x
            du_x_dr0_y = u0x * du1_dr0_y * u2 + u0x * u1 * du2_dr0_y
            du_x_dr0_z = u0x * du1_dr0_z * u2 + u0x * u1 * du2_dr0_z

            du_y_dr0_x = u0y * du1_dr0_x * u2 + u0y * u1 * du2_dr0_x
            du_y_dr0_y = u0y * du1_dr0_y * u2 + u0y * u1 * du2_dr0_y
            du_y_dr0_z = u0y * du1_dr0_z * u2 + u0y * u1 * du2_dr0_z

            du_z_dr0_x = u0z * du1_dr0_x * u2 + u0z * u1 * du2_dr0_x
            du_z_dr0_y = u0z * du1_dr0_y * u2 + u0z * u1 * du2_dr0_y
            du_z_dr0_z = u0z * du1_dr0_z * u2 + u0z * u1 * du2_dr0_z

            du_x_dr1_x = du0_x_dr1_x * u1 * u2 + u0x * du1_dr1_x * u2 + u0x * u1 * du2_dr0_x
            du_x_dr1_y = du0_x_dr1_y * u1 * u2 + u0x * du1_dr1_y * u2 + u0x * u1 * du2_dr0_y
            du_x_dr1_z = du0_x_dr1_z * u1 * u2 + u0x * du1_dr1_z * u2 + u0x * u1 * du2_dr0_z

            du_y_dr1_x = du0_y_dr1_x * u1 * u2 + u0y * du1_dr1_x * u2 + u0y * u1 * du2_dr1_x
            du_y_dr1_y = du0_y_dr1_y * u1 * u2 + u0y * du1_dr1_y * u2 + u0y * u1 * du2_dr1_y
            du_y_dr1_z = du0_y_dr1_z * u1 * u2 + u0y * du1_dr1_z * u2 + u0y * u1 * du2_dr1_z

            du_z_dr1_x = du0_z_dr1_x * u1 * u2 + u0z * du1_dr1_x * u2 + u0z * u1 * du2_dr1_x
            du_z_dr1_y = du0_z_dr1_y * u1 * u2 + u0z * du1_dr1_y * u2 + u0z * u1 * du2_dr1_y
            du_z_dr1_z = du0_z_dr1_z * u1 * u2 + u0z * du1_dr1_z * u2 + u0z * u1 * du2_dr1_z

            du_x_dr2_x = du0_x_dr2_x * u1 * u2 + u0x * du1_dr2_x * u2 + u0x * u1 * du2_dr2_x
            du_x_dr2_y = du0_x_dr2_y * u1 * u2 + u0x * du1_dr2_y * u2 + u0x * u1 * du2_dr2_y
            du_x_dr2_z = du0_x_dr2_z * u1 * u2 + u0x * du1_dr2_z * u2 + u0x * u1 * du2_dr2_z

            du_y_dr2_x = du0_y_dr2_x * u1 * u2 + u0y * du1_dr2_x * u2 + u0y * u1 * du2_dr2_x
            du_y_dr2_y = du0_y_dr2_y * u1 * u2 + u0y * du1_dr2_y * u2 + u0y * u1 * du2_dr2_y
            du_y_dr2_z = du0_y_dr2_z * u1 * u2 + u0y * du1_dr2_z * u2 + u0y * u1 * du2_dr2_z

            du_z_dr2_x = du0_z_dr2_x * u1 * u2 + u0z * du1_dr2_x * u2 + u0z * u1 * du2_dr2_x
            du_z_dr2_y = du0_z_dr2_y * u1 * u2 + u0z * du1_dr2_y * u2 + u0z * u1 * du2_dr2_y
            du_z_dr2_z = du0_z_dr2_z * u1 * u2 + u0z * du1_dr2_z * u2 + u0z * u1 * du2_dr2_z

            du_x_dx0_x = -du_x_dr1_x - du_x_dr2_x
            du_x_dx0_y = -du_x_dr1_y - du_x_dr2_y
            du_x_dx0_z = -du_x_dr1_z - du_x_dr2_z

            du_y_dx0_x = -du_y_dr1_x - du_y_dr2_x
            du_y_dx0_y = -du_y_dr1_y - du_y_dr2_y
            du_y_dx0_z = -du_y_dr1_z - du_y_dr2_z

            du_z_dx0_x = -du_z_dr1_x - du_z_dr2_x
            du_z_dx0_y = -du_z_dr1_y - du_z_dr2_y
            du_z_dx0_z = -du_z_dr1_z - du_z_dr2_z

            du_x_dx1_x = du_x_dr1_x - du_x_dr0_x
            du_x_dx1_y = du_x_dr1_y - du_x_dr0_y
            du_x_dx1_z = du_x_dr1_z - du_x_dr0_z

            du_y_dx1_x = du_y_dr1_x - du_y_dr0_x
            du_y_dx1_y = du_y_dr1_y - du_y_dr0_y
            du_y_dx1_z = du_y_dr1_z - du_y_dr0_z

            du_z_dx1_x = du_z_dr1_x - du_z_dr0_x
            du_z_dx1_y = du_z_dr1_y - du_z_dr0_y
            du_z_dx1_z = du_z_dr1_z - du_z_dr0_z

            du_x_dx2_x = du_x_dr2_x + du_x_dr0_x
            du_x_dx2_y = du_x_dr2_y + du_x_dr0_y
            du_x_dx2_z = du_x_dr2_z + du_x_dr0_z

            du_y_dx2_x = du_y_dr2_x + du_y_dr0_x
            du_y_dx2_y = du_y_dr2_y + du_y_dr0_y
            du_y_dx2_z = du_y_dr2_z + du_y_dr0_z

            du_z_dx2_x = du_z_dr2_x + du_z_dr0_x
            du_z_dx2_y = du_z_dr2_y + du_z_dr0_y
            du_z_dx2_z = du_z_dr2_z + du_z_dr0_z

            du_x_dx1_x = np.where(np.isnan(du_x_dx1_x), 0, du_x_dx1_x)
            du_x_dx1_y = np.where(np.isnan(du_x_dx1_y), 0, du_x_dx1_y)
            du_x_dx1_z = np.where(np.isnan(du_x_dx1_z), 0, du_x_dx1_z)

            du_y_dx1_x = np.where(np.isnan(du_y_dx1_x), 0, du_y_dx1_x)
            du_y_dx1_y = np.where(np.isnan(du_y_dx1_y), 0, du_y_dx1_y)
            du_y_dx1_z = np.where(np.isnan(du_y_dx1_z), 0, du_y_dx1_z)

            du_z_dx1_x = np.where(np.isnan(du_z_dx1_x), 0, du_z_dx1_x)
            du_z_dx1_y = np.where(np.isnan(du_z_dx1_y), 0, du_z_dx1_y)
            du_z_dx1_z = np.where(np.isnan(du_z_dx1_z), 0, du_z_dx1_z)

            du_x_dx2_x = np.where(np.isnan(du_x_dx2_x), 0, du_x_dx2_x)
            du_x_dx2_y = np.where(np.isnan(du_x_dx2_y), 0, du_x_dx2_y)
            du_x_dx2_z = np.where(np.isnan(du_x_dx2_z), 0, du_x_dx2_z)

            du_y_dx2_x = np.where(np.isnan(du_y_dx2_x), 0, du_y_dx2_x)
            du_y_dx2_y = np.where(np.isnan(du_y_dx2_y), 0, du_y_dx2_y)
            du_y_dx2_z = np.where(np.isnan(du_y_dx2_z), 0, du_y_dx2_z)

            du_z_dx2_x = np.where(np.isnan(du_z_dx2_x), 0, du_z_dx2_x)
            du_z_dx2_y = np.where(np.isnan(du_z_dx2_y), 0, du_z_dx2_y)
            du_z_dx2_z = np.where(np.isnan(du_z_dx2_z), 0, du_z_dx2_z)

            for idx in range(self.num_rings * self.num_turbines):
                dX_dX_sub = dX_dX[:, idx * self.num_points * 3:(idx + 1) * self.num_points * 3]

                dX_dX_sub[0::3, 0:-3:3] += du_x_dx1_x[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                dX_dX_sub[1::3, 0:-3:3] += du_y_dx1_x[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                dX_dX_sub[2::3, 0:-3:3] += du_z_dx1_x[idx * self.num_elements:(idx + 1) * self.num_elements, :].T

                dX_dX_sub[0::3, 1:-2:3] += du_x_dx1_y[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                dX_dX_sub[1::3, 1:-2:3] += du_y_dx1_y[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                dX_dX_sub[2::3, 1:-2:3] += du_z_dx1_y[idx * self.num_elements:(idx + 1) * self.num_elements, :].T

                dX_dX_sub[0::3, 2:-1:3] += du_x_dx1_z[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                dX_dX_sub[1::3, 2:-1:3] += du_y_dx1_z[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                dX_dX_sub[2::3, 2:-1:3] += du_z_dx1_z[idx * self.num_elements:(idx + 1) * self.num_elements, :].T

                dX_dX_sub[0::3, 3::3] += du_x_dx2_x[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                dX_dX_sub[1::3, 3::3] += du_y_dx2_x[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                dX_dX_sub[2::3, 3::3] += du_z_dx2_x[idx * self.num_elements:(idx + 1) * self.num_elements, :].T

                dX_dX_sub[0::3, 4::3] += du_x_dx2_y[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                dX_dX_sub[1::3, 4::3] += du_y_dx2_y[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                dX_dX_sub[2::3, 4::3] += du_z_dx2_y[idx * self.num_elements:(idx + 1) * self.num_elements, :].T

                dX_dX_sub[0::3, 5::3] += du_x_dx2_z[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                dX_dX_sub[1::3, 5::3] += du_y_dx2_z[idx * self.num_elements:(idx + 1) * self.num_elements, :].T
                dX_dX_sub[2::3, 5::3] += du_z_dx2_z[idx * self.num_elements:(idx + 1) * self.num_elements, :].T

            #     dX_dX
            du_x_dx0_x = np.where(np.isnan(du_x_dx0_x), 0, du_x_dx0_x)
            du_x_dx0_y = np.where(np.isnan(du_x_dx0_y), 0, du_x_dx0_y)
            du_x_dx0_z = np.where(np.isnan(du_x_dx0_z), 0, du_x_dx0_z)

            du_y_dx0_x = np.where(np.isnan(du_y_dx0_x), 0, du_y_dx0_x)
            du_y_dx0_y = np.where(np.isnan(du_y_dx0_y), 0, du_y_dx0_y)
            du_y_dx0_z = np.where(np.isnan(du_y_dx0_z), 0, du_y_dx0_z)

            du_z_dx0_x = np.where(np.isnan(du_z_dx0_x), 0, du_z_dx0_x)
            du_z_dx0_y = np.where(np.isnan(du_z_dx0_y), 0, du_z_dx0_y)
            du_z_dx0_z = np.where(np.isnan(du_z_dx0_z), 0, du_z_dx0_z)

            dX_dX[0::3, 0::3] += np.diag(np.sum(du_x_dx0_x, 0))
            dX_dX[0::3, 1::3] += np.diag(np.sum(du_x_dx0_y, 0))
            dX_dX[0::3, 2::3] += np.diag(np.sum(du_x_dx0_z, 0))

            dX_dX[1::3, 0::3] += np.diag(np.sum(du_y_dx0_x, 0))
            dX_dX[1::3, 1::3] += np.diag(np.sum(du_y_dx0_y, 0))
            dX_dX[1::3, 2::3] += np.diag(np.sum(du_y_dx0_z, 0))

            dX_dX[2::3, 0::3] += np.diag(np.sum(du_z_dx0_x, 0))
            dX_dX[2::3, 1::3] += np.diag(np.sum(du_z_dx0_y, 0))
            dX_dX[2::3, 2::3] += np.diag(np.sum(du_z_dx0_z, 0))

            h = self.time_step
            dX_dX *= h
            dX_dX += np.eye(self.X_index_end)

            dX_dX[3 * self.num_points:] = dX_dX[:-3 * self.num_points]
            # for wt in range(self.num_turbines):
            #     dX_dX[wt * 1:wt * E1 + self.num_elements] = 0
            for wt in range(self.num_turbines):
                dX_dX[wt * P1: wt * P1 + 3 * self.num_points] = 0

            ## dX_dU
            # dX_dU[0::3,0::3] = h*normalised_weights
            dX_dU[0::3, 0::3] = h * normalised_weights.T
            # dX_dU[0::3, 1::3] = h*normalised_weights
            # dX_dU[0::3, 2::3] = h*normalised_weights

            # dX_dU[1::3, 0::3] = h*normalised_weights
            dX_dU[1::3, 1::3] = h * normalised_weights.T
            # dX_dU[1::3, 2::3] = h*normalised_weights

            # dX_dU[2::3, 0::3] = h*normalised_weights
            # dX_dU[2::3, 1::3] = h*normalised_weights
            dX_dU[2::3, 2::3] = h * normalised_weights.T

            dX_dU[3 * self.num_points:] = dX_dU[:-3 * self.num_points]
            dX_dU[0:3 * self.num_points] = 0

            ##   dGamma_dGamma
            dGamma_dGamma[:] = np.diag(np.ones(E - self.num_elements), -self.num_elements)
            # off-diagonal, zero ring already zero
            # for wt in range(self.num_turbines):
            #     dGamma_dGamma[wt * E1:wt * E1 + self.num_elements] = 0

            ##   dX_dGamma
            dX_dGamma[0::3, :] = ((1 / Gammak) * u0x * u1 * u2).T
            dX_dGamma[1::3, :] = ((1 / Gammak) * u0y * u1 * u2).T
            dX_dGamma[2::3, :] = ((1 / Gammak) * u0z * u1 * u2).T
            #     dX_dGamma[0::3,:] = np.dot(((1/Gammak) * u0x * u1 * u2).T, dGamma_dGamma)
            #     dX_dGamma[1::3,:] = np.dot(((1/Gammak) * u0y * u1 * u2).T, dGamma_dGamma)
            #     dX_dGamma[2::3,:] = np.dot(((1/Gammak) * u0z * u1 * u2).T, dGamma_dGamma)
            # zero ring does not move, shift dependency down
            dX_dGamma[3 * self.num_points:, :] = h * dX_dGamma[:-3 * self.num_points, :]
            for idx in range(self.num_turbines):
                dX_dGamma[idx * P1:idx * P1 + 3 * self.num_points, :] = 0

            for wt in range(self.num_turbines):
                dqn_dq[P + wt * E1:P + wt * E1 + self.num_elements] = dGamma0_dq[wt]

            ##   ddU_dU
            dU_dU[:] = np.diag(np.ones(P - 3 * self.num_points), -3 * self.num_points)
            # off-diagonal, zero ring already zero
            # for wt in range(self.num_turbines):
            #     dGamma_dGamma[wt * P1:wt * P1 + 3 * self.num_points] = 0

            dqn_dq[self.G_index_start:self.G_index_start + self.num_elements] = dGamma0_dq
            dqn_dq = np.where(np.isnan(dqn_dq), 0, dqn_dq)

        dqn_dm = np.zeros((self.num_states, len(controls)))

        if with_tangent:
            for wt in range(self.num_turbines):
                dqn_dm[P + wt * E1: P + wt * E1 + self.num_elements] = dGamma0_dm[wt]
                dqn_dm[wt * P1: wt * P1 + 3 * self.num_points] = dX0_dm[wt]

            dqn_dm[self.M_index_start:self.M_index_end] = dM0_dm

        return self.state_vector_from_states(points, vortex_strengths, free_flow, new_controls), dqn_dq, dqn_dm


def rot_z_2d(psi):
    """2D rotation matrix, clockwise positive around z-axis

    :param psi: rotation angle (degrees)
    :returns: 2x2 rotation matrix
    """
    psi = np.deg2rad(psi)
    R = np.array([[np.cos(psi), np.sin(psi)],
                  [-np.sin(psi), np.cos(psi)]])
    return R


def drot_z_dpsi_2d(psi):
    """Derivative to angle of 3D rotation matrix, clockwise positive around z-axis

    :param psi: rotation angle (degrees)
    :returns: 2x2 rotation matrix derivative (degrees$^{-1}$)
    """
    psi = np.deg2rad(psi)
    dR_dpsi = np.array([[-np.sin(psi), np.cos(psi)],
                        [-np.cos(psi), -np.sin(psi)]])
    return np.deg2rad(dR_dpsi)


def rot_z_3d(psi):
    """3D rotation matrix, clockwise positive around z-axis

    :param psi: rotation angle (degrees)
    :returns: 3x3 rotation matrix
    """
    psi = np.deg2rad(psi)
    R = np.array([[np.cos(psi), np.sin(psi), 0.],
                  [-np.sin(psi), np.cos(psi), 0.],
                  [0., 0., 1.]])
    return R


def drot_z_dpsi_3d(psi):
    """Derivative to angle of 3D rotation matrix, clockwise positive around z-axis

    :param psi: rotation angle (degrees)
    :returns: 3x3 rotation matrix derivative (degrees$^{-1}$)
    """
    psi = np.deg2rad(psi)
    dR_dpsi = np.array([[-np.sin(psi), np.cos(psi), 0],
                        [-np.cos(psi), -np.sin(psi), 0],
                        [0, 0, 0.]])
    return np.deg2rad(dR_dpsi)


def construct_gradient(dqn_dq, dqn_dm, dphi_dq, dphi_dm):
    """ Construct gradient of a cost function through backwards propagation with the discrete adjoint

    :param dqn_dq: [num_steps x num_states x num_states]
    :param dqn_dm: [num_steps x num_states x num_controls]
    :param dphi_dq: [num_steps x num_states]
    :param dphi_dm: [num_steps x num_controls]
    :return: gradient array
    """
    num_steps = dqn_dq.shape[0]
    num_controls = dqn_dm.shape[-1]
    num_states = dqn_dq.shape[-1]

    gradient = np.zeros((num_steps, 1, num_controls))
    lamda_v = np.zeros((num_steps + 1, num_states, 1))

    lamda_v[-1] = dphi_dq[-1].T
    # gradient[-1] = dphi_dm[-1]

    for idx in range(num_steps, 0, -1):
        n = idx - 1
        lamda_v[n] = (dphi_dq[n] + lamda_v[n + 1].T @ dqn_dq[n]).T
        gradient[n] = dphi_dm[n] + (dqn_dm[n].T @ lamda_v[n + 1]).T

    return np.squeeze(gradient)
