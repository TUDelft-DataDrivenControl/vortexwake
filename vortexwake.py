import numpy as np
import json


class VortexWake:
    """Class for wake simulation with Free-Vortex method.
    2D or 3D, with discrete adjoint for gradients.

    """

    def __init__(self, config_file):
        with open(config_file, "r") as cf:
            config = json.load(cf)
            if config["dimension"] != self.dim:
                raise ValueError("Trying to instantiate 2D FVW with 3D configuration")
            # self.dim = config["dimension"]
            self.num_elements = config.get("num_elements", 1)
            if self.dim == 2:
                self.num_elements = 1

            self.num_rings = config["num_rings"]
            self.num_points = self.num_elements + 1
            self.num_controls = 2
            self.num_turbines = config.get("num_turbines", 1)

            self.turbine_positions = np.array(config.get("turbine_positions", [[0., 0., 0.]]))

            self.total_rings = self.num_rings * self.num_turbines
            self.total_points = self.num_points * self.total_rings
            self.total_elements = self.num_elements * self.total_rings
            self.total_controls = self.num_controls * self.num_turbines
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
            self.M_index_end = self.M_index_start + self.num_controls * self.num_turbines

            self.unit_vector_x = np.zeros(self.dim)
            self.unit_vector_x[0] = 1
            # structure of control vector
            # todo: check number of controls
            self.induction_idx = 0
            self.yaw_idx = 1

    def states_from_state_vector(self, q):
        """Unpack state column vector into state arrays

        :param q:
        :return:
        """
        X = q[self.X_index_start: self.X_index_end].reshape(self.total_rings, self.num_points, self.dim)
        G = q[self.G_index_start: self.G_index_end].reshape(self.total_elements, 1)
        U = q[self.U_index_start: self.U_index_end].reshape(self.total_rings, self.num_points, self.dim)
        M = q[self.M_index_start: self.M_index_end].reshape(self.total_controls, 1)
        return X, G, U, M

    def state_vector_from_states(self, X, G, U, M):
        """Pack state arrays into a single column vector

        :param X:
        :param G:
        :param U:
        :param M:
        :return:
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

        :return: states
        """
        X, G, U, M = self.states_from_state_vector(np.zeros((self.num_states, 1)))
        (X0, G0, U0, M0), derivatives = self.new_rings(np.zeros(self.num_states), np.zeros(self.total_controls),
                                                       self.unit_vector_x)
        X[::self.num_rings] = X0
        G[::self.num_rings] = G0
        U[::self.num_rings] = U0
        M[:] = M0
        return X, G, U, M

    def new_rings(self, states, controls, inflow, with_tangent=False):
        """Generate values for new rings to initialised.

        :param states: state vector
        :param controls: control vector
        :param inflow: inflow [dim] or [num_points x dim]
        :param with_tangent:
        :return: new ring states, new ring state derivatives
        """
        X0 = np.zeros((self.num_turbines, self.num_points, self.dim))
        G0 = np.zeros((self.num_turbines, 1))
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
        for wt in range(self.num_turbines):
            X0[wt, :] = X0[wt, :] @ self.rot_z(psi[wt]).T
            X0[wt] += self.turbine_positions[wt]

            thrust_coefficient = 4 * a[wt] / (1 - a[wt])
            n = self.unit_vector_x @ self.rot_z(psi[wt]).T

            # todo: move deg2rad conversion to rotation matrix
            G0 = self.time_step * thrust_coefficient * (1 / 2) * (ur[wt].T @ n) ** 2

            if with_tangent:
                dn_dpsi = self.unit_vector_x @ self.drot_z_dpsi(psi[wt]).T

                dX0_dm[wt, :, self.yaw_idx + self.num_controls * wt] = np.reshape(
                    X0[wt, :] @ self.drot_z_dpsi(psi[wt]).T,
                    (self.num_points * self.dim,))

                dG0_dur = h * thrust_coefficient * n * (ur[wt].T @ n)
                dG0_dq[wt] = dG0_dur @ dur_dq[wt]

                dG0_da = h * (1 / 2) * (ur[wt].T @ n) ** 2 * (4 / (1 - a[wt]) ** 2)
                dG0_dpsi = h * thrust_coefficient * (ur[wt].T @ n) * (
                        ur[wt].T @ dn_dpsi + n.T @ dur_dm[wt][:, self.yaw_idx + self.num_controls * wt])
                dG0_dm[wt, self.induction_idx + self.num_controls * wt] = dG0_da
                dG0_dm[wt, self.yaw_idx + self.num_controls * wt] = dG0_dpsi

        return (X0, G0, U0, M0), ((dX0_dq, dX0_dm), (dG0_dq, dG0_dm), (dU0_dq, dU0_dm), (dM0_dq, dM0_dm))

    def disc_velocity(self, states, controls, with_tangent):
        Warning("disc velocity not implemented yet")
        ur = np.zeros((self.num_turbines, self.dim))
        dur_dq = np.zeros((self.num_turbines, self.dim, self.num_states))
        dur_dm = np.zeros((self.num_turbines, self.dim, self.total_controls))
        return ur, dur_dq, dur_dm

    # todo:

    # todo:
    # def update_state(self, q):
    # def update_state_with_tangent(self, q):
    # def run_forward(self):
    # def velocity(self, q, m, pt)
    # def velocity_tangent(self, q, m, pt)
    # def disc_velocity(self, q, m)
    # def disc_velocity_tangent(self, q, m)
    # def disc_velocity_virtual_tangent(self, q, m, pt, yaw)
    # def calculate_power()
    #  def calculate_virtual_power()


# todo:
# def evaluate_cost_function(
# construct_gradient


class VortexWake2D(VortexWake):
    def __init__(self, config_file):
        self.dim = 2

        VortexWake.__init__(self, config_file)
        self.num_elements = 1
        self.rot_z = rot_z_2d
        self.drot_z_dpsi = drot_z_dpsi_2d

        azimuthal_angles = np.array([0, np.pi])
        self.y0 = self.radius * np.cos(azimuthal_angles)


class VortexWake3D(VortexWake):
    def __init__(self, config_file):
        self.dim = 3
        VortexWake.__init__(self, config_file)

        self.rot_z = rot_z_3d
        self.drot_z_dpsi = drot_z_dpsi_3d

        azimuthal_angles = np.arange(0., self.num_points) * (2 * np.pi) / self.num_elements
        self.y0 = self.radius * np.cos(azimuthal_angles)
        self.z0 = self.radius * np.sin(azimuthal_angles)

    def velocity(self, states, controls, points, with_tangent):
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

        du_dm = np.zeros((n_p * 3, len(controls)))

        return result, du_dq, du_dm


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
