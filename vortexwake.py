import numpy as np
import json


class VortexWake:

    def __init__(self, config_file):
        with open(config_file, "r") as cf:
            config = json.load(cf)

            self.dim = config["dimension"]
            self.num_elements = config.get("num_elements", 1)

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

            self.X_index_start = 0
            self.X_index_end = self.dim * self.total_points
            self.G_index_start = self.X_index_end
            # todo: reformulate such that single Gamma per ring
            self.G_index_end = self.G_index_start + self.total_elements
            self.U_index_start = self.G_index_end
            self.U_index_end = self.U_index_start + self.dim * self.total_points
            self.M_index_start = self.U_index_end
            self.M_index_end = self.M_index_start + self.num_controls * self.num_turbines

            radius = 0.5
            if self.dim == 2:
                azimuthal_angles = np.array([0, np.pi])
            elif self.dim == 3:
                azimuthal_angles = np.arange(0., self.num_points) * (2 * np.pi) / self.num_elements

            self.y0 = radius * np.cos(azimuthal_angles)
            self.z0 = radius * np.sin(azimuthal_angles)

            # structure of control vector
            # todo: check number of controls
            self.induction_idx = 0
            self.yaw_idx = 1

            self.unit_vector_x = np.zeros(self.dim)
            self.unit_vector_x[0] = 1
            if self.dim == 2:
                self.rot_z = rot_z_2d
                self.drot_z_dpsi = drot_z_dpsi_2d
            elif self.dim == 3:
                self.rot_z = rot_z_3d
                self.drot_z_dpsi = drot_z_dpsi_3d

    def states_from_state_vector(self, q):
        X = q[self.X_index_start: self.X_index_end].reshape(self.total_rings, self.num_points, self.dim)
        G = q[self.G_index_start: self.G_index_end].reshape(self.total_elements, 1)
        U = q[self.U_index_start: self.U_index_end].reshape(self.total_rings, self.num_points, self.dim)
        M = q[self.M_index_start: self.M_index_end].reshape(self.num_controls, 1)
        return X, G, U, M

    def state_vector_from_states(self, X, G, U, M):
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

    # todo:
    # def initialise_states(self):
    # def new_rings(self):
    # def new_rings_with_tangent(self):

    def new_rings(self, states, controls, inflow, with_tangent=False):
        X0 = np.zeros((self.num_turbines, self.num_points, self.dim))
        G0 = np.zeros((self.num_turbines, 1))
        U0 = np.zeros((self.num_turbines, self.num_points, self.dim))
        M0 = np.zeros((self.num_controls, 1))

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
