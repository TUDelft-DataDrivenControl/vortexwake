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


def rot_z(psi):
    """
    3D rotation matrix, clockwise positive around z-axis
    :param psi: rotation angle (radians)
    :return: 3x3 rotation matrix
    """
    R = np.array([[np.cos(psi), np.sin(psi), 0.],
                  [-np.sin(psi), np.cos(psi), 0.],
                  [0., 0., 1.]])
    return R


def drot_z_dpsi(psi):
    """
    Derivative to angle of 3D rotation matrix, clockwise positive around z-axis
    :param psi: rotation angle (radians)
    :return: 3x3 rotation matrix derivative
    """
    dR_dpsi = np.array([[-np.sin(psi), np.cos(psi), 0],
                        [-np.cos(psi), -np.sin(psi), 0],
                        [0, 0, 0.]])
    return dR_dpsi
