### From https://arxiv.org/pdf/1703.07373.pdf Eq. (19) (Left)
import numpy as np

# quadrotor physical constants
g = 9.81; d0 = 10; d1 = 8; n0 = 10; kT = 0.91

# non-linear dynamics
def f(x, u):
    x, y, z, vx, vy, vz, theta_x, theta_y, omega_x, omega_y = x.reshape(-1).tolist()
    az, ax, ay = u.reshape(-1).tolist()
    dot_x = np.array([
     vx,
     vy,
     vz,
     g * np.tan(theta_x),
     g * np.tan(theta_y),
     kT * az - g,
     -d1 * theta_x + omega_x,
     -d1 * theta_y + omega_y,
     -d0 * theta_x + n0 * ax,
     -d0 * theta_y + n0 * ay])
    return dot_x

# linearization
# The state variables are x, y, z, vx, vy, vz, theta_x, theta_y, omega_x, omega_y
A = np.zeros([10, 10])
A[0, 3] = 1.
A[1, 4] = 1.
A[2, 5] = 1.
A[3, 6] = g
A[4, 7] = g
A[6, 6] = -d1
A[6, 8] = 1
A[7, 7] = -d1
A[7, 9] = 1
A[8, 6] = -d0
A[9, 7] = -d0

B = np.zeros([10, 3])
B[5, 0] = kT
B[8, 1] = n0
B[9, 2] = n0
