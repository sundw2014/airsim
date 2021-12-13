import numpy as np

# quadrotor physical constants
g = 9.81

# non-linear dynamics
def f(x, u):
    x, y, z, vx, vy, vz, theta_x, theta_y = x.reshape(-1).tolist()
    az, omega_x, omega_y = u.reshape(-1).tolist()
    dot_x = np.array([
     vx,
     vy,
     vz,
     g * np.tan(theta_x),
     g * np.tan(theta_y),
     az,
     omega_x,
     omega_y])
    return dot_x
