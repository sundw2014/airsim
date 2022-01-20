import numpy as np

vf = 2.8
L = 1.75
delta_max = 0.61
K = 0.45

# non-linear dynamics
def f(x):
    # x: bs x n
    # f: bs x n
    d, psi = x.reshape(-1).tolist()

    delta_pre = psi + np.arctan(K * d / vf)
    if np.abs(delta_pre) < delta_max:
        delta = delta_pre
    elif delta_pre >= delta_max:
        # print('delta_max')
        delta = delta_max
    else:
        # print('-delta_max')
        delta = -delta_max

    dot_x = np.array([- vf * np.sin(delta - psi),
                      - vf * np.sin(delta) / L])

    return dot_x

def dfdx(x):
    # x: bs x n
    # f: bs x n
    d, psi = x.reshape(-1).tolist()

    delta = psi + np.arctan(K * d / vf)
    DdeltaDd = 1 / (1 + (K * d / vf) ** 2) * K / vf
    if np.abs(delta) < delta_max:
        J = np.array([[- vf * np.cos(delta - psi) * DdeltaDd, 0], 
                      [- vf * np.cos(delta) / L * DdeltaDd, - vf * np.cos(delta) / L * 1]])
    else:
        J = np.array([[0, 0], 
                      [0, - vf * np.cos(delta_max) / L * 1]])

    return J

def dudx(x):
    # x: bs x n
    # f: bs x n
    d, psi = x.reshape(-1).tolist()

    delta_pre = psi + np.arctan(K * d / vf)
    DdeltaDpsi = 1
    DdeltaDd = 1 / (1 + (K * d / vf) ** 2) * K / vf
    return np.array([DdeltaDd, DdeltaDpsi])

def g(x):
    # x: bs x n
    # f: bs x n
    d, psi = x.reshape(-1).tolist()

    delta_pre = psi + np.arctan(K * d / vf)
    if np.abs(delta_pre) < delta_max:
        delta = delta_pre
    elif delta_pre >= delta_max:
        delta = delta_max
    else:
        delta = -delta_max

    return delta

# non-linear dynamics
def fxz(x, z):
    # x: bs x n
    # f: bs x n
    d, psi = x.reshape(-1).tolist()
    delta = g(x.reshape(-1) + z.reshape(-1))
    dot_x = np.array([- vf * np.sin(delta - psi),
                      - vf * np.sin(delta) / L])

    return dot_x
