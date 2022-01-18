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
        print('delta_max')
        delta = delta_max
    else:
        print('-delta_max')
        delta = -delta_max

    dot_x = np.array([- vf * np.sin(delta - psi),
                      - vf * np.sin(delta)])

    return dot_x
