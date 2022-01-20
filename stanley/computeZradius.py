import numpy as np
import scipy
import scipy.optimize
from dynamics import fxz

P = np.array([[1,0],[0,1.5]])

def V(x):
    x = x.reshape(-1,1)
    return x.T.dot(P).dot(x)

def DVDx(x):
    x = x.reshape(-1,1)
    return 2 * x.T.dot(P)

def dVdt(x, z):
    x = x.reshape(-1,1)
    return DVDx(x).dot(fxz(x,z).reshape(-1,1))

x_lb = np.array([-1, -np.pi / 3])
x_ub = np.array([1, np.pi / 3])

def objective(i):
    x, z = i[:2], i[2:]
    return -dVdt(x, z)


for radius in np.arange(0, 3, 0.01):
    x = scipy.optimize.brute(objective, np.array([x_lb, x_ub]).T.tolist()+[[-radius, radius], [-radius, radius]], finish=None)
    if objective(x) > 0:
        print('radius = %.3f is fine'%radius)
    else:
        break
