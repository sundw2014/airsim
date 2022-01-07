import numpy as np
import scipy
from dynamics import A, B, kT, f, g
from controller import K

Acl = A-B.dot(K)
P = scipy.linalg.solve_continuous_lyapunov(Acl.T, -np.eye(A.shape[0]))

def V(x):
	x = x.reshape(-1,1)
	return x.T.dot(P).dot(x)

def DVDx(x):
	x = x.reshape(-1,1)
	return 2 * x.T.dot(P)

def dVdt(x):
	x = x.reshape(-1,1)
	return DVDx(x).dot(f(x, -K.dot(x) + np.array([0, 0, g / kT]).reshape(3, 1)).reshape(-1,1))

x_lb = np.array([-1, -1, -10 / 180. * np.pi, -1, -1, -1, -10 / 180. * np.pi, -1, -1, -1])
x_ub = np.array([1, 1, 10 / 180. * np.pi, 1, 1, 1, 10 / 180. * np.pi, 1, 1, 1])


LHS = []
RHS = []

for _ in range(1000):
	x = x_lb + np.random.rand(*x_lb.shape) * (x_ub - x_lb)
	LHS.append(dVdt(x).item())
	RHS.append(V(x).item())

LHS = np.array(LHS)
RHS = np.array(RHS)

from IPython import embed; embed()
