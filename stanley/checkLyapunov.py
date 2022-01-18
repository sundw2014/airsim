import numpy as np
import scipy
import scipy.optimize
from dynamics import f

P = np.array([[1,0],[0,1.5]])

def V(x):
	x = x.reshape(-1,1)
	return x.T.dot(P).dot(x)

def DVDx(x):
	x = x.reshape(-1,1)
	return 2 * x.T.dot(P)

def dVdt(x):
	x = x.reshape(-1,1)
	return DVDx(x).dot(f(x).reshape(-1,1))

x_lb = np.array([-1, -np.pi / 3])
x_ub = np.array([1, np.pi / 3])

LHS = []
RHS = []

for _ in range(1000):
	x = x_lb + np.random.rand(*x_lb.shape) * (x_ub - x_lb)
	LHS.append(dVdt(x).item())
	RHS.append(V(x).item())

LHS = np.array(LHS)
RHS = np.array(RHS)

_lambda = np.max(LHS / RHS)
print('lambda by samples: %.4f'%_lambda)

def _lambda(x):
    return -dVdt(x).item() / V(x).item()

x = scipy.optimize.brute(_lambda, np.array([x_lb, x_ub]).T.tolist(), finish=None)

# from IPython import embed; embed()

print('lambda by optimization: %.4f'%-_lambda(x))

