import numpy as np
import scipy
import scipy.optimize
from dynamics import dfdx, dudx

x_lb = np.array([-1, -np.pi / 3])
x_ub = np.array([1, np.pi / 3])

def objective(x):
	return - np.linalg.norm(dfdx(x), ord=2)

x = scipy.optimize.brute(objective, np.array([x_lb, x_ub]).T.tolist(), finish=None)

# from IPython import embed; embed()

print('Lipschitz constant of f by optimization: %.4f'%-objective(x))

def objective(x):
	return - np.sqrt((dudx(x)**2).sum())

x = scipy.optimize.brute(objective, np.array([x_lb, x_ub]).T.tolist(), finish=None)

# from IPython import embed; embed()

print('Lipschitz constant of u by optimization: %.4f'%-objective(x))
