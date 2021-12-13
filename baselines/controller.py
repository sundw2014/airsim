# 3D Control of Quadcopter
# based on https://github.com/juanmed/quadrotor_sim/blob/master/3D_Quadrotor/3D_control_with_body_drag.py
# The dynamics is from pp. 17, Eq. (2.22). https://www.kth.se/polopoly_fs/1.588039.1550155544!/Thesis%20KTH%20-%20Francesco%20Sabatino.pdf
# The linearization is from Different Linearization Control Techniques for
# a Quadrotor System (many typos)

import dynamics
from dynamics import g
import numpy as np
import scipy
from scipy.integrate import odeint

def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u
    """
    # http://www.mwm.im/lqr-controllers-with-python/
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R) * (B.T * X))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return np.asarray(K), np.asarray(X), np.asarray(eigVals)

####################### linearization ##################
# The state variables are x, y, z, vx, vy, vz, theta_x, theta_y
A = np.zeros([8,8])
A[0, 3] = 1.
A[1, 4] = 1.
A[2, 5] = 1.
A[3, 6] = g
A[4, 7] = g
B = np.zeros([8, 3])
B[5, 0] = 1.
B[6, 1] = 1.
B[7, 2] = 1.

####################### solve LQR #######################
n = A.shape[0]
m = B.shape[1]
Q = np.eye(n)
Q[0, 0] = 10.
Q[1, 1] = 10.
Q[2, 2] = 10.
R = np.diag([1., 1., 1.])
K, _, _ = lqr(A, B, Q, R)

####################### The controller ######################
def u(x, goal):
    goal = np.array(goal)
    return K.dot(np.array(goal.reshape(-1).tolist()+[0, ] * 5) - x)

######################## The closed_loop system #######################
def cl_nonlinear(x, t, goal):
    x = np.array(x)
    dot_x = dynamics.f(x, u(x, goal))
    return dot_x

# simulate
def simulate(x, goal, dt):
    curr_position = np.array(x)[:3]
    error = goal - curr_position
    distance = np.sqrt((error**2).sum())
    if distance > 1:
        goal = curr_position + error / distance
    return odeint(cl_nonlinear, x, [0, dt], args=(goal,))[-1]
