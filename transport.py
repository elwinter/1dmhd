"""Problem definition file for the transport equation.

Reference:
http://ramanujan.math.trinity.edu/rdaileda/teach/s14/m3357/lectures/lecture_1_21_slides.pdf
"""


import numpy as np
import tensorflow as tf


# Define the boundaries.
x0 = 0
x1 = 1
t0 = 0
t1 = 1

# Transport speed
v = 0.1


def differential_equation(xt, Y, delY):
    n = xt.shape[0]
    x = tf.reshape(xt[:, 0], (n, 1))
    t = tf.reshape(xt[:, 1], (n, 1))
    dY_dx = tf.reshape(delY[:, 0], (n, 1))
    dY_dt = tf.reshape(delY[:, 1], (n, 1))
    G = dY_dt + v*dY_dx
    return G


def compute_boundary_conditions(xt):
    n = len(xt)
    bc = np.empty(n)
    for (i, (x, t)) in enumerate(xt):
        if np.isclose(t, t0):
            Y = x*np.exp(-x**2)
        else:
            raise Exception
        bc[i] = Y
    return bc


x_center = 0.2

def f0(x):
    dx = x - x_center
    Y = tf.math.exp(dx**2)
    return Y

def analytical_solution(xt):
    n = xt.shape[0]
    x = tf.reshape(xt[:, 0], (n, 1))
    t = tf.reshape(xt[:, 1], (n, 1))
    return f0(x - v*t)


def create_training_data(nx, nt):
    # Create the arrays of all training points x, t.
    x = np.linspace(x0, x1, nx)
    t = np.linspace(t0, t1, nt)
    X = np.repeat(x, nt)
    T = np.tile(t, nx)
    xt = np.vstack([X, T]).T

    # Now split the training data into two groups - inside the BC,
    # and on the BC.
    # Initialize the mask to keep everything.
    mask = np.ones(nx*nt, dtype=bool)
    # Mask off the points at t = 0.
    mask[::nt] = False
    xt_in = xt[mask]
    mask = np.logical_not(mask)
    xt_bc = xt[mask]
    return xt, xt_in, xt_bc
