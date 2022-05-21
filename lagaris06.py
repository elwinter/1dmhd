"""Problem definition file for Lagaris problem 6."""

import numpy as np
from numpy import pi, sin
import tensorflow as tf


# Define the boundaries.
x0 = 0
x1 = 1
y0 = 0
y1 = 1


def differential_equation(xy, Y, delY, del2Y):
    n = xy.shape[0]
    x = tf.reshape(xy[:, 0], (n, 1))
    y = tf.reshape(xy[:, 1], (n, 1))
    (d2Y_dx2, d2Y_dy2) = del2Y
    G = d2Y_dx2 + d2Y_dy2 - (2 - pi**2*y**2)*tf.math.sin(pi*x)
    return G


def compute_boundary_conditions(xy):
    nxy = len(xy)
    bc = np.empty(nxy)
    for (i, (x, y)) in enumerate(xy):
        if np.isclose(x, x0):
            z = 0
        elif np.isclose(x, x1):
            z = 0
        elif np.isclose(y, y0):
            z = 0
        elif np.isclose(y, y1):
            z = 2*sin(pi*x)
        else:
            raise Exception
        bc[i] = z
    return bc


def analytical_solution(xy):
    n = xy.shape[0]
    x = tf.reshape(xy[:, 0], (n, 1))
    y = tf.reshape(xy[:, 1], (n, 1))
    return y**2*tf.math.sin(pi*x)


def create_training_data(nx, ny):
    # Create the arrays of all training points x, y.
    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)
    X = np.repeat(x, ny)
    Y = np.tile(y, nx)
    xy = np.vstack([X, Y]).T

    # Now split the training data into two groups - inside the BC, and on the BC.
    # Initialize the mask to keep everything.
    mask = np.ones(nx*ny, dtype=bool)
    # Mask off the points at x = 0.
    mask[:ny] = False
    # Mask off the points at x = 1.
    mask[-ny:] = False
    # Mask off the points at y = 0.
    mask[::ny] = False
    # Mask off the points at y = 1
    mask[ny - 1::ny] = False
    xy_in = xy[mask]
    mask = np.logical_not(mask)
    xy_bc = xy[mask]
    return xy, xy_in, xy_bc
