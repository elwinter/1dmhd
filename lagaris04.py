"""Problem definition file for Lagaris problem 4."""

import numpy as np
import tensorflow as tf


# Define the initial condition.
x0 = 0
x1 = 3
ic1 = 0.0
ic2 = 1.0


def differential_equation_1(X, Y, delY):
    x = X
    (y1, y2) = Y
    (dy1_dx, dy2_dx) = delY
    G = dy1_dx - tf.math.cos(x) - y1**2 - y2 + 1 + x**2 + tf.math.sin(x)**2
    return G


def differential_equation_2(X, Y, delY):
    x = X
    (y1, y2) = Y
    (dy1_dx, dy2_dx) = delY
    G = dy2_dx - 2*x + (1 + x**2)*tf.math.sin(x) - y1*y2
    return G


def analytical_solution_1(X):
    x = X
    Y = tf.math.sin(x)
    return Y


def analytical_solution_2(X):
    x = X
    Y = 1 + x**2
    return Y


def create_training_data(nx):
    """Create the training data.

    Create and return a set of training data, consisting of points evenly
    spaced in x. Also return copies of the data containing only internal
    points, and only boundary points.

    Parameters
    ----------
    nx : int
        Number of points in x--dimension.
    
    Returns
    -------
    x : np.ndarray, shape (nx,)
        Array of all x points.
    x_in : np.ndarray, shape (nx - 1,)
        Array of all interior points.
    x_bc : np.ndarray, shape (1,)
        Array of the single initial point.
    """
    # Create the array of all training points x.
    x = np.linspace(0, 1, nx)

    # Now split the training data into two groups - inside the BC, and on the BC.
    # Initialize the mask to keep everything.
    mask = np.ones(len(x), dtype=bool)
    # Mask off the point at x = 0.
    mask[0] = False
    x_in = x[mask]
    mask = np.logical_not(mask)
    x_bc = x[mask]
    return x, x_in, x_bc
