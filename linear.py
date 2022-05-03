"""Problem definition file for a simple linear ODE.

This problem definition file describes:

    dy/dx = 1
    y(0) = 0
    y(x) = x

The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the solution code.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""

import numpy as np
import tensorflow as tf


# Define the initial condition.
ic = 0.0


# Define the ODE.

def ode(X, Y, delY):
    n = X.shape[0]
    x = tf.reshape(X[:, 0], (n, 1))
    y = Y
    (del_y,) = delY
    dy_dx = tf.reshape(del_y[:, 0], (n, 1))
    G = dy_dx - 1
    return G


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
