"""Problem definition file for linear.

The differential equation is: dy/dx - m = 0

The analytical solution is: y = m*x + b.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


import numpy as np
import tensorflow as tf


# Define the slope and intercept of the line.
m = 1.0
b = 0.0

# Define the boundary condition at x = x0.
bc0 = b

# Define the boundaries.
x0 = 0.0
x1 = 1.0


def differential_equation(x, y, dy_dx):
    """The differential equation, in TensorFlow form.

    Parameters
    ----------
    x : tf.Variable, shape (n, 1)
        Independent variable values for computation of differential equation.
    y : tf.Tensor, shape (n, 1)
        Dependent variable values at each x-value. This is the array of the
        current estimates of the solution at each x-value.
    dy_dx : tf.Tensor, shape(n, 1)
        1st derivative values dy/dx at each x-value.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of equation at each x-value, nominally 0.
    """
    G = dy_dx - m
    return G


def compute_boundary_conditions(x):
    """Compute the boundary conditions, in NumPy form.

    Parameters
    ----------
    x : np.ndarray of float
        Values of x on the boundaries, shape (1,)

    Returns
    -------
    bc : np.ndarray of float
        Values of y on the boundaries, shape (1,)
    """
    nx = len(x)  # Should be 1 since this is a 1st-order ODE.
    bc = np.empty(nx)
    for (i, xx) in enumerate(x):
        if np.isclose(xx, x0):
            z = bc0
        else:
            raise Exception
        bc[i] = z
    return bc


def analytical_solution(x):
    """Analytical solution.

    Parameters
    ----------
    x : np.ndarray of float, shape (n, 1)
        Independent variable values for computation of solution.

    Returns
    -------
    y : np.ndarray of float, shape (n, 1)
        Analytical solution at each x-value.
    """
    y = m*x + b
    return y


def analytical_derivative(x):
    """Analytical derivative of solution.

    Parameters
    ----------
    x : np.ndarray of float (n, 1)
        Independent variable values for computation of solution.

    Returns
    -------
    dy_dx : np.ndarray of float, shape (n, 1)
        Analytical 1st derivative at each x-value.
    """
    dy_dx = np.empty((len(x),))
    dy_dx[...] = m
    return dy_dx


def create_training_data(nx):
    """Create the training data in NumPy form.

    Create and return a set of training data, consisting of points evenly
    spaced in x. Also return copies of the data containing only internal
    points, and only boundary points.

    Parameters
    ----------
    nx : int
        Number of points in x-dimension.

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
    x = np.linspace(x0, x1, nx)

    # Now split the training data into two groups - inside the BC, and on the
    # BC.
    # Initialize the mask to keep everything.
    mask = np.ones(len(x), dtype=bool)

    # Mask off the point at x = x0.
    mask[0] = False
    x_in = x[mask]

    # Invert the mask to get the boundary point.
    mask = np.logical_not(mask)
    x_bc = x[mask]

    # Return the training data and its subsets.
    return x, x_in, x_bc
