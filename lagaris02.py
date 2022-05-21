"""Problem definition file for Lagaris problem 2."""


import numpy as np
import tensorflow as tf


# Define the boundaries.
x0 = 0
x1 = 2


# Define the initial condition.
bc0 = 0.0


def differential_equation(x, y, dy_dx):
    """Linear first-order ODE.

    Parameters
    ----------
    x : tf.Variable, shape (n, 1)
        Independent variable values for computation of ODE.
    y : tf.Tensor, shape (n, 1)
        Dependent variable values at each x-value. This is the array of the
        current estimates of the solution at each x-value.
    dy_dx : tf.Tensor, shape(n, 1)
        1st derivative values for y at each x-value.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of equation at each x-value, nominally 0.
    """
    G = dy_dx + y/5 - tf.math.exp(-x/5)*tf.math.cos(x)
    return G


def compute_boundary_conditions(x):
    """Compute the boundary conditions.

    Parameters
    ----------
    x : np.ndarray of float
        Values of x on the boundaries, shape (1,)

    Returns
    -------
    bc : np.ndarray of float
        Values of y on the boundaries, shape (1,)
    """
    nx = len(x)
    bc = np.empty(nx)
    for (i, xx) in enumerate(x):
        if np.isclose(xx, x0):
            z = bc0
        else:
            raise Exception
        bc[i] = z
    return bc


def analytical_solution(x):
    """Analytical solution to ODE.

    Analytical solution to linear ODE.

    n is the number of evaluation points for the equation,
    equal to the length of x.

    Parameters
    ----------
    x : tf.Variable, shape (n, 1)
        Independent variable values for computation of solution.

    Returns
    -------
    y : tf.Tensor, shape (n, 1)
        Analytical solution at each x-value.
    """
    Y = tf.math.exp(-x/5)*tf.math.sin(x)
    return Y


def analytical_derivative(x):
    """Analytical derivative of solution.

    Analytical derivative of solution.

    n is the number of evaluation points for the equation,
    equal to the length of x.

    Parameters
    ----------
    x : tf.Variable, shape (n, 1)
        Independent variable values for computation of solution.

    Returns
    -------
    dy_dx : tf.Tensor, shape (n, 1)
        Analytical 1st derivative at each x-value.
    """
    dy_dx = 0.2*tf.math.exp(-x/5)*(5.0*tf.math.cos(x) - tf.math.sin(x))
    return dy_dx


def create_training_data(nx):
    """Create the training data.

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

    # Now split the training data into two groups - inside the BC, and on the BC.
    # Initialize the mask to keep everything.
    mask = np.ones(len(x), dtype=bool)
    # Mask off the point at x = x0.
    mask[0] = False
    x_in = x[mask]
    mask = np.logical_not(mask)
    x_bc = x[mask]
    return x, x_in, x_bc
