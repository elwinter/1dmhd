"""Problem definition file for Lagaris problem 3 (IVP)."""


import numpy as np
import tensorflow as tf


# Define the boundaries.
x0 = 0
x1 = 1

# Define the boundary conditions at x = x0.
bc0_0 = 0.0
bc0_1 = 1.0


def differential_equation(x, y, dy_dx, d2y_dx2):
    """Linear 2nd-order ODE.

    Parameters
    ----------
    x : tf.Variable, shape (n, 1)
        Independent variable values for computation of ODE.
    y : tf.Tensor, shape (n, 1)
        Dependent variable values at each x-value. This is the array of the
        current estimates of the solution at each x-value.
    dy_dx : tf.Tensor, shape(n, 1)
        1st derivative values for y at each x-value.
    d2y_dx2 : tf.Tensor, shape(n, 1)
        2nd derivative values for y at each x-value.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of equation at each x-value.
    """
    G = d2y_dx2 + 0.2*dy_dx + y + tf.exp(-0.2*x)*tf.cos(x)
    return G


def compute_boundary_conditions(x):
    """Compute the boundary conditions.

    Parameters
    ----------
    x : np.ndarray of float
        Value of x on the boundary, shape (1,)

    Returns
    -------
    bc0 : np.ndarray of float
        Values of y on the boundary, shape (1,)
    bc1 : np.ndarray of float
        Values of dy/dx on the boundary, shape (1,)
    """
    nx = len(x)
    bc0 = np.empty(nx)
    bc1 = np.empty(nx)
    for (i, xx) in enumerate(x):
        if np.isclose(xx, x0):
            z0 = bc0_0
            z1 = bc0_1
        else:
            raise Exception
        bc0[i] = z0
        bc1[i] = z1
    return bc0, bc1


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
    y = tf.math.exp(-x/5)*tf.math.sin(x)
    return y


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
    dy_dx = 0.2*tf.math.exp(-0.2*x)*(5.0*tf.math.cos(x) - tf.math.sin(x))
    return dy_dx


def analytical_2nd_derivative(x):
    """Analytical 2nd derivative of solution.

    Analytical 2nd derivative of solution.

    n is the number of evaluation points for the equation,
    equal to the length of x.

    Parameters
    ----------
    x : tf.Variable, shape (n, 1)
        Independent variable values for computation of solution.

    Returns
    -------
    d2y_dx2 : tf.Tensor, shape (n, 1)
        Analytical 2nd derivative at each x-value.
    """
    d2y_dx2 = (
        0.2*tf.math.exp(-0.2*x)*(-5.*tf.math.sin(x) - tf.math.cos(x))
        - 0.04*tf.math.exp(-0.2*x)*(5.0*tf.math.cos(x) - tf.math.sin(x))
    )
    return d2y_dx2


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
    x_bc : np.ndarray, shape (2,)
        Array of the initial and final points.
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
    mask = np.logical_not(mask)
    x_bc = x[mask]
    return x, x_in, x_bc
