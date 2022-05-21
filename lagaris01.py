"""Problem definition file for Lagaris problem 1."""

import numpy as np
import tensorflow as tf


# Define the boundaries.
x0 = 0
x1 = 1

# Define the boundary condition at x = x0.
bc0 = 1.0


def differential_equation(x, y, dy_dx):
    """Linear first-order ODE.

    Parameters
    ----------
    x : tf.Variable, shape (n, 1)
        Independent variable values for computation of ODE.
    y : tf.Tensor, shape (n, 1)
        Dependent variable values at each x-value. This is the array of the
        current estimates of the solution at each x-value.
    dy_dx : tf.Tensor, shape(n, m)
        1st derivative values for y at each x-value.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of equations at each x-value, nominally 0.
    """
    G = (
        dy_dx + (x + (1 + 3*x**2)/(1 + x + x**3))*y - x**3 -
        2*x - x**2*(1 + 3*x**2)/(1 + x + x**3)
    )
    return G


def compute_boundary_conditions(x):
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
    equal to the length of X.

    m is the number of independent variables (1 for ODE).

    neq is the number of equations being solved (1 for ODE), and is
    assumed to be the same as the number of dependent variables.

    Parameters
    ----------
    x : tf.Variable, each shape (n, m)
        Independent variable values for computation of solution.

    Returns
    -------
    Y : tf.Tensor, shape (n, neq)
        Value of equations at each x-value.
    """
    Y = tf.math.exp(-x**2/2)/(1 + x + x**3) + x**2
    return Y


def analytical_derivative(x):
    """Analytical derivative of solution.

    Analytical derivative of solution.

    n is the number of evaluation points for the equation,
    equal to the length of X.

    m is the number of independent variables (1 for ODE).

    neq is the number of equations being solved (1 for ODE), and is
    assumed to be the same as the number of dependent variables.

    Parameters
    ----------
    x : tf.Variable, each shape (n, m)
        Independent variable values for computation of solution.

    Returns
    -------
    dy_dx : tf.Tensor, shape (n, 1)
        Analytical 1st derivative at each x-value.
    """
    dy_dx = 2*x - tf.math.exp(-x**2/2)*(1 + x + 4*x**2 + x**4)/(1 + x + x**3)**2
    return dy_dx


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
