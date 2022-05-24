"""Problem definition file for quadratic 2nd-order ODE BVP."""


import numpy as np
import tensorflow as tf


# Define the boundaries.
x0 = 0.0
x1 = 1.0

# Define the boundary condition at x = x0.
bc0 = 0.0
bc1 = 0.0


def differential_equation(x, y, dy_dx, d2y_dx2):
    """2nd-order ODE BVP.

    Parameters
    ----------
    x : tf.Variable, shape (n, 1)
        Independent variable values for computation of ODE.
    y : tf.Tensor, shape(n, 1)
        Dependent variable value at each x-value.
    dy_dx : tf.Tensor, shape(n, 1)
        Value of dy/dx at each x-value.
    d2y_dx2 : tf.Tensor, each shape(n, 1)
        Value of d2y/dx2 at each x-value.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of equation at each x-value.
    """
    G = y - 0.5*x*dy_dx + d2y_dx2 - 2
    return G


def compute_boundary_conditions(x):
    """Compute the boundary conditions.

    Parameters
    ----------
    x : np.ndarray of float
        Values of x on the boundaries, shape (2,)

    Returns
    -------
    bc : np.ndarray of float
        Values of y on the boundaries, shape (2,)
    """
    nx = len(x)
    bc = np.empty(nx)
    for (i, xx) in enumerate(x):
        if np.isclose(xx, x0):
            z = bc0
        elif np.isclose(xx, x1):
            z = bc1
        else:
            raise Exception
        bc[i] = z
    return bc


def analytical_solution(x):
    """Analytical solution to differential_equation.

    Analytical solution to differential_equation.

    n is the number of evaluation points for the equation,
    equal to the length of x.

    Parameters
    ----------
    x : tf.Variable, shape (n, 1)
        Independent variable values for computation of solution.

    Returns
    -------
    y : tf.Tensor, shape (n, neq)
        Value of equation at each x-value.
    """
    y = x**2
    return y


def analytical_1st_derivative(x):
    """Analytical 1st derivative of solution.

    Analytical 1st derivative of solution.

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
    dy_dx = 2*x
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
    n = len(x)
    d2y_dx2 = np.empty((n, 1))
    d2y_dx2[...] = 2.0
    return d2y_dx2


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
    x_in : np.ndarray, shape (nx - 2,)
        Array of all interior points.
    x_bc : np.ndarray, shape (2,)
        Array of the first and last points.
    """
    # Create the array of all training points x.
    x = np.linspace(x0, x1, nx)

    # Now split the training data into two groups - inside the BC, and on the BC.
    # Initialize the mask to keep everything.
    mask = np.ones(len(x), dtype=bool)
    # Mask off the point at x = 0 and x = 1.
    mask[0] = False
    mask[-1] = False
    x_in = x[mask]
    mask = np.logical_not(mask)
    x_bc = x[mask]
    return x, x_in, x_bc
