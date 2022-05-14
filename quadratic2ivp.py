"""Problem definition file for 2nd-order ODE IVP.

This problem definition file describes:

    d2y_dx2 - 2 = 0
    y(0) = 0
    dy_dx(0) = 0
    y(x) = x**2

The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the solution code.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""

import numpy as np


# Define the initial conditions.
x0 = 0
x1 = 1
bc0a = 0.0
bc0b = 0.0

def differential_equation(x, y, dy_dx, d2y_dx2):
    """2nd-order ODE IVP.

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
    G = d2y_dx2 - 2.0
    return G


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
    # Mask off the point at x = 0 (only, since IVP).
    mask[0] = False
    x_in = x[mask]
    mask = np.logical_not(mask)
    x_bc = x[mask]
    return x, x_in, x_bc
