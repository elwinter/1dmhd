"""Problem definition file for 2nd-order ODE IVP.

This module implements problem 3 (IVP version) in Lagaris et al (1998)
(2nd order ODE IVP).

Note that an upper-case 'Y' is used to represent the Greek psi from the
original equation.

The equation is defined on the domain [0, 1]:

The analytical form of the equation is:
    G(x, Y, dY/dx, d2Y/dx2) = d2Y/dx2 + 1/5 dY/dx + Y + 1/5 exp(-x/5)*cos(x) = 0

with boundary conditions:

y(0) = 0
dy_dx(0) = 1

This equation has the analytical solution for the supplied initial
conditions:

Ya(x) = exp(-x/5)*sin(x)

Reference:

Isaac Elias Lagaris, Aristidis Likas, and Dimitrios I. Fotiadis,
"Artificial Neural Networks for Solving Ordinary and Partial Differential
Equations", *IEEE Transactions on Neural Networks* **9**(5), pp. 987-999,
1998

The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the solution code.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""

import math as m
import numpy as np
import tensorflow as tf


# Define the boundary conditions.
x0 = 0
x1 = 1
bc0a = 0.0
bc0b = 1.0

def differential_equation(X, Y, delY, del2Y):
    """Linear 2nd-order ODE.

    Equation 29 from Lagaris et al (1998).

    n is the number of evaluation points for the equation,
    equal to the length of X.

    m is the number of independent variables (1 for ODE).

    neq is the number of equations being solved (1 for ODE), and is
    assumed to be the same as the number of dependent variables.

    Parameters
    ----------
    X : tf.Variable, each shape (n, m)
        Independent variable values for computation of ODE.
    Y : List of neq tf.Tensor, each shape(n, 1)
        Dependent variable values at each x-value.
    delY : List of neq tf.Tensor, each shape(n, m)
        Gradient values at each x-value.
    del2Y : List of neq tf.Tensor, each shape(n, m)
        2nd derivative values at each x-value.

    Returns
    -------
    G : tf.Tensor, shape (n, neq)
        Value of equations at each x-value.
    """
    x = X
    y = Y[0]
    dy_dx = delY[0]
    d2y_dx2 = del2Y[0]
    G = d2y_dx2 + 0.2*dy_dx + y + tf.exp(-0.2*x)*tf.cos(x)
    return G


def analytical_solution(X):
    """Analytical solution to ODE.

    Analytical solution to linear ODE.

    n is the number of evaluation points for the equation,
    equal to the length of X.

    m is the number of independent variables (1 for ODE).

    neq is the number of equations being solved (1 for ODE), and is
    assumed to be the same as the number of dependent variables.

    Parameters
    ----------
    X : tf.Variable, each shape (n, m)
        Independent variable values for computation of solution.

    Returns
    -------
    Y : tf.Tensor, shape (n, neq)
        Value of equations at each x-value.
    """
    x = X
    Y = tf.math.exp(-x/5)*tf.math.sin(x)
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
