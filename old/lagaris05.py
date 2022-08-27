"""Problem definition file for Lagaris problem 5."""


import numpy as np
import tensorflow as tf


# Define the boundaries.
x0 = 0
x1 = 1
y0 = 0
y1 = 1


def differential_equation(xy, Y, delY, del2Y):
    """2nd-order PDE.

    Parameters
    ----------
    xy : tf.Variable, shape (n, 2)
        Independent variable (x, y) values for computation of PDE.
    Y : tf.Tensor, shape (n, 1)
        Dependent variable values at each (x, y)-value. This is the array of
        the current estimates of the solution at each point.
    delY : tf.Tensor, shape(n, 2)
        1st derivative values wrt (x, y) at each point.
    del2Y : tf.Tensor, shape(n, 2)
        2nd derivative values wrt (x, y) at each point.

    Returns
    -------
    G : tf.Tensor, shape (n, 1)
        Value of equation at each point.
    """
    n = xy.shape[0]
    x = tf.reshape(xy[:, 0], (n, 1))
    y = tf.reshape(xy[:, 1], (n, 1))
    # dY_dx = tf.reshape(delY[:, 0], (n, 1))
    # dY_dy = tf.reshape(delY[:, 1], (n, 1))
    d2Y_dx2 = tf.reshape(del2Y[:, 0], (n, 1))
    d2Y_dy2 = tf.reshape(del2Y[:, 1], (n, 1))
    G = d2Y_dx2 + d2Y_dy2 - (x - 2 + y**3 + 6*y)*tf.math.exp(-x)
    return G


def compute_boundary_conditions(xy):
    """Compute the boundary conditions.

    Parameters
    ----------
    xy : np.ndarray of float, shape (n_bc, 2)
        Values of (x, y) on the boundaries

    Returns
    -------
    bc : np.ndarray of float
        Values of Y on the boundaries, shape (n_bc,)
    """
    nxy = len(xy)
    bc = np.empty(nxy)
    for (i, (x, y)) in enumerate(xy):
        if np.isclose(x, x0):
            z = y**3
        elif np.isclose(x, x1):
            z = (1 + y**3)/np.e
        elif np.isclose(y, y0):
            z = x*np.exp(-x)
        elif np.isclose(y, y1):
            z = (x + 1)*np.exp(-x)
        else:
            raise Exception
        bc[i] = z
    return bc


def analytical_solution(xy):
    """Analytical solution.

    Analytical solution.

    n is the number of evaluation points for the equation,
    equal to the length of xy.

    Parameters
    ----------
    xy : tf.Variable, shape (n, 2)
        Independent variable values (x, y) for computation of solution.

    Returns
    -------
    Y : tf.Tensor, shape (n, 1)
        Analytical solution at each point.
    """
    n = xy.shape[0]
    x = tf.reshape(xy[:, 0], (n, 1))
    y = tf.reshape(xy[:, 1], (n, 1))
    return (x + y**3)*tf.math.exp(-x)


def analytical_x_derivative_1(xy):
    """Analytical 1st x-derivative of solution.

    Analytical 1st x-derivative of solution.

    n is the number of evaluation points for the equation,
    equal to the length of xy.

    Parameters
    ----------
    xy : tf.Variable, shape (n, 2)
        Independent variable values (x, y) for computation of solution.

    Returns
    -------
    dY_dx : tf.Tensor, shape (n, 1)
        Analytical 1st x-derivative at each point.
    """
    n = len(xy)
    x = tf.reshape(xy[:, 0], (n, 1))
    y = tf.reshape(xy[:, 1], (n, 1))
    dY_dx = -tf.math.exp(-x)*(x + y**3 - 1)
    return dY_dx


def analytical_y_derivative_1(xy):
    """Analytical 1st y-derivative of solution.

    Analytical 1st y-derivative of solution.

    n is the number of evaluation points for the equation,
    equal to the length of xy.

    Parameters
    ----------
    xy : tf.Variable, shape (n, 2)
        Independent variable values (x, y) for computation of solution.

    Returns
    -------
    dY_dy : tf.Tensor, shape (n, 1)
        Analytical 1st y-derivative at each point.
    """
    n = len(xy)
    x = tf.reshape(xy[:, 0], (n, 1))
    y = tf.reshape(xy[:, 1], (n, 1))
    dY_dy = 3.0*tf.math.exp(-x)*y**2
    return dY_dy


def analytical_x_derivative_2(xy):
    """Analytical 2nd x-derivative of solution.

    Analytical 2nd x-derivative of solution.

    n is the number of evaluation points for the equation,
    equal to the length of xy.

    Parameters
    ----------
    xy : tf.Variable, shape (n, 2)
        Independent variable values (x, y) for computation of solution.

    Returns
    -------
    d2Y_dx2 : tf.Tensor, shape (n, 1)
        Analytical 2nd x-derivative at each point.
    """
    n = len(xy)
    x = tf.reshape(xy[:, 0], (n, 1))
    y = tf.reshape(xy[:, 1], (n, 1))
    d2Y_dx2 = tf.math.exp(-x)*(x + y**3 - 2)
    return d2Y_dx2


def analytical_y_derivative_2(xy):
    """Analytical 2nd y-derivative of solution.

    Analytical 2nd y-derivative of solution.

    n is the number of evaluation points for the equation,
    equal to the length of xy.

    Parameters
    ----------
    xy : tf.Variable, shape (n, 2)
        Independent variable values (x, y) for computation of solution.

    Returns
    -------
    d2Y_dy2 : tf.Tensor, shape (n, 1)
        Analytical 2nd y-derivative at each point.
    """
    n = len(xy)
    x = tf.reshape(xy[:, 0], (n, 1))
    y = tf.reshape(xy[:, 1], (n, 1))
    d2Y_dy2 = 6.0*tf.math.exp(-x)*y
    return d2Y_dy2


def create_training_data(nx, ny):
    """Create the training data.

    Create and return a set of training data, consisting of points evenly
    spaced in x and y. Also return copies of the data containing only internal
    points, and only boundary points.

    Parameters
    ----------
    nx, ny : int
        Number of points in x-dimension and y-dimension.

    Returns
    -------
    xy : np.ndarray, shape (nx*ny, 2)
        Array of all x points.
    xy_in : np.ndarray, shape ((nx - 2)*(ny - 2), 2)
        Array of all interior points.
    xy_bc : np.ndarray, shape (2*(nx + ny - 2), 2)
        Array of the boundary points.
    """
    # Create the arrays of all training points x, y.
    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)
    X = np.repeat(x, ny)
    Y = np.tile(y, nx)
    xy = np.vstack([X, Y]).T

    # Now split the training data into two groups - inside the BC, and on the
    # BC.
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
