"""Problem definition file for a simple 1-D MHD problem.

This problem definition file describes an Alfven wave: unit pressure and
density, with a constant axial magnetic field (B0x = constant).

The problem is defined on the domain 0 <= (x, t) <= 1. Velocity is initially
0. The converged solution should show wave propagation. This version uses
periodic boundary conditions in x.

The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by the TensorFlow
code.

NOTE: In all code, below, the following indices are assigned to physical
variables (all are perturbations to initial values):

0: rho1  # mass density perturbation
1: v1x   # x-component of velocity perturbation
2: v1y   # y-component of velocity perturbation
3: v1z   # z-component of velocity perturbation
4: B1y   # y-component of magnetic field perturbation
5: B1z   # z-xomponent of magnetic field perturbation

Pressure P is a function of the initial conditions and rho.

These equations are derived from the ideal MHD equations developed in Russel
et al, applying the assumptions used for Alfven waves (sections 3.5, 3.6 in
Russell et al, 2018).

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


import numpy as np
import tensorflow as tf


# Names of dependent variables.
variable_names = ["rho1", "v1x", "v1y", "v1z", "B1y", "B1z"]
# variable_names = ["v1y", "B1y"]

# Number of dependent variables.
n_var = len(variable_names)

# Number of independent variables.
n_dim = 2  # (x, t)

# Define the problem domain.
x0 = 0.0
x1 = 1.0
t0 = 0.0
t1 = 1.0

# Adiabatic index.
gamma = 1.4

# Density and perturbation at t = 0, for all x.
rho0 = 1.0
rho10 = 0.0

# Velocity components at t = 0, for all x.
v0x = 0.0
v0y = 0.0
v0z = 0.0

# Velocity perturbation components at t = 0, for all x.
v1x0 = 0.0
v1y0 = 0.0
v1z0 = 0.0

# Magnetic field components at t = 0, for all x.
B0x = 1.0
B0y = 0.0
B0z = 0.0

# Magnetic field perturbation components at t = 0, for all x.
B1x0 = 0.0
B1y0 = 0.0
B1z0 = 0.0

# Initial pressure at t = 0 for all x.
P0 = 1.0

# Alfven speed.
C_alfven = B0x/np.sqrt(rho0)

# Wavelength and wavenumber of initial perturbation.
wavelength = 1.0
kx = 2*np.pi/wavelength

# Frequency and angular frequency of initial perturbation.
f = C_alfven/wavelength
w = 2*np.pi*f

# Amplitude of initial perturbations in vy and By.
a_vy = 0.1
a_By = 0.1


def create_training_data(nx, nt):
    """Create the training data.

    Create and return a set of training data of points evenly spaced in x and
    t. Flatten the data to a list of pairs of points. Also return copies
    of the data containing only internal points, and only boundary points.

    Parameters
    ----------
    nx, nt : int
        Number of points in x- and t- dimensions.

    Returns
    -------
    xt : np.ndarray, shape (nx*nt, 2)
        Array of all [x, t] points.
    xt_in : np.ndarray, shape ((nx - 1)*(nt - 2)), 2)
        Array of all [x, t] points within boundary.
    xt_bc : np.ndarray, shape (nx + 2*(nt - 1), 2)
        Array of all [x, t] points at boundary.
    """
    # Create the array of all training points (x, t), looping over t then x.
    x = np.linspace(x0, x1, nx)
    t = np.linspace(t0, t1, nt)
    X = np.repeat(x, nt)
    T = np.tile(t, nx)
    xt = np.vstack([X, T]).T

    # Now split the training data into two groups - inside the BC, and on the
    # BC.
    # Initialize the mask to keep everything.
    mask = np.ones(len(xt), dtype=bool)
    # Mask off the points at x = 0.
    mask[:nt] = False
    # Mask off the points at x = 1.
    mask[-nt:] = False
    # Mask off the points at t = 0.
    mask[::nt] = False

    # Extract the internal points.
    xt_in = xt[mask]

    # Invert the mask and extract the boundary points.
    mask = np.logical_not(mask)
    xt_bc = xt[mask]
    return xt, xt_in, xt_bc


def compute_boundary_conditions(xt):
    """Compute the boundary conditions.

    Parameters
    ----------
    xt : np.ndarray of float
        Values of (x, t) on the boundaries, shape (n_bc, 2)

    Returns
    -------
    bc : np.ndarray of float, shape (n_bc, n_var)
        Values of each dependent variable on boundary.
    """
    n = len(xt)
    bc = np.empty((n, n_var))
    for (i, (x, t)) in enumerate(xt):
        if np.isclose(x, x0):
            # Periodic perturbation at x = 0.
            bc[i, :] = [
                rho10,
                v1x0,
                a_vy*np.sin(-w*t),
                v1z0,
                a_By*np.sin(-w*t - np.pi),
                B1z0
            ]
        elif np.isclose(x, x1):
            # Periodic perturbation at x = 1, same as at x = 0.
            bc[i, :] = [
                rho10,
                v1x0,
                a_vy*np.sin(-w*t),
                v1z0,
                a_By*np.sin(-w*t - np.pi),
                B1z0
            ]
        elif np.isclose(t, t0):
            bc[i, :] = [
                rho10,
                v1x0,
                a_vy*np.sin(kx*x),
                v1z0,
                a_By*np.sin(kx*x + np.pi),
                B1z0
            ]
        else:
            raise ValueError
    return bc


# Define the differential equations using TensorFlow operations.

# @tf.function
def pde_rho1(xt, Y1, del_Y1):
    """Differential equation for rho1.

    Evaluate the differential equation for rho1 (density perturbation).

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y1 : list of n_var tf.Tensor, each shape (n, 1)
        Perturbations of dependent variables at each training point.
    del_Y1 : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradients of Y1 wrt (x, t) at each training point.

    Returns
    -------
    G : tf.Tensor, shape(n, 1)
        Value of differential equation at each training point.
    """
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    (rho1, v1x, v1y, v1z, B1y, B1z) = Y1
    (del_rho1, del_v1x, del_v1y, del_v1z, del_B1y, del_B1z) = del_Y1
    # drho1_dx = tf.reshape(del_rho1[:, 0], (n, 1))
    drho1_dt = tf.reshape(del_rho1[:, 1], (n, 1))
    dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
    # dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
    # dv1y_dx = tf.reshape(del_v1y[:, 0], (n, 1))
    # dv1y_dt = tf.reshape(del_v1y[:, 1], (n, 1))
    # dv1z_dx = tf.reshape(del_v1z[:, 0], (n, 1))
    # dv1z_dt = tf.reshape(del_v1z[:, 1], (n, 1))
    # dB1y_dx = tf.reshape(del_B1y[:, 0], (n, 1))
    # dB1y_dt = tf.reshape(del_B1y[:, 1], (n, 1))
    # dB1z_dx = tf.reshape(del_B1z[:, 0], (n, 1))
    # dB1z_dt = tf.reshape(del_B1z[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = drho1_dt + rho0*dv1x_dx
    return G


# @tf.function
def pde_v1x(xt, Y1, del_Y1):
    """Differential equation for v1x.

    Evaluate the differential equation for v1x (x-component of velocity
    perturbation).

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y1 : list of n_var tf.Tensor, each shape (n, 1)
        Perturbations of dependent variables at each training point.
    del_Y1 : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradients of Y1 wrt (x, t) at each training point.

    Returns
    -------
    G : tf.Tensor, shape(n, 1)
        Value of differential equation at each training point.
    """
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    # (rho1, v1x, v1y, v1z, B1y, B1z) = Y1
    (del_rho1, del_v1x, del_v1y, del_v1z, del_B1y, del_B1z) = del_Y1
    drho1_dx = tf.reshape(del_rho1[:, 0], (n, 1))
    # drho1_dt = tf.reshape(del_rho1[:, 1], (n, 1))
    # dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
    dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
    # dv1y_dx = tf.reshape(del_v1y[:, 0], (n, 1))
    # dv1y_dt = tf.reshape(del_v1y[:, 1], (n, 1))
    # dv1z_dx = tf.reshape(del_v1z[:, 0], (n, 1))
    # dv1z_dt = tf.reshape(del_v1z[:, 1], (n, 1))
    # dB1y_dx = tf.reshape(del_B1y[:, 0], (n, 1))
    # dB1y_dt = tf.reshape(del_B1y[:, 1], (n, 1))
    # dB1z_dx = tf.reshape(del_B1z[:, 0], (n, 1))
    # dB1z_dt = tf.reshape(del_B1z[:, 1], (n, 1))

    # Compute the presure derivative from the density derivative.
    dP1_dx = gamma*P0/rho0*drho1_dx

    # G is a Tensor of shape (n, 1).
    G = rho0*dv1x_dt + dP1_dx
    return G


# @tf.function
def pde_v1y(xt, Y1, del_Y1):
    """Differential equation for v1y.

    Evaluate the differential equation for v1y (y-component of velocity
    perturbation).

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y1 : list of n_var tf.Tensor, each shape (n, 1)
        Perturbations of dependent variables at each training point.
    del_Y1 : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradients of Y1 wrt (x, t) at each training point.

    Returns
    -------
    G : tf.Tensor, shape(n, 1)
        Value of differential equation at each training point.
    """
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    # (rho1, v1x, v1y, v1z, B1y, B1z) = Y1
    (del_rho1, del_v1x, del_v1y, del_v1z, del_B1y, del_B1z) = del_Y1
#    (del_v1y, del_B1y) = del_Y1
    # drho1_dx = tf.reshape(del_rho1[:, 0], (n, 1))
    # drho1_dt = tf.reshape(del_rho1[:, 1], (n, 1))
    # dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
    # dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
    # dv1y_dx = tf.reshape(del_v1y[:, 0], (n, 1))
    dv1y_dt = tf.reshape(del_v1y[:, 1], (n, 1))
    # dv1z_dx = tf.reshape(del_v1z[:, 0], (n, 1))
    # dv1z_dt = tf.reshape(del_v1z[:, 1], (n, 1))
    dB1y_dx = tf.reshape(del_B1y[:, 0], (n, 1))
    # dB1y_dt = tf.reshape(del_B1y[:, 1], (n, 1))
    # dB1z_dx = tf.reshape(del_B1z[:, 0], (n, 1))
    # dB1z_dt = tf.reshape(del_B1z[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = rho0*dv1y_dt - B0x*dB1y_dx
    return G


# @tf.function
def pde_v1z(xt, Y1, del_Y1):
    """Differential equation for v1z.

    Evaluate the differential equation for v1z (z-component of velocity
    perturbation).

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y1 : list of n_var tf.Tensor, each shape (n, 1)
        Perturbations of dependent variables at each training point.
    del_Y1 : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradients of Y1 wrt (x, t) at each training point.

    Returns
    -------
    G : tf.Tensor, shape(n, 1)
        Value of differential equation at each training point.
    """
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    # (rho1, v1x, v1y, v1z, B1y, B1z) = Y1
    (del_rho1, del_v1x, del_v1y, del_v1z, del_B1y, del_B1z) = del_Y1
    # drho1_dx = tf.reshape(del_rho1[:, 0], (n, 1))
    # drho1_dt = tf.reshape(del_rho1[:, 1], (n, 1))
    # dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
    # dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
    # dv1y_dx = tf.reshape(del_v1y[:, 0], (n, 1))
    # dv1y_dt = tf.reshape(del_v1y[:, 1], (n, 1))
    # dv1z_dx = tf.reshape(del_v1z[:, 0], (n, 1))
    dv1z_dt = tf.reshape(del_v1z[:, 1], (n, 1))
    # dB1y_dx = tf.reshape(del_B1y[:, 0], (n, 1))
    # dB1y_dt = tf.reshape(del_B1y[:, 1], (n, 1))
    dB1z_dx = tf.reshape(del_B1z[:, 0], (n, 1))
    # dB1z_dt = tf.reshape(del_B1z[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = rho0*dv1z_dt - B0x*dB1z_dx
    return G


# @tf.function
def pde_B1y(xt, Y1, del_Y1):
    """Differential equation for B1y.

    Evaluate the differential equation for B1y (y-component of magnetic
    field perturbation).

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y1 : list of n_var tf.Tensor, each shape (n, 1)
        Perturbations of dependent variables at each training point.
    del_Y1 : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradients of Y1 wrt (x, t) at each training point.

    Returns
    -------
    G : tf.Tensor, shape(n, 1)
        Value of differential equation at each training point.
    """
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    # (rho1, v1x, v1y, v1z, B1y, B1z) = Y1
    (del_rho1, del_v1x, del_v1y, del_v1z, del_B1y, del_B1z) = del_Y1
#    (del_v1y, del_B1y) = del_Y1
    # drho1_dx = tf.reshape(del_rho1[:, 0], (n, 1))
    # drho1_dt = tf.reshape(del_rho1[:, 1], (n, 1))
    # dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
    # dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
    dv1y_dx = tf.reshape(del_v1y[:, 0], (n, 1))
    # dv1y_dt = tf.reshape(del_v1y[:, 1], (n, 1))
    # dv1z_dx = tf.reshape(del_v1z[:, 0], (n, 1))
    # dv1z_dt = tf.reshape(del_v1z[:, 1], (n, 1))
    # dB1y_dx = tf.reshape(del_B1y[:, 0], (n, 1))
    dB1y_dt = tf.reshape(del_B1y[:, 1], (n, 1))
    # dB1z_dx = tf.reshape(del_B1z[:, 0], (n, 1))
    # dB1z_dt = tf.reshape(del_B1z[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = dB1y_dt - B0x*dv1y_dx
    return G


# @tf.function
def pde_B1z(xt, Y1, del_Y1):
    """Differential equation for B1z.

    Evaluate the differential equation for B1z (z-component of magnetic
    field perturbation).

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y1 : list of n_var tf.Tensor, each shape (n, 1)
        Perturbations of dependent variables at each training point.
    del_Y1 : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradients of Y1 wrt (x, t) at each training point.

    Returns
    -------
    G : tf.Tensor, shape(n, 1)
        Value of differential equation at each training point.
    """
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    # (rho1, v1x, v1y, v1z, B1y, B1z) = Y1
    (del_rho1, del_v1x, del_v1y, del_v1z, del_B1y, del_B1z) = del_Y1
    # drho1_dx = tf.reshape(del_rho1[:, 0], (n, 1))
    # drho1_dt = tf.reshape(del_rho1[:, 1], (n, 1))
    # dv1x_dx = tf.reshape(del_v1x[:, 0], (n, 1))
    # dv1x_dt = tf.reshape(del_v1x[:, 1], (n, 1))
    # dv1y_dx = tf.reshape(del_v1y[:, 0], (n, 1))
    # dv1y_dt = tf.reshape(del_v1y[:, 1], (n, 1))
    dv1z_dx = tf.reshape(del_v1z[:, 0], (n, 1))
    # dv1z_dt = tf.reshape(del_v1z[:, 1], (n, 1))
    # dB1y_dx = tf.reshape(del_B1y[:, 0], (n, 1))
    # dB1y_dt = tf.reshape(del_B1y[:, 1], (n, 1))
    # dB1z_dx = tf.reshape(del_B1z[:, 0], (n, 1))
    dB1z_dt = tf.reshape(del_B1z[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = dB1z_dt - B0x*dv1z_dx
    return G


# Make a list of all of the differential equations.
differential_equations = [
    pde_rho1,
    pde_v1x,
    pde_v1y,
    pde_v1z,
    pde_B1y,
    pde_B1z,
]
