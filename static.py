"""Problem definition file for static 1-D MHD problem.

This problem definition file describes a static situation: unit pressure and
density, all else is 0. The run should just converge to the initial values for
each quantity.

The problem is defined on the domain 0 <= (x, t) <= 1. Velocity and magnetic
field are 0. Bx is fixed at a constant value.

The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by 1dmhd.py.

NOTE: In all code, below, the following indices are assigned to physical
variables:

0: rho
1: P
2: vx
3: vy
4: vz
5: By
6: Bz

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


import numpy as np
import tensorflow as tf


# Names of dependent variables.
variable_names = ["rho", "P", "vx", "vy", "vz", "By", "Bz"]

# Number of independent variables.
n_dim = 2

# Number of dependent variables.
n_var = len(variable_names)

# Define the problem domain.
x0 = 0.0
x1 = 1.0
t0 = 0.0
t1 = 1.0

# Adiabatic index.
gamma = 1.5

# Strength of axial magnetic field.
Bx = 0.0

# Conditions at x = 0, all t.
bc0 = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Conditions at t = 0, all x.
bc1 = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def create_training_data(nx, nt):
    """Create the training data.

    Create and return a set of training data of points evenly spaced in x and
    t. Flatten the data to a list of pairs of points. Also return copies
    of the data containing only internal points, and only boundary points.

    Boundary conditions are computed for all points where at x = 0 or t = 0.

    Parameters
    ----------
    nx, nt : int
        Number of points in x- and t-dimensions.

    Returns
    -------
    xt : np.ndarray, shape (nx*nt, 2)
        Array of all [x, t] points.
    xt_in : np.ndarray, shape ((nx - 1)*(nt - 1)), 2)
        Array of all [x, t] points.
    xt_bc : np.ndarray, shape (nt + nx - 1, 2)
        Array of all [x, t] points.
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
    bc : np.ndarray of float, shape (n_bc, 7)
        Values of each physical variable on boundaries.
    """
    n = len(xt)
    bc = np.empty((n, n_var))
    for (i, (xx, tt)) in enumerate(xt):
        if np.isclose(xx, x0):
            bc[i, :] = bc0[:]
        elif np.isclose(xx, x1):
            bc[i, :] = bc1[:]
        elif np.isclose(tt, t0):
            bc[i, :] = bc0[:]
        else:
            raise ValueError
    return bc


# Define the differential equations using TensorFlow operations.

# These equations are taken from:

# https://www.csun.edu/~jb715473/examples/mhd1d.htm

# The original equations are:

# For 1-D flow, div(B) = 0, so Bx is constant.

# The general form of each differential equation is (d are
# partial derivatives)

#     dU/dt + dF/dx = 0

#     U = (rho, rho*vx, rho*vy, rho*vz, By, Bz, E)

#           / rho*vx                       \
#          |  rho*vx**2 + Ptot - Bx**2      |
#      F = |  rho*vx*vy - Bx*By             |
#          |  rho*vx*vz - Bx*Bz             |
#          |  By*vx - Bx*vy                 |
#          |  Bz*vx - Bx*vz                 |
#           \ (E + Ptot)*vx - Bx*(B dot v) /

#     Ptot = P + B**2/2

#     P = (gamma - 1)*(E - rho*v**2/2 - B**2/2)

# xt is the tf.Variable [x, t] of all of the training points.
# Y is the list of tf.Variable [rho, vx, vy, vz, By, Bz, P]
# del_Y is the list of gradients [del_rho, del_vx, del_vy, del_vz, del_By, del_Bz, del_P]

# @tf.function
def pde_rho(xt, Y, del_Y):
    """Differential equation for rho.

    Evaluate the differential equation for rho (density).

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each training point.
    del_Y : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradient wrt (x, t) at each training point, for each
        dependent variable.
    """
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    (rho, P, vx, vy, vz, By, Bz) = Y
    (del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz) = del_Y
    drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
    # dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    dvx_dx = tf.reshape(del_vx[:, 0], (n, 1))
    # dvy_dx = tf.reshape(del_vy[:, 0], (n, 1))
    # dvz_dx = tf.reshape(del_vz[:, 0], (n, 1))
    # dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
    # dP_dt = tf.reshape(del_P[:, 1], (n, 1))
    # dvx_dt = tf.reshape(del_vx[:, 1], (n, 1))
    # dvy_dt = tf.reshape(del_vy[:, 1], (n, 1))
    # dvz_dt = tf.reshape(del_vz[:, 1], (n, 1))
    # dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))
    # G is a Tensor of shape (n, 1).
    G = drho_dt + rho*dvx_dx + drho_dx*vx
    return G

# @tf.function
def pde_P(xt, Y, del_Y):
    """Differential equation for P (actually E).

    Evaluate the differential equation for pressure (or energy density).

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each training point.
    del_Y : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradient wrt (x, t) at each training point, for each
        dependent variable.
    """
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    (rho, P, vx, vy, vz, By, Bz) = Y
    (del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz) = del_Y
    drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
    dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    dvx_dx = tf.reshape(del_vx[:, 0], (n, 1))
    dvy_dx = tf.reshape(del_vy[:, 0], (n, 1))
    dvz_dx = tf.reshape(del_vz[:, 0], (n, 1))
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
    dP_dt = tf.reshape(del_P[:, 1], (n, 1))
    dvx_dt = tf.reshape(del_vx[:, 1], (n, 1))
    dvy_dt = tf.reshape(del_vy[:, 1], (n, 1))
    dvz_dt = tf.reshape(del_vz[:, 1], (n, 1))
    dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
    dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))

    # Compute the total pressure and x-derivative.
    Ptot = P + 0.5*(Bx**2 + By**2 + Bz**2)
    # dBx_dx and dBx_dt are 0.
    dPtot_dx = dP_dx + By*dBy_dx + Bz*dBz_dx
    E = (
        P/(gamma - 1)
        + 0.5*rho*(vx**2 + vy**2 + vz**2)
        + 0.5*(Bx**2 + By**2 + Bz**2)
    )
    dE_dx = (
        dP_dx/(gamma - 1)
        + rho*(vx*dvx_dx + vy*dvy_dx + vz*dvz_dx)
        + drho_dx*0.5*(vx**2 + vy**2  + vz**2)
        + By*dBy_dx + Bz*dBz_dx
    )
    dE_dt = (
        dP_dt/(gamma - 1)
        + rho*(vx*dvx_dt + vy*dvy_dt + vz*dvz_dt)
        + drho_dt*0.5*(vx**2 + vy**2  + vz**2)
        + By*dBy_dt + Bz*dBz_dt
    )
    G = (
        dE_dt + (E + Ptot)*dvx_dx + (dE_dx + dPtot_dx)*vx
        - Bx*(Bx*dvx_dx + By*dvy_dx + dBy_dx*vy + Bz*dvz_dx + dBz_dx*vz)
    )
    return G

# @tf.function
def pde_vx(xt, Y, del_Y):
    """Differential equation for vx.

    Evaluate the differential equation for vx.

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each training point.
    del_Y : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradient wrt (x, t) at each training point, for each
        dependent variable.
    """
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    (rho, P, vx, vy, vz, By, Bz) = Y
    (del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz) = del_Y
    drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
    dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    dvx_dx = tf.reshape(del_vx[:, 0], (n, 1))
    # dvy_dx = tf.reshape(del_vy[:, 0], (n, 1))
    # dvz_dx = tf.reshape(del_vz[:, 0], (n, 1))
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
    # dP_dt = tf.reshape(del_P[:, 1], (n, 1))
    dvx_dt = tf.reshape(del_vx[:, 1], (n, 1))
    # dvy_dt = tf.reshape(del_vy[:, 1], (n, 1))
    # dvz_dt = tf.reshape(del_vz[:, 1], (n, 1))
    # dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))
    # dBx_dx is 0.
    dPtot_dx = dP_dx + By*dBy_dx + Bz*dBz_dx
    G = (
        rho*dvx_dt + drho_dt*vx
        + rho*2*vx*dvx_dx + drho_dx*vx**2 + dPtot_dx
    )
    return G

# @tf.function
def pde_vy(xt, Y, del_Y):
    """Differential equation for vy.

    Evaluate the differential equation for vy.

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each training point.
    del_Y : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradient wrt (x, t) at each training point, for each
        dependent variable.
    """
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    (rho, P, vx, vy, vz, By, Bz) = Y
    (del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz) = del_Y
    drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
    # dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    dvx_dx = tf.reshape(del_vx[:, 0], (n, 1))
    dvy_dx = tf.reshape(del_vy[:, 0], (n, 1))
    # dvz_dx = tf.reshape(del_vz[:, 0], (n, 1))
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
    # dP_dt = tf.reshape(del_P[:, 1], (n, 1))
    # dvx_dt = tf.reshape(del_vx[:, 1], (n, 1))
    dvy_dt = tf.reshape(del_vy[:, 1], (n, 1))
    # dvz_dt = tf.reshape(del_vz[:, 1], (n, 1))
    dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))
    # dBx_dx is 0.
    G = (
        rho*dvy_dt + drho_dt*vy
        + rho*vx*dvy_dx + rho*dvx_dx*vy + drho_dx*vx*vy
        - Bx*dBy_dx
    )
    return G

# @tf.function
def pde_vz(xt, Y, del_Y):
    """Differential equation for vz.

    Evaluate the differential equation for vz.

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each training point.
    del_Y : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradient wrt (x, t) at each training point, for each
        dependent variable.
    """
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    (rho, P, vx, vy, vz, By, Bz) = Y
    (del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz) = del_Y
    drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
    # dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    dvx_dx = tf.reshape(del_vx[:, 0], (n, 1))
    # dvy_dx = tf.reshape(del_vy[:, 0], (n, 1))
    dvz_dx = tf.reshape(del_vz[:, 0], (n, 1))
    # dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
    # dP_dt = tf.reshape(del_P[:, 1], (n, 1))
    dvx_dt = tf.reshape(del_vx[:, 1], (n, 1))
    # dvy_dt = tf.reshape(del_vy[:, 1], (n, 1))
    dvz_dt = tf.reshape(del_vz[:, 1], (n, 1))
    # dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))
    # dBx_dx is 0.
    G = (
        rho*dvz_dt + drho_dt*vz
        + rho*vx*dvz_dx + rho*dvx_dx*vz + drho_dx*vx*vz
        - Bx*dBz_dx
    )
    return G

# @tf.function
def pde_By(xt, Y, del_Y):
    """Differential equation for By.

    Evaluate the differential equation for By.

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each training point.
    del_Y : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradient wrt (x, t) at each training point, for each
        dependent variable.
    """
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    (rho, P, vx, vy, vz, By, Bz) = Y
    (del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz) = del_Y
    # drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
    # dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    dvx_dx = tf.reshape(del_vx[:, 0], (n, 1))
    dvy_dx = tf.reshape(del_vy[:, 0], (n, 1))
    # dvz_dx = tf.reshape(del_vz[:, 0], (n, 1))
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    # drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
    # dP_dt = tf.reshape(del_P[:, 1], (n, 1))
    # dvx_dt = tf.reshape(del_vx[:, 1], (n, 1))
    # dvy_dt = tf.reshape(del_vy[:, 1], (n, 1))
    # dvz_dt = tf.reshape(del_vz[:, 1], (n, 1))
    dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))
    # dBx_dx is 0.
    G = dBy_dt + By*dvx_dx + dBy_dx*vx - Bx*dvy_dx
    return G

# @tf.function
def pde_Bz(xt, Y, del_Y):
    """Differential equation for Bz.

    Evaluate the differential equation for Bz.

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each training point.
    del_Y : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradient wrt (x, t) at each training point, for each
        dependent variable.
    """
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    (rho, P, vx, vy, vz, By, Bz) = Y
    (del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz) = del_Y
    # drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
    # dP_dx = tf.reshape(del_P[:, 0], (n, 1))
    dvx_dx = tf.reshape(del_vx[:, 0], (n, 1))
    # dvy_dx = tf.reshape(del_vy[:, 0], (n, 1))
    # dvz_dx = tf.reshape(del_vz[:, 0], (n, 1))
    # dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    # drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
    # dP_dt = tf.reshape(del_P[:, 1], (n, 1))
    # dvx_dt = tf.reshape(del_vx[:, 1], (n, 1))
    # dvy_dt = tf.reshape(del_vy[:, 1], (n, 1))
    # dvz_dt = tf.reshape(del_vz[:, 1], (n, 1))
    # dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))
    dvx_dx  = del_vx[:, 0]
    dvz_dx  = del_vz[:, 0]
    dBz_dx  = del_Bz[:, 0]
    dBz_dt  = del_Bz[:, 1]
    G = dBz_dt + Bz*dvx_dx + dBz_dx*vx - Bx*dvz_dx
    return G


# Make a list of all of the differential equations.
differential_equations = [
    pde_rho,
    pde_P,
    pde_vx,
    pde_vy,
    pde_vz,
    pde_By,
    pde_vz
]
