"""Problem definition file for a simple 1-D MHD problem.

The plasma is initially at rest with a fixed axial magnetic field. Starting
at t = 0, a slow vy perturbation is applied at x = 0.

The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by 1dmhd.py.

NOTE: In all code, below, the following indices are assigned to physical
variables:

0: rho  # mass density
1: px   # x-momentum density = rho*vx
2: py   # y-momentum density = rho*vy
3: pz   # z-momentum density = rho*vz
4: By   # y-component of magnetic field
5: Bz   # z-xomponent of magnetic field
6: E    # Total energy density

These equations are based on the notes of Jorge Balbas (2020), from
California State University, Northridge.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


from matplotlib.axis import XTick
import numpy as np
import tensorflow as tf


# Names of dependent variables.
variable_names = ["rho", "px", "py", "pz", "By", "Bz", "E"]

# Number of dependent variables.
n_var = len(variable_names)

# Number of independent variables.
n_dim = 2

# Define the problem domain.
x0 = 0.0
x1 = 1.0
t0 = 0.0
t1 = 1.0

# Adiabatic index.
gamma = 1.4

# Ambient density at t = 0, for all x.
rho0 = 1.0

# Velocity components at t = 0, for all x.
vx0 = 0.0
vy0 = 0.0
vz0 = 0.0

# Momentum density components at t = 0, for all x.
px0 = rho0*vx0
py0 = rho0*vy0
pz0 = rho0*vz0

# Magnetic field components at t = 0, for all x.
Bx0 = 0.0
By0 = 0.0
Bz0 = 0.0

# Ambient pressure at t = 0, for all x.
P0 = 1.0


def py0t_perturbation(xt):
    """Compute the py perturbation at x=0.

    Compute the py perturbation at x=0.

    Assume the y-momentum increases linearly with time.

    Parameters
    ----------
    xt : list of float, shape (2,)
        Independent variable values (x, t).

    Returns
    -------
    py0t : float
        y-momentum density perturbations at (0, t).
    """
    (x, t) = xt
    a = 0.1
    py0t = py0 + a*t
    return py0t


#=====
# Definition needed here to compute E0.
def total_energy_density(P, rho, px, py, pz, Bx, By, Bz):
    """Compute the total energy density.

    Compute the total energy density.

    Parameters
    ----------
    P : np.ndarray of float, shape (n,)
        Values for the thermal pressure.
    rho : np.ndarray of float, shape (n,)
        Values for the mass density.
    px : np.ndarray of float, shape (n,)
        Values for the y-component of the momentum density.
    py : np.ndarray of float, shape (n,)
        Values for the z-component of the momentum density.
    pz : np.ndarray of float, shape (n,)
        Values for the x-component of the momentum density.
    Bx : np.ndarray of float, shape (n,)
        Values for the x-component of the magnetic field.
    By : np.ndarray of float, shape (n,)
        Values for the y-component of the magnetic field.
    Bz : np.ndarray of float, shape (n,)
        Values for the z-component of the magnetic field.

    Returns
    -------
    E : np.ndarray of float, shape (n,)
        Values for thermal pressure.
    """
    E = (
        P/(gamma - 1)
        + 0.5*(px**2 + py**2 + pz**2)/rho
        + 0.5*(Bx**2 + By**2 + Bz**2)
    )
    return E
#=====

# Total energy density at t = 0, for all x.
E0 = total_energy_density(P0, rho0, px0, py0, pz0, Bx0, By0, Bz0)

# Conditions at (x, t) = (0, t)
bc0t = [rho0, px0, py0, pz0, By0, Bz0, E0]

# Conditions at (x, t) = (x>0, 0)
bcx0 = [rho0, px0, py0, pz0, By0, Bz0, E0]

# Scale factors are needed to normalize physical quantities to a 0-1 range,
# which is required for a stable solution. Computations in this module are
# done in physical units, and are scaled to dimensionless units when passed
# back to the network.

# Scale factors for each dependent variable.
# s = np.array([1.0, 1.0e-9, 1.0e-3, 1.0e-3, 1.0e-3, 1.0, 1.0])


def total_pressure(P, Bx, By, Bz):
    """Compute the total (thermal + magnetic) pressure.

    Compute the total (thermal + magnetic) pressure.

    Parameters
    ----------
    P : np.ndarray of float, shape (n,)
        Values for the thermal pressure.
    Bx : np.ndarray of float, shape (n,)
        Values for the x-component of the magnetic field.
    By : np.ndarray of float, shape (n,)
        Values for the y-component of the magnetic field.
    Bz : np.ndarray of float, shape (n,)
        Values for the z-component of the magnetic field.

    Returns
    -------
    Ptot : np.ndarray of float, shape (n,)
        Values for total (thermal + magnetic) pressure.
    """
    Ptot = P + 0.5*(Bx**2 + By**2 + Bz**2)
    return Ptot


def thermal_pressure(E, rho, px, py, pz, Bx, By, Bz):
    """Compute the thermal pressure.

    Compute the thermal pressure.

    Parameters
    ----------
    E : np.ndarray of float, shape (n,)
        Values for the energy density.
    rho : np.ndarray of float, shape (n,)
        Values for the mass density.
    px : np.ndarray of float, shape (n,)
        Values for the y-component of the momentum density.
    py : np.ndarray of float, shape (n,)
        Values for the z-component of the momentum density.
    pz : np.ndarray of float, shape (n,)
        Values for the x-component of the momentum density.
    Bx : np.ndarray of float, shape (n,)
        Values for the x-component of the magnetic field.
    By : np.ndarray of float, shape (n,)
        Values for the y-component of the magnetic field.
    Bz : np.ndarray of float, shape (n,)
        Values for the z-component of the magnetic field.

    Returns
    -------
    P : np.ndarray of float, shape (n,)
        Values for thermal pressure.
    """
    P = (gamma - 1)*(
        E
        - 0.5*(px**2 + py**2 + pz**2)/rho
        - 0.5*(Bx**2 + By**2 + Bz**2)
    )
    return P


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
    for (i, (x, t)) in enumerate(xt):
        if np.isclose(x, x0):
            bc[i, :] = bc0t
            bc[i, 2] = py0t_perturbation([x, t])
        elif np.isclose(t, t0):
            bc[i, :] = bcx0
        else:
            raise ValueError
    return bc


# Define the differential equations using TensorFlow operations.

# The general form of each differential equation is (d are
# partial derivatives)

#     dU/dt + dF/dx = 0

#     U = (rho, px, py, pz, By, Bz, E)
#     F = flux of U

#           / px                               \
#          |  px**2/rho + Ptot - Bx**2          |
#          |  px*py/rho - Bx*By                 |
#      F = |  px*pz/rho - Bx*Bz                 |
#          |  (By*px - Bx*py)/rho               |
#          |  (Bz*px - Bx*pz)/rho               |
#           \ (E + Ptot)*px/rho - Bx*(B dot v) /

#     Ptot = P + B**2/2

#     P = (gamma - 1)*(
#             E - 0.5*(px**2 + py**2 + pz**2)/rho
#             - 0.5*(Bx**2 + By**2 + Bz**2)
#         )

# xt is the tf.Variable [x, t] of all of the training points.
# Y is the list of tf.Variable [rho, px, py, pz, By, Bz, E]
# del_Y is the list of gradients
# [del_rho, del_px, del_py, del_pz, del_By, del_Bz, del_E]

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
        Values of gradients of Y wrt (x, t) at each training point.

    Returns
    -------
    G : tf.Tensor, shape(n, 1)
        Value of differential equation at each training point.
    """
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    (rho, px, py, pz, By, Bz, E) = Y
    (del_rho, del_px, del_py, del_pz, del_By, del_Bz, del_E) = del_Y
    # drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
    drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
    dpx_dx = tf.reshape(del_px[:, 0], (n, 1))
    # dpx_dt = tf.reshape(del_px[:, 1], (n, 1))
    # dpy_dx = tf.reshape(del_py[:, 0], (n, 1))
    # dpy_dt = tf.reshape(del_py[:, 1], (n, 1))
    # dpz_dx = tf.reshape(del_pz[:, 0], (n, 1))
    # dpz_dt = tf.reshape(del_pz[:, 1], (n, 1))
    # dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    # dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))
    # dE_dx = tf.reshape(del_E[:, 0], (n, 1))
    # dE_dt = tf.reshape(del_E[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = drho_dt + dpx_dx
    return G


# @tf.function
def pde_px(xt, Y, del_Y):
    """Differential equation for px.

    Evaluate the differential equation for px (x-component of momentum
    density).

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each training point.
    del_Y : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradient wrt (x, t) at each training point, for each
        dependent variable.

    Returns
    -------
    G : tf.Tensor, shape(n, 1)
        Value of differential equation at each training point.
    """
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    (rho, px, py, pz, By, Bz, E) = Y
    (del_rho, del_px, del_py, del_pz, del_By, del_Bz, del_E) = del_Y
    drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
    # drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
    dpx_dx = tf.reshape(del_px[:, 0], (n, 1))
    dpx_dt = tf.reshape(del_px[:, 1], (n, 1))
    dpy_dx = tf.reshape(del_py[:, 0], (n, 1))
    # dpy_dt = tf.reshape(del_py[:, 1], (n, 1))
    dpz_dx = tf.reshape(del_pz[:, 0], (n, 1))
    # dpz_dt = tf.reshape(del_pz[:, 1], (n, 1))
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    # dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
    dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))
    dE_dx = tf.reshape(del_E[:, 0], (n, 1))
    # dE_dt = tf.reshape(del_E[:, 1], (n, 1))

    # Calculate the x-gradients for the thermal pressure and the total
    # pressure.
    dP_dx = (gamma - 1)*(
        dE_dx
        - ((px*dpx_dx + py*dpy_dx + pz*dpz_dx)/rho
           - (px**2 + py**2 + pz**2)/rho**2*drho_dx)
        - By*dBy_dx - Bz*dBz_dx
    )
    dPtot_dx = dP_dx + By*dBy_dx + Bz*dBz_dx
    # G is a Tensor of shape (n, 1).
    G = dpx_dt + 2*px/rho*dpx_dx - px**2/rho**2*drho_dx + dPtot_dx
    return G


# @tf.function
def pde_py(xt, Y, del_Y):
    """Differential equation for py.

    Evaluate the differential equation for py (y-component of momentum
    density).

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each training point.
    del_Y : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradient wrt (x, t) at each training point, for each
        dependent variable.

    Returns
    -------
    G : tf.Tensor, shape(n, 1)
        Value of differential equation at each training point.
    """
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    (rho, px, py, pz, By, Bz, E) = Y
    (del_rho, del_px, del_py, del_pz, del_By, del_Bz, del_E) = del_Y
    drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
    # drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
    dpx_dx = tf.reshape(del_px[:, 0], (n, 1))
    # dpx_dt = tf.reshape(del_px[:, 1], (n, 1))
    dpy_dx = tf.reshape(del_py[:, 0], (n, 1))
    dpy_dt = tf.reshape(del_py[:, 1], (n, 1))
    # dpz_dx = tf.reshape(del_pz[:, 0], (n, 1))
    # dpz_dt = tf.reshape(del_pz[:, 1], (n, 1))
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    # dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))
    # dE_dx = tf.reshape(del_E[:, 0], (n, 1))
    # dE_dt = tf.reshape(del_E[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = dpy_dt + (px*dpy_dx + dpx_dx*py)/rho - px*py/rho**2*drho_dx - Bx0*dBy_dx
    return G


# @tf.function
def pde_pz(xt, Y, del_Y):
    """Differential equation for pz.

    Evaluate the differential equation for pz (z-component of momentum
    density).

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each training point.
    del_Y : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradient wrt (x, t) at each training point, for each
        dependent variable.

    Returns
    -------
    G : tf.Tensor, shape(n, 1)
        Value of differential equation at each training point.
    """
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    (rho, px, py, pz, By, Bz, E) = Y
    (del_rho, del_px, del_py, del_pz, del_By, del_Bz, del_E) = del_Y
    drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
    # drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
    dpx_dx = tf.reshape(del_px[:, 0], (n, 1))
    # dpx_dt = tf.reshape(del_px[:, 1], (n, 1))
    # dpy_dx = tf.reshape(del_py[:, 0], (n, 1))
    # dpy_dt = tf.reshape(del_py[:, 1], (n, 1))
    dpz_dx = tf.reshape(del_pz[:, 0], (n, 1))
    dpz_dt = tf.reshape(del_pz[:, 1], (n, 1))
    # dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    # dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
    dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))
    # dE_dx = tf.reshape(del_E[:, 0], (n, 1))
    # dE_dt = tf.reshape(del_E[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = dpz_dt + (px*dpz_dx + dpx_dx*pz)/rho - px*pz/rho**2*drho_dx - Bx0*dBz_dx
    return G


# @tf.function
def pde_By(xt, Y, del_Y):
    """Differential equation for By.

    Evaluate the differential equation for By (y-component of magnetic
    field).

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each training point.
    del_Y : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradient wrt (x, t) at each training point, for each
        dependent variable.

    Returns
    -------
    G : tf.Tensor, shape(n, 1)
        Value of differential equation at each training point.
    """
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    (rho, px, py, pz, By, Bz, E) = Y
    (del_rho, del_px, del_py, del_pz, del_By, del_Bz, del_E) = del_Y
    drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
    # drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
    dpx_dx = tf.reshape(del_px[:, 0], (n, 1))
    # dpx_dt = tf.reshape(del_px[:, 1], (n, 1))
    dpy_dx = tf.reshape(del_py[:, 0], (n, 1))
    # dpy_dt = tf.reshape(del_py[:, 1], (n, 1))
    # dpz_dx = tf.reshape(del_pz[:, 0], (n, 1))
    # dpz_dt = tf.reshape(del_pz[:, 1], (n, 1))
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
    # dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))
    # dE_dx = tf.reshape(del_E[:, 0], (n, 1))
    # dE_dt = tf.reshape(del_E[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        dBy_dt
        + (By*dpx_dx + dBy_dx*px - Bx0*dpy_dx)/rho
        - (By*px - Bx0*py)/rho**2*drho_dx
    )
    return G


# @tf.function
def pde_Bz(xt, Y, del_Y):
    """Differential equation for Bz.

    Evaluate the differential equation for Bz (z-component of magnetic
    field).

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each training point.
    del_Y : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradient wrt (x, t) at each training point, for each
        dependent variable.

    Returns
    -------
    G : tf.Tensor, shape(n, 1)
        Value of differential equation at each training point.
    """
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    (rho, px, py, pz, By, Bz, E) = Y
    (del_rho, del_px, del_py, del_pz, del_By, del_Bz, del_E) = del_Y
    drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
    # drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
    dpx_dx = tf.reshape(del_px[:, 0], (n, 1))
    # dpx_dt = tf.reshape(del_px[:, 1], (n, 1))
    # dpy_dx = tf.reshape(del_py[:, 0], (n, 1))
    # dpy_dt = tf.reshape(del_py[:, 1], (n, 1))
    dpz_dx = tf.reshape(del_pz[:, 0], (n, 1))
    # dpz_dt = tf.reshape(del_pz[:, 1], (n, 1))
    # dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    # dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
    dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))
    # dE_dx = tf.reshape(del_E[:, 0], (n, 1))
    # dE_dt = tf.reshape(del_E[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    G = (
        dBz_dt
        + (Bz*dpx_dx+dBz_dx*px - Bx0*dpz_dx)/rho
        - (Bz*px - Bx0*pz)/rho**2*drho_dx
    )
    return G


# @tf.function
def pde_E(xt, Y, del_Y):
    """Differential equation for E.

    Evaluate the differential equation for E (total energy density).

    Parameters
    ----------
    xt : tf.Variable, shape (n, 2)
        Values of (x, t) at each training point.
    Y : list of n_var tf.Tensor, each shape (n, 1)
        Values of dependent variables at each training point.
    del_Y : list of n_var tf.Tensor, each shape (n, 2)
        Values of gradient wrt (x, t) at each training point, for each
        dependent variable.

    Returns
    -------
    G : tf.Tensor, shape(n, 1)
        Value of differential equation at each training point.
    """
    # Each of these Tensors is shape (n, 1).
    n = xt.shape[0]
    # x = tf.reshape(xt[:, 0], (n, 1))
    # t = tf.reshape(xt[:, 1], (n, 1))
    (rho, px, py, pz, By, Bz, E) = Y
    (del_rho, del_px, del_py, del_pz, del_By, del_Bz, del_E) = del_Y
    drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
    # drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
    dpx_dx = tf.reshape(del_px[:, 0], (n, 1))
    # dpx_dt = tf.reshape(del_px[:, 1], (n, 1))
    dpy_dx = tf.reshape(del_py[:, 0], (n, 1))
    # dpy_dt = tf.reshape(del_py[:, 1], (n, 1))
    dpz_dx = tf.reshape(del_pz[:, 0], (n, 1))
    # dpz_dt = tf.reshape(del_pz[:, 1], (n, 1))
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    # dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
    dBz_dx = tf.reshape(del_Bz[:, 0], (n, 1))
    # dBz_dt = tf.reshape(del_Bz[:, 1], (n, 1))
    dE_dx = tf.reshape(del_E[:, 0], (n, 1))
    dE_dt = tf.reshape(del_E[:, 1], (n, 1))

    # G is a Tensor of shape (n, 1).
    P = thermal_pressure(E, rho, px, py, pz, Bx0, By, Bz)
    Ptot = total_pressure(P, Bx0, By, Bz)
    dP_dx = (gamma - 1)*(
        dE_dx
        - ((px*dpx_dx + py*dpy_dx + pz*dpz_dx)/rho
           - (px**2 + py**2 + pz**2)/rho**2*drho_dx)
        - By*dBy_dx - Bz*dBz_dx
    )
    dPtot_dx = dP_dx + By*dBy_dx + Bz*dBz_dx
    G = (
        dE_dt
        + (E + Ptot)*(-px/rho**2*drho_dx + dpx_dx/rho)
        + (dE_dx + dPtot_dx)*px/rho
        - ((Bx0*px +By*py + Bz*pz)*(-Bx0/rho**2*drho_dx)
           + (Bx0*dpx_dx + By*dpy_dx + dBy_dx*py + Bz*dpz_dx
              + dBz_dx*pz)*Bx0/rho)
    )
    return G


# Make a list of all of the differential equations.
differential_equations = [
    pde_rho,
    pde_px,
    pde_py,
    pde_pz,
    pde_By,
    pde_Bz,
    pde_E
]


if __name__ == "__main__":

    # Set up test data.
    n = 10
    rho_test = 0.1 + np.linspace(0, 1, n)
    px_test = 0.2 + np.linspace(0, 1.1, n)
    py_test = 0.3 + np.linspace(0, 1.2, n)
    pz_test = 0.4 + np.linspace(0, 1.3, n)
    Bx_test = 0.5 + np.linspace(0, 2.1, n)
    By_test = 0.6 + np.linspace(0, 2.2, n)
    Bz_test = 0.7 + np.linspace(0, 2.3, n)
    E_test = 10 + np.linspace(0, 30, n)

    # Test the total pressure function.
    P_test = np.linspace(0, 4, n)
    Ptot_test = total_pressure(P_test, Bx_test, By_test, Bz_test)
    for i in range(n):
        Ptot_ref = P_test[i] + 0.5*(
            Bx_test[i]**2 + By_test[i]**2 + Bz_test[i]**2
        )
        assert np.isclose(Ptot_test[i], Ptot_ref)

    # Test the thermal pressure function.
    P_test = thermal_pressure(E_test, rho_test, px_test, py_test, pz_test,
                              Bx_test, By_test, Bz_test)
    for i in range(n):
        P_ref = (gamma - 1)*(
            E_test[i]
            - 0.5*(px_test[i]**2 + py_test[i]**2 + pz_test[i]**2)/rho_test[i]
            - 0.5*(Bx_test[i]**2 + By_test[i]**2 + Bz_test[i]**2)
        )
        assert np.isclose(P_test[i], P_ref)

    # Test the total energy density function.
    Etot_test = total_energy_density(P_test, rho_test, px_test, py_test, pz_test,
                                     Bx_test, By_test, Bz_test)
    for i in range(n):
        Etot_ref = E_test[i]
        assert np.isclose(Etot_test[i], Etot_ref)
