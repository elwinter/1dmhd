"""Problem definition file for alfven.py.

This problem definition file describes a static situation = unit pressure and
density, all else is 0. The run should just converge to fixed values of 0 or 1
for each quantity.

The functions in this module are defined using a combination of Numpy and
TensorFlow operations, so they can be used efficiently by 1dmhd.py.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""

import numpy as np
import tensorflow as tf


# Adiabatic index.
gamma = 1.4

# Inlet conditions
rho_0 = 1.0
vx_0  = 0.0
vy_0  = 0.0
vz_0  = 0.0
Bx_0  = 1.0e-3
By_0  = 0.0
Bz_0  = 0.0
P_0   = 1.0e-9

# Outlet conditions
rho_1 = 1.0
vx_1  = 0.0
vy_1  = 1.0e-3
vz_1  = 0.0
Bx_1  = 1.0e-3
By_1  = 0.0
Bz_1  = 0.0
P_1   = 1.0e-9

# Position of start and end of initial vy perturbation.
perturbation_x0 = 0.45
perturbation_x1 = 0.55


# Define the boundary conditions.
# 0 = at x = 0 or t = 0
# 1 = at x = 1 or t = 1

def compute_boundary_conditions(xt_bc):
    """Compute the boundary conditions."""
    n = len(xt_bc)
    rho_bc = np.zeros(n)
    P_bc = np.zeros(n)
    vx_bc = np.zeros(n)
    vy_bc = np.zeros(n)
    vz_bc = np.zeros(n)
    By_bc = np.zeros(n)
    Bz_bc = np.zeros(n)
    bc = np.zeros((n, 7))  # 7 variables
    for i in range(n):
        xt = xt_bc[i]
        # if xt[0] == 0:
        #     bc[i] = (rho_0, P_0, vx_0, vy_0, vz_0, By_0, Bz_0)
        # elif xt[0] == 1:
        #     bc[i] = (rho_1, P_1, vx_1, vy_0, vz_1, By_1, Bz_1)
        if xt[1] == 0:
            if xt[0] >= perturbation_x0 and xt[0] <= perturbation_x1:
                bc[i] = (rho_0, P_0, vx_0, vy_1, vz_0, By_0, Bz_0)
            else:
                bc[i] = (rho_0, P_0, vx_0, vy_0, vz_0, By_0, Bz_0)
        else:
            raise ValueError
    return bc

# Define the boundary conditions.

def f0_rho(xt):
    """Boundary condition for rho at (x, t) = (0, t)."""
    return tf.constant(rho_0, shape=(xt.shape[0],))

def f1_rho(xt):
    """Boundary condition for rho at (x, t) = (1, t)."""
    return tf.constant(rho_1, shape=(xt.shape[0],))

def g0_rho(xt):
    """Initial condition for rho at (x, t) = (x, 0)."""
    return tf.constant(rho_0, shape=(xt.shape[0],))

def f0_vx(xt):
    """Boundary condition for vx at (x, t) = (0, t)."""
    return tf.constant(vx_0, shape=(xt.shape[0],))

def f1_vx(xt):
    """Boundary condition for vx at (x, t) = (1, t)."""
    return tf.constant(vx_1, shape=(xt.shape[0],))

def g0_vx(xt):
    """Initial condition for vx at (x, t) = (x, 0)."""
    return tf.constant(vx_0, shape=(xt.shape[0],))

def f0_vy(xt):
    """Boundary condition for vy at (x, t) = (0, t)."""
    return tf.constant(vy_0, shape=(xt.shape[0],))

def f1_vy(xt):
    """Boundary condition for vy at (x, t) = (1, t)."""
    # USE vy_0 HERE!
    return tf.constant(vy_0, shape=(xt.shape[0],))

def g0_vy(xt):
    """Initial condition for vy at (x, t) = (x, 0)."""
    x = xt[:, 0]
    g0 = np.full(x.shape[0], vy_0)
    g0[np.where(x.numpy() >= perturbation_x0)] = vy_1
    g0[np.where(x.numpy() > perturbation_x1)] = vy_0
    g0 = tf.Variable(g0, dtype="float32")
    return g0

def f0_vz(xt):
    """Boundary condition for vz at (x, t) = (0, t)."""
    return tf.constant(vz_0, shape=(xt.shape[0],))

def f1_vz(xt):
    """Boundary condition for vz at (x, t) = (1, t)."""
    return tf.constant(vz_1, shape=(xt.shape[0],))

def g0_vz(xt):
    """Initial condition for vz at (x, t) = (x, 0)."""
    return tf.constant(vz_0, shape=(xt.shape[0],))

def f0_Bx(xt):
    """Boundary condition for Bx at (x, t) = (0, t)."""
    return tf.constant(Bx_0, shape=(xt.shape[0],))

def f1_Bx(xt):
    """Boundary condition for Bx at (x, t) = (1, t)."""
    return tf.constant(Bx_1, shape=(xt.shape[0],))

def g0_Bx(xt):
    """Initial condition for Bx at (x, t) = (x, 0)."""
    return tf.constant(Bx_0, shape=(xt.shape[0],))

def f0_By(xt):
    """Boundary condition for By at (x, t) = (0, t)."""
    return tf.constant(By_0, shape=(xt.shape[0],))

def f1_By(xt):
    """Boundary condition for By at (x, t) = (1, t)."""
    return tf.constant(By_1, shape=(xt.shape[0],))

def g0_By(xt):
    """Initial condition for By at (x, t) = (x, 0)."""
    return tf.constant(By_0, shape=(xt.shape[0],))

def f0_Bz(xt):
    """Boundary condition for Bz at (x, t) = (0, t)."""
    return tf.constant(Bz_0, shape=(xt.shape[0],))

def f1_Bz(xt):
    """Boundary condition for Bz at (x, t) = (1, t)."""
    return tf.constant(Bz_1, shape=(xt.shape[0],))

def g0_Bz(xt):
    """Initial condition for Bz at (x, t) = (x, 0)."""
    return tf.constant(Bz_0, shape=(xt.shape[0],))

def f0_P(xt):
    """Boundary condition for P at (x, t) = (0, t)."""
    return tf.constant(P_0, shape=(xt.shape[0],))

def f1_P(xt):
    """Boundary condition for P at (x, t) = (1, t)."""
    return tf.constant(P_1, shape=(xt.shape[0],))

def g0_P(xt):
    """Initial condition for P at (x, t) = (x, 0)."""
    return tf.constant(P_0, shape=(xt.shape[0],))
