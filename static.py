"""Problem definition file for 1dmhd.py.

This problem definition file describes a static situation: unit pressure and
density, all else is 0. The run should just converge to fixed values for each
quantity.

The problem is defined on the domain 0 <= (x, t) <= 1. Velocity and magnetic
field are 0.

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

# x and t limits for solution domain.
x_min = 0.0
x_max = 1.0
t_min = 0.0
t_max = 1.0

# Conditions at x = x_min, all t.
rho_0 = 1.0
P_0   = 1.0
vx_0  = 0.0
vy_0  = 0.0
vz_0  = 0.0
Bx_0  = 0.0
By_0  = 0.0
Bz_0  = 0.0

# Conditions at x = x_max, all t.
rho_1 = 1.0
P_1   = 1.0
vx_1  = 0.0
vy_1  = 0.0
vz_1  = 0.0
Bx_1  = 0.0
By_1  = 0.0
Bz_1  = 0.0


# Define the boundary conditions.
# 0 = at x_min or t_min
# 1 = at x_max or t_max

# Density
def f0_rho(xt):
    """Boundary condition for rho at (x_min, t)."""
    return tf.constant(rho_0, shape=(xt.shape[0],))

def f1_rho(xt):
    """Boundary condition for rho at (x_max, t)."""
    return tf.constant(rho_1, shape=(xt.shape[0],))

def g0_rho(xt):
    """Initial condition for rho at (x, t_min)."""
    return tf.constant(rho_0, shape=(xt.shape[0],))

# Thermal pressure

def f0_P(xt):
    """Boundary condition for P at (x_min, t)."""
    return tf.constant(P_0, shape=(xt.shape[0],))

def f1_P(xt):
    """Boundary condition for P at (x_max, t)."""
    return tf.constant(P_1, shape=(xt.shape[0],))

def g0_P(xt):
    """Initial condition for P at (x, t_min)."""
    return tf.constant(P_0, shape=(xt.shape[0],))

# x-velocity

def f0_vx(xt):
    """Boundary condition for vx at (x_min, t)."""
    return tf.constant(vx_0, shape=(xt.shape[0],))

def f1_vx(xt):
    """Boundary condition for vx at (x_max, t)."""
    return tf.constant(vx_1, shape=(xt.shape[0],))

def g0_vx(xt):
    """Initial condition for vx at (x, t_min)."""
    return tf.constant(vx_0, shape=(xt.shape[0],))

# y-velocity

def f0_vy(xt):
    """Boundary condition for vy at (x_min, t)."""
    return tf.constant(vy_0, shape=(xt.shape[0],))

def f1_vy(xt):
    """Boundary condition for vy at (x_max, t)."""
    return tf.constant(vy_1, shape=(xt.shape[0],))

def g0_vy(xt):
    """Initial condition for vy at (x, t_min)."""
    return tf.constant(vy_0, shape=(xt.shape[0],))

# z-velocity

def f0_vz(xt):
    """Boundary condition for vz at (x_min, t)."""
    return tf.constant(vz_0, shape=(xt.shape[0],))

def f1_vz(xt):
    """Boundary condition for vz at (x_max, t)."""
    return tf.constant(vz_1, shape=(xt.shape[0],))

def g0_vz(xt):
    """Initial condition for vz at (x, t_min)."""
    return tf.constant(vz_0, shape=(xt.shape[0],))

# x-component of B

def f0_Bx(xt):
    """Boundary condition for Bx at (x_min, t)."""
    return tf.constant(Bx_0, shape=(xt.shape[0],))

def f1_Bx(xt):
    """Boundary condition for Bx at (x_max, t)."""
    return tf.constant(Bx_1, shape=(xt.shape[0],))

def g0_Bx(xt):
    """Initial condition for Bx at (x, t_min)."""
    return tf.constant(Bx_0, shape=(xt.shape[0],))

# y-component of B

def f0_By(xt):
    """Boundary condition for By at (x_min, t)."""
    return tf.constant(By_0, shape=(xt.shape[0],))

def f1_By(xt):
    """Boundary condition for By at (x_max, t)."""
    return tf.constant(By_1, shape=(xt.shape[0],))

def g0_By(xt):
    """Initial condition for By at (x, t_min)."""
    return tf.constant(By_0, shape=(xt.shape[0],))

# z-component of B

def f0_Bz(xt):
    """Boundary condition for Bz at (x_min, t)."""
    return tf.constant(Bz_0, shape=(xt.shape[0],))

def f1_Bz(xt):
    """Boundary condition for Bz at (x_max, t)."""
    return tf.constant(Bz_1, shape=(xt.shape[0],))

def g0_Bz(xt):
    """Initial condition for Bz at (x, t_min)."""
    return tf.constant(Bz_0, shape=(xt.shape[0],))
