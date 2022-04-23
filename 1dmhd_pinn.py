#!/usr/bin/env python

"""Use a set of neural networks to solve the 1-D MHD equations for Alfven waves.

This program will use a set of neural networks to solve the coupled partial
differential equations of one-dimensional ideal MHD. This version is
customized to examine Alfven waves.

This code assumes Bx is constant, and so the gradient of Bx is 0.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard Python modules.
import argparse
import datetime
from importlib import import_module
from itertools import repeat
import os
import platform
import sys

# Import 3rd-party modules.
import numpy as np

# Import TensorFlow.
import tensorflow as tf


# Global object to hold the problem definition.
p = None


# Program constants

# Program description.
description = "Solve the 1-D MHD equations for Alfven waves with a set of neural networks."

# Default identifier string for a runs.
default_problem = "alfven_wave"

# Name of system information file.
system_information_file = "system_information.txt"

# Name of hyperparameter record file.
hyperparameter_file = "hyperparameters.py"

# # Name of optimizer used for training.
# optimizer_name = "Adam"

# Initial parameter ranges
w0_range = [-0.1, 0.1]
u0_range = [-0.1, 0.1]
v0_range = [-0.1, 0.1]

# # Number of dimensions
# m = 2

# Default random number generator seed.
default_random_seed = 0

# Default number of hidden nosdes per layer.
default_H = 10

# Default maximum number of training epochs.
default_max_epochs = 1000

# Default learning rate.
default_learning_rate = 0.01


# Absolute tolerance for consecutive loss function values to indicate
# convergence.
default_tolerance = 1e-6

# Default number of training points in each dimension.
default_nx_train = 11
default_nt_train = 11


def create_command_line_parser():
    """Create the command-line argument parser.

    Create the command-line parser.

    Parameters
    ----------
    None

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser for command-line arguments.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-d", "--debug", action="store_true", default=False,
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=default_learning_rate,
        help="Learning rate for training (default: %(default)s)"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=default_max_epochs,
        help="Maximum number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--n_hid", type=int, default=default_H,
        help="Number of hidden nodes per layer (default: %(default)s)"
    )
    parser.add_argument(
        "--nt_train", type=int, default=default_nt_train,
        help="Number of equally-spaced training points in t dimension (default: %(default)s)"
    )
    parser.add_argument(
        "--nx_train", type=int, default=default_nx_train,
        help="Number of equally-spaced training points in x dimension (default: %(default)s)"
    )
    parser.add_argument(
        "--problem", type=str, default=default_problem,
        help="Name of problem to solve (default: %(default)s)"
    )
    parser.add_argument(
        "--seed", type=int, default=default_random_seed,
        help="Seed for random number generator (default: %(default)s)"
    )
    parser.add_argument(
        "--tolerance", type=float, default=default_tolerance,
        help="Absolute loss function convergence tolerance (default: %(default)s)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Print verbose output (default: %(default)s)."
    )
    return parser


def create_output_directory(path=None):
    """Create an output directory for the results.
    
    Create the specified directory. Abort if it already exists.

    Parameters
    ----------
    path : str
        Path to directory to create.
    
    Returns
    -------
    None

    Raises
    ------
    Exception
        If the directory exists.
    """
    try:
        os.mkdir(path)
    except:
        pass


def save_system_information(output_dir):
    """Save a summary of system characteristics.
    
    Save a summary of the host system in the specified directory.

    Parameters
    ----------
    output_dir : str
        Path to directory to contain the report.
    
    Returns
    -------
    None
    """
    path = os.path.join(output_dir, system_information_file)
    with open(path, "w") as f:
        f.write("System report:\n")
        f.write("Start time: %s\n" % datetime.datetime.now())
        f.write("Host name: %s\n" % platform.node())
        f.write("Platform: %s\n" % platform.platform())
        f.write("uname: " + " ".join(platform.uname()) + "\n")
        f.write("Python version: %s\n" % sys.version)
        f.write("Python build: %s\n" % " ".join(platform.python_build()))
        f.write("Python compiler: %s\n" % platform.python_compiler())
        f.write("Python implementation: %s\n" % platform.python_implementation())
        f.write("Python file: %s\n" % __file__)


def save_hyperparameters(output_dir, args):
    """Save the neural network hyperparameters.
    
    Print a record of the hyperparameters of the neural networks in the
    specified directory.

    Parameters
    ----------
    output_dir : str
        Path to directory to contain the report.
    args : dict
        Dictionary of command-line arguments.

    Returns
    -------
    None
    """
    path = os.path.join(output_dir, hyperparameter_file)
    with open(path, "w") as f:
        f.write("learning_rate = %s\n" % repr(args.learning_rate))
        f.write("max_epochs = %s\n" % repr(args.max_epochs))
        f.write("H = %s\n" % repr(args.n_hid))
        f.write("w0_range = %s\n" % repr(w0_range))
        f.write("u0_range = %s\n" % repr(u0_range))
        f.write("v0_range = %s\n" % repr(v0_range))
        f.write("nx_train = %s\n" % repr(args.nx_train))
        f.write("nt_train = %s\n" % repr(args.nt_train))
        f.write("random_seed = %s\n" % repr(args.seed))
        f.write("tol = %s\n" % repr(args.tolerance))


def prod(n):
    """Compute the product of the elements of a list of numbers.
    
    Compute the product of the elements of a list of numbers.

    Parameters
    ----------
    n : list of int
        List of integers
    
    Returns
    -------
    p : int
        Product of numbers in list.
    """
    p = 1
    for nn in n:
        p *= nn
    return p


def create_training_grid2(shape):
    """Create a grid of training data.

    Create a grid of normalized training data described by the input shape.

    The input n is a list containing the numbers of evenly-
    spaced data points to use in each dimension.  For example, for an
    (x, y, z) grid, with n = [3, 4, 5], we will get a grid with 3 points
    along the x-axis, 4 points along the y-axis, and 5 points along the
    z-axis, for a total of 3*4*5 = 60 points. The points along each dimension
    are evenly spaced in the range [0, 1]. When there is m = 1 dimension, a
    list is returned, containing the evenly-spaced points in the single
    dimension.  When m > 1, a list of lists is returned, where each sub-list
    is the coordinates of a single point, in the order [x1, x2, ..., xm],
    where the coordinate order corresponds to the order of coordinate counts
    in the input list n.

    Parameters
    ----------
    shape : list of int
        List of dimension sizes for training data.
    
    Returns
    -------
    X : list
        List of training points
    """

    # Determine the number of dimensions in the result.
    m = len(shape)

    # Handle 1-D and (n>1)-D cases differently.
    if m == 1:
        n = shape[0]
        X = [i/(n - 1) for i in range(n)]
    else:
        # Compute the evenly-spaced points along each dimension.
        x = [[i/(n - 1) for i in range(n)] for n in shape]

        # Assemble all possible point combinations.
        X = []
        p1 = None
        p2 = 1
        for j in range(m - 1):
            p1 = prod(shape[j + 1:])
            XX = [xx for item in x[j] for xx in repeat(item, p1)]*p2
            X.append(XX)
            p2 *= shape[j]
        X.append(x[-1]*p2)
        X = list(zip(*X))

    # Return the list of training points.
    return X


def create_training_data(nx_train, nt_train):
    """Create the training data.
    
    Create and return a set of training data of points evenly space in x and
    t. Flatten the data to a list of pairs of points.
    
    Parameters
    ----------
    nx_train, nt_train : int
        Number of points to use in x- and t-dimensions.
    
    Returns
    -------
    xt_train : np.ndarray, shape (nx_train*nt_train, 2)
        Array of [x, t] points.
    """
    xt_train = np.array(create_training_grid2([nx_train, nt_train]), dtype="float32")

    # Now split the training data into two groups - inside the BC, and on the BC.
    mask = np.ones(len(xt_train), dtype=bool)
    mask[:nx_train] = False
    mask[-nx_train:] = False
    mask[::nx_train] = False
    # Keep t=1 inside.
    # mask[nt_train - 1::nx_train] = False
    xt_inside = xt_train[mask]
    mask = np.logical_not(mask)
    xt_bc = xt_train[mask]
    return xt_train, xt_inside, xt_bc


def build_model(H):
    """Build a single-layer neural network model.
    
    Build a fully-connected, single-layer neural network with single output.

    Parameters
    ----------
    H : int
        Number of nodes to use in the hidden layer.
    
    Returns
    -------
    net : tf.keras.Sequential
        The neural network.s
    """
    hidden_layer = tf.keras.layers.Dense(
        units=H, use_bias=True,
        activation=tf.keras.activations.sigmoid,
        kernel_initializer=tf.keras.initializers.RandomUniform(*w0_range),
        bias_initializer=tf.keras.initializers.RandomUniform(*u0_range)
    )
    output_layer = tf.keras.layers.Dense(
        units=1,
        activation=tf.keras.activations.linear,
        kernel_initializer=tf.keras.initializers.RandomUniform(*v0_range),
        use_bias=False,
    )
    model = tf.keras.Sequential([hidden_layer, output_layer])
    return model


# # Define the differential equations using TensorFlow operations.

# These equations are taken from:

# https://www.csun.edu/~jb715473/examples/mhd1d.htm

# The original equations are:

# For 1-D flow, div(B) = 0, so Bx is constant.

# The general form of the each differential equation is (d are
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

# xt is the list of tf.Variable [x, t].
# Y is the list of tf.Variable [rho, vx, vy, vz, By, Bz, E]
# del_Y is the list of gradients [del_rho, del_vx, del_vy, del_vz, del_By, del_Bz, del_P]

# @tf.function
def pde_rho(xt, Y, del_Y):
    """Differential equation for rho."""
    x = xt[:, 0]
    t = xt[:, 1]
    (rho, vx, vy, vz, By, Bz, P) = Y
    (del_rho, del_vx, del_vy, del_vz, del_By, del_Bz, del_P) = del_Y
    drho_dx = del_rho[:, 0]
    drho_dt = del_rho[:, 1]
    dvx_dx  =  del_vx[:, 0]
    G = drho_dt + rho*dvx_dx + drho_dx*vx
    return G

# @tf.function
def pde_vx(xt, Y, del_Y):
    """Differential equation for vx."""
    x = xt[:, 0]
    t = xt[:, 1]
    (rho, vx, vy, vz, By, Bz, P) = Y
    Bx = p.Bx_0
    (del_rho, del_vx, del_vy, del_vz, del_By, del_Bz, del_P) = del_Y
    drho_dx = del_rho[:, 0]
    drho_dt = del_rho[:, 1]
    dvx_dx  =  del_vx[:, 0]
    dvx_dt  =  del_vx[:, 1]
    dBx_dx  =  0
    dBy_dx  =  del_By[:, 0]
    dBz_dx  =  del_Bz[:, 0]
    dP_dx   =   del_P[:, 0]
    dPtot_dx = dP_dx + Bx*dBx_dx + By*dBy_dx + Bz*dBz_dx
    G = (
        rho*dvx_dt + drho_dt*vx
        + rho*2*vx*dvx_dx + drho_dx*vx**2 + dPtot_dx - 2*Bx*dBx_dx
    )
    return G

# @tf.function
def pde_vy(xt, Y, del_Y):
    """Differential equation for vy."""
    x = xt[:, 0]
    t = xt[:, 1]
    (rho, vx, vy, vz, By, Bz, P) = Y
    Bx = p.Bx_0
    (del_rho, del_vx, del_vy, del_vz, del_By, del_Bz, del_P) = del_Y
    drho_dx = del_rho[:, 0]
    drho_dt = del_rho[:, 1]
    dvx_dx  =  del_vx[:, 0]
    dvy_dx  =  del_vy[:, 0]
    dvy_dt  =  del_vy[:, 1]
    dBx_dx  =  0
    dBy_dx  =  del_By[:, 0]
    G = (
        rho*dvy_dt + drho_dt*vy
        + rho*vx*dvy_dx + rho*dvx_dx*vy + drho_dx*vx*vy
        - Bx*dBy_dx - dBx_dx*By
    )
    return G

# @tf.function
def pde_vz(xt, Y, del_Y):
    """Differential equation for vz."""
    x = xt[:, 0]
    t = xt[:, 1]
    (rho, vx, vy, vz, By, Bz, P) = Y
    Bx = p.Bx_0
    (del_rho, del_vx, del_vy, del_vz, del_By, del_Bz, del_P) = del_Y
    drho_dx = del_rho[:, 0]
    drho_dt = del_rho[:, 1]
    dvx_dx  =  del_vx[:, 0]
    dvz_dx  =  del_vz[:, 0]
    dvz_dt  =  del_vz[:, 1]
    dBx_dx  =  0
    dBz_dx  =  del_Bz[:, 0]
    G = (
        rho*dvz_dt + drho_dt*vz
        + rho*vx*dvz_dx + rho*dvx_dx*vz + drho_dx*vx*vz
        - Bx*dBz_dx - dBx_dx*Bz
    )
    return G

# @tf.function
def pde_By(xt, Y, del_Y):
    """Differential equation for By."""
    x = xt[:, 0]
    t = xt[:, 1]
    (rho, vx, vy, vz, By, Bz, P) = Y
    Bx = p.Bx_0
    (del_rho, del_vx, del_vy, del_vz, del_By, del_Bz, del_P) = del_Y
    dvx_dx = del_vx[:, 0]
    dvy_dx = del_vy[:, 0]
    dBx_dx = 0
    dBy_dx = del_By[:, 0]
    dBy_dt = del_By[:, 1]
    G = dBy_dt + By*dvx_dx + dBy_dx*vx - Bx*dvy_dx - dBx_dx*vy
    return G

# @tf.function
def pde_Bz(xt, Y, del_Y):
    """Differential equation for Bz."""
    x = xt[:, 0]
    t = xt[:, 1]
    (rho, vx, vy, vz, By, Bz, P) = Y
    Bx = p.Bx_0
    (del_rho, del_vx, del_vy, del_vz, del_By, del_Bz, del_P) = del_Y
    dvx_dx  = del_vx[:, 0]
    dvz_dx  = del_vz[:, 0]
    dBx_dx  = 0
    dBz_dx  = del_Bz[:, 0]
    dBz_dt  = del_Bz[:, 1]
    G = dBz_dt + Bz*dvx_dx + dBz_dx*vx - Bx*dvz_dx - dBx_dx*vz
    return G

# @tf.function
def pde_P(xt, Y, del_Y):
    """Differential equation for P (actually E)."""
    x = xt[:, 0]
    t = xt[:, 1]
    (rho, vx, vy, vz, By, Bz, P) = Y
    Bx = p.Bx_0
    (del_rho, del_vx, del_vy, del_vz, del_By, del_Bz, del_P) = del_Y
    drho_dx  = del_rho[:, 0]
    drho_dt  = del_rho[:, 1]
    dvx_dx   =  del_vx[:, 0]
    dvx_dt   =  del_vx[:, 1]
    dvy_dx   =  del_vy[:, 0]
    dvy_dt   =  del_vy[:, 1]
    dvz_dx   =  del_vz[:, 0]
    dvz_dt   =  del_vz[:, 1]
    dBx_dx   =  0
    dBx_dt   =  0
    dBy_dx   =  del_By[:, 0]
    dBy_dt   =  del_By[:, 1]
    dBz_dx   =  del_Bz[:, 0]
    dBz_dt   =  del_Bz[:, 1]
    dP_dx    =   del_P[:, 0]
    dP_dt    =   del_P[:, 1]
    Ptot = P + 0.5*(Bx**2 + By**2 + Bz**2)
    dPtot_dx = dP_dx + Bx*dBx_dx + By*dBy_dx + Bz*dBz_dx
    E = (
        P/(p.gamma - 1)
        + 0.5*rho*(vx**2 + vy**2 + vz**2)
        + 0.5*(Bx**2 + By**2 + Bz**2)
    )
    dE_dx = (
        dP_dx/(p.gamma - 1)
        + rho*(vx*dvx_dx + vy*dvy_dx + vz*dvz_dx)
        + drho_dx*0.5*(vx**2 + vy**2  + vz**2)
        + Bx*dBx_dx + By*dBy_dx + Bz*dBz_dx
    )
    dE_dt = (
        dP_dt/(p.gamma - 1)
        + rho*(vx*dvx_dt + vy*dvy_dt + vz*dvz_dt)
        + drho_dt*0.5*(vx**2 + vy**2  + vz**2)
        + Bx*dBx_dt + By*dBy_dt + Bz*dBz_dt
    )
    G = (
        dE_dt + (E + Ptot)*dvx_dx + (dE_dx + dPtot_dx)*vx
        - Bx*(Bx*dvx_dx + dBx_dx*vx + By*dvy_dx + dBy_dx*vy + Bz*dvz_dx + dBz_dx*vz)
        - dBx_dx*(Bx*vx + By*vy +Bz*vz)
    )
    return G


# # Define the trial functions.

# Placeholders for boundary coordinate pairs.
x0t = None
x1t = None

# # @tf.function
# def Ytrial_rho(xt, N):
#     """Trial solution for rho."""
#     x = xt[:, 0]
#     t = xt[:, 1]
#     f0 = p.f0_rho
#     f1 = p.f1_rho
#     g0 = p.g0_rho
#     A = (1 - x)*f0(xt) + x*f1(xt) + (1 - t)*(g0(xt) - ((1 - x)*g0(x0t) + x*g0(x1t)))
#     P = x*(1 - x)*t
#     Y = A + P*N[:, 0]
#     return Y

# # @tf.function
# def Ytrial_vx(xt, N):
#     """Trial solution for vx."""
#     x = xt[:, 0]
#     t = xt[:, 1]
#     f0 = p.f0_vx
#     f1 = p.f1_vx
#     g0 = p.g0_vx
#     A = (1 - x)*f0(xt) + x*f1(xt) + (1 - t)*(g0(xt) - ((1 - x)*g0(x0t) + x*g0(x1t)))
#     P = x*(1 - x)*t
#     Y = A + P*N[:, 0]
#     return Y

# # @tf.function
# def Ytrial_vy(xt, N):
#     """Trial solution for vy."""
#     x = xt[:, 0]
#     t = xt[:, 1]
#     f0 = p.f0_vy
#     f1 = p.f1_vy
#     g0 = p.g0_vy
#     A = (1 - x)*f0(xt) + x*f1(xt) + (1 - t)*(g0(xt) - ((1 - x)*g0(x0t) + x*g0(x1t)))
#     P = x*(1 - x)*t
#     Y = A + P*N[:, 0]
#     return Y

# # @tf.function
# def Ytrial_vz(xt, N):
#     """Trial solution for vz."""
#     x = xt[:, 0]
#     t = xt[:, 1]
#     f0 = p.f0_vz
#     f1 = p.f1_vz
#     g0 = p.g0_vz
#     A = (1 - x)*f0(xt) + x*f1(xt) + (1 - t)*(g0(xt) - ((1 - x)*g0(x0t) + x*g0(x1t)))
#     P = x*(1 - x)*t
#     Y = A + P*N[:, 0]
#     return Y

# # @tf.function
# def Ytrial_Bx(xt, N):
#     """Trial solution for Bx."""
#     x = xt[:, 0]
#     t = xt[:, 1]
#     f0 = p.f0_Bx
#     f1 = p.f1_Bx
#     g0 = p.g0_Bx
#     A = (1 - x)*f0(xt) + x*f1(xt) + (1 - t)*(g0(xt) - ((1 - x)*g0(x0t) + x*g0(x1t)))
#     P = x*(1 - x)*t
#     Y = A + P*N[:, 0]
#     return Y

# # @tf.function
# def Ytrial_By(xt, N):
#     """Trial solution for By."""
#     x = xt[:, 0]
#     t = xt[:, 1]
#     f0 = p.f0_By
#     f1 = p.f1_By
#     g0 = p.g0_By
#     A = (1 - x)*f0(xt) + x*f1(xt) + (1 - t)*(g0(xt) - ((1 - x)*g0(x0t) + x*g0(x1t)))
#     P = x*(1 - x)*t
#     Y = A + P*N[:, 0]
#     return Y

# # @tf.function
# def Ytrial_Bz(xt, N):
#     """Trial solution for Bz."""
#     x = xt[:, 0]
#     t = xt[:, 1]
#     f0 = p.f0_Bz
#     f1 = p.f1_Bz
#     g0 = p.g0_Bz
#     A = (1 - x)*f0(xt) + x*f1(xt) + (1 - t)*(g0(xt) - ((1 - x)*g0(x0t) + x*g0(x1t)))
#     P = x*(1 - x)*t
#     Y = A + P*N[:, 0]
#     return Y

# # @tf.function
# def Ytrial_P(xt, N):
#     """Trial solution for P."""
#     x = xt[:, 0]
#     t = xt[:, 1]
#     f0 = p.f0_P
#     f1 = p.f1_P
#     g0 = p.g0_P
#     A = (1 - x)*f0(xt) + x*f1(xt) + (1 - t)*(g0(xt) - ((1 - x)*g0(x0t) + x*g0(x1t)))
#     P = x*(1 - x)*t
#     Y = A + P*N[:, 0]
#     return Y


def main():
    """Begin main program."""

    # Set up the command-line parser.
    parser = create_command_line_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    debug = args.debug
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    H = args.n_hid
    nt_train = args.nt_train
    nx_train = args.nx_train
    random_seed = args.seed
    problem = args.problem
    tol = args.tolerance
    verbose = args.verbose

    # Import the problem to solve.
    global p
    p = import_module(problem)

    # Set up the output directory under the current directory.
    output_dir = os.path.join(".", problem)
    create_output_directory(output_dir)

    # Record system information and run parameters.
    save_system_information(output_dir)
    save_hyperparameters(output_dir, args)

    # Create and save the training data.
    xt_train, xt_train_in, xt_train_bc = create_training_data(nx_train, nt_train)
    n_train = len(xt_train)
    n_in = len(xt_train_in)
    n_bc = len(xt_train_bc)
    np.savetxt(os.path.join(output_dir, "xt_train.dat"), xt_train)

    # Compute the boundary condition values.
    rho_bc = np.zeros(n_bc)
    vx_bc = np.zeros(n_bc)
    vy_bc = np.zeros(n_bc)
    vz_bc = np.zeros(n_bc)
    By_bc = np.zeros(n_bc)
    Bz_bc = np.zeros(n_bc)
    P_bc = np.zeros(n_bc)
    for i in range(n_bc):
        xt = xt_train_bc[i]
        if xt[0] == 0:
            rho_bc[i] = p.rho_0
            vx_bc[i]  = p.vx_0
            vy_bc[i]  = p.vy_0
            vz_bc[i]  = p.vz_0
            By_bc[i]  = p.By_0
            Bz_bc[i]  = p.Bz_0
            P_bc[i]   = p.P_0
        elif xt[0] == 1:
            rho_bc[i] = p.rho_1
            vx_bc[i]  = p.vx_1
            vy_bc[i]  = p.vy_0
            vz_bc[i]  = p.vz_1
            By_bc[i]  = p.By_1
            Bz_bc[i]  = p.Bz_1
            P_bc[i]   = p.P_1
        elif xt[1] == 0:
            rho_bc[i] = p.rho_0
            vx_bc[i]  = p.vx_0
            vy_bc[i]  = p.vy_1
            vz_bc[i]  = p.vz_0
            By_bc[i]  = p.By_0
            Bz_bc[i]  = p.Bz_0
            P_bc[i]   = p.P_0
    rho_bc = tf.Variable(rho_bc, dtype="float32")
    vx_bc = tf.Variable(vx_bc, dtype="float32")
    vy_bc = tf.Variable(vy_bc, dtype="float32")
    vz_bc = tf.Variable(vz_bc, dtype="float32")
    By_bc = tf.Variable(By_bc, dtype="float32")
    Bz_bc = tf.Variable(Bz_bc, dtype="float32")
    P_bc = tf.Variable(P_bc, dtype="float32")

    # Build the models.
    model_rho = build_model(H)
    model_vx  = build_model(H)
    model_vy  = build_model(H)
    model_vz  = build_model(H)
    model_By  = build_model(H)
    model_Bz  = build_model(H)
    model_P   = build_model(H)

    # Create the optimizers.
    optimizer_rho = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer_vx  = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer_vy  = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer_vz  = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer_By  = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer_Bz  = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer_P   = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Train the models.

    # Create history variables.
    losses_rho = []
    losses_vx  = []
    losses_vy  = []
    losses_vz  = []
    losses_By  = []
    losses_Bz  = []
    losses_P   = []
    losses     = []

    phist_rho = []
    phist_vx  = []
    phist_vy  = []
    phist_vz  = []
    phist_By  = []
    phist_Bz  = []
    phist_P   = []

    # Set the random number seed for reproducibility.
    tf.random.set_seed(random_seed)

    # Rename the training data Variables for convenience.
    xt_train_var = tf.Variable(xt_train)
    xt = xt_train_var
    xt_train_in_var = tf.Variable(xt_train_in)
    xt_in = xt_train_in_var
    xt_train_bc_var = tf.Variable(xt_train_bc)
    xt_bc = xt_train_bc_var

    # Clear the convergence flag to start.
    converged = False

    t_start = datetime.datetime.now()
    print("Training started at", t_start, max_epochs)

    for epoch in range(max_epochs):

        # Run the forward pass.
        with tf.GradientTape(persistent=True) as tape1:
            with tf.GradientTape(persistent=True) as tape0:

                # Compute the network outputs at the interior training points.
                rho_in = model_rho(xt_in)
                vx_in  = model_vx( xt_in)
                vy_in  = model_vy( xt_in)
                vz_in  = model_vz( xt_in)
                By_in  = model_By( xt_in)
                Bz_in  = model_Bz( xt_in)
                P_in   = model_P(  xt_in)

                # Compute the network outputs on the boundaries.
                rho_bcm = model_rho(xt_bc)
                vx_bcm  = model_vx( xt_bc)
                vy_bcm  = model_vy( xt_bc)
                vz_bcm  = model_vz( xt_bc)
                By_bcm  = model_By( xt_bc)
                Bz_bcm  = model_Bz( xt_bc)
                P_bcm   = model_P(  xt_bc)

            # Compute the gradients of the trial solutions wrt inputs at the interior training points.
            del_rho_in = tape0.gradient(rho_in, xt_in)
            del_vx_in  = tape0.gradient(vx_in,  xt_in)
            del_vy_in  = tape0.gradient(vy_in,  xt_in)
            del_vz_in  = tape0.gradient(vz_in,  xt_in)
            del_By_in  = tape0.gradient(By_in,  xt_in)
            del_Bz_in  = tape0.gradient(Bz_in,  xt_in)
            del_P_in   = tape0.gradient(P_in,   xt_in)

            # Compute the estimates of the differential equations at the interior training points.
            Y_in = [rho_in, vx_in, vy_in, vz_in, By_in, Bz_in, P_in]
            del_Y_in = [del_rho_in, del_vx_in, del_vy_in, del_vz_in, del_By_in, del_Bz_in, del_P_in]
            G_rho_in = pde_rho(xt_in, Y_in, del_Y_in)
            G_vx_in  = pde_vx( xt_in, Y_in, del_Y_in)
            G_vy_in  =  pde_vy(xt_in, Y_in, del_Y_in)
            G_vz_in  =  pde_vz(xt_in, Y_in, del_Y_in)
            G_By_in  =  pde_By(xt_in, Y_in, del_Y_in)
            G_Bz_in  =  pde_Bz(xt_in, Y_in, del_Y_in)
            G_P_in   =   pde_P(xt_in, Y_in, del_Y_in)

            # Compute the errors in the computed BC.
            E_rho_bc = rho_bcm - rho_bc
            E_vx_bc = vx_bcm - vx_bc
            E_vy_bc = vy_bcm - vy_bc
            E_vz_bc = vz_bcm - vz_bc
            E_By_bc = By_bcm - By_bc
            E_Bz_bc = Bz_bcm - Bz_bc
            E_P_bc = P_bcm - P_bc

            # Compute the loss functions for the interior training points.
            L_rho_in = tf.math.sqrt(tf.reduce_sum(G_rho_in**2)/n_in)
            L_vx_in  = tf.math.sqrt(tf.reduce_sum(G_vx_in**2) /n_in)
            L_vy_in  = tf.math.sqrt(tf.reduce_sum(G_vy_in**2) /n_in)
            L_vz_in  = tf.math.sqrt(tf.reduce_sum(G_vz_in**2) /n_in)
            L_By_in  = tf.math.sqrt(tf.reduce_sum(G_By_in**2) /n_in)
            L_Bz_in  = tf.math.sqrt(tf.reduce_sum(G_Bz_in**2) /n_in)
            L_P_in   = tf.math.sqrt(tf.reduce_sum(G_P_in**2)  /n_in)
            L_in = L_rho_in + L_vx_in + L_vy_in + L_vz_in + L_By_in + L_Bz_in + L_P_in

            # Compute the loss functions for the boundary points.
            L_rho_bc = tf.math.sqrt(tf.reduce_sum(E_rho_bc**2)/n_bc)
            L_vx_bc  = tf.math.sqrt(tf.reduce_sum(E_vx_bc**2) /n_bc)
            L_vy_bc  = tf.math.sqrt(tf.reduce_sum(E_vy_bc**2) /n_bc)
            L_vz_bc  = tf.math.sqrt(tf.reduce_sum(E_vz_bc**2) /n_bc)
            L_By_bc  = tf.math.sqrt(tf.reduce_sum(E_By_bc**2) /n_bc)
            L_Bz_bc  = tf.math.sqrt(tf.reduce_sum(E_Bz_bc**2) /n_bc)
            L_P_bc   = tf.math.sqrt(tf.reduce_sum(E_P_bc**2)  /n_bc)
            L_bc = L_rho_bc + L_vx_bc + L_vy_bc + L_vz_bc + L_By_bc + L_Bz_bc + L_P_bc

            # Compute the total losses.
            L_rho = L_rho_in + L_rho_bc
            L_vx = L_vx_in + L_vx_bc
            L_vy = L_vy_in + L_vy_bc
            L_vz = L_vz_in + L_vz_bc
            L_By = L_By_in + L_By_bc
            L_Bz = L_Bz_in + L_Bz_bc
            L_P = L_P_in + L_P_bc
            L = L_in + L_bc

        # Save the current losses.
        losses_rho.append(L_rho.numpy())
        losses_vx.append( L_vx.numpy())
        losses_vy.append( L_vy.numpy())
        losses_vz.append( L_vz.numpy())
        losses_By.append( L_By.numpy())
        losses_Bz.append( L_Bz.numpy())
        losses_P.append(  L_P.numpy())
        losses.append(    L.numpy())

    #     # Check for convergence.
    #     if epoch > 1:
    #         loss_delta = losses[-1] - losses[-2]
    #         if abs(loss_delta) <= tol:
    #             converged = True
    #             break

        # Compute the gradient of the loss function wrt the network parameters.
        pgrad_rho = tape1.gradient(L, model_rho.trainable_variables)
        pgrad_vx  = tape1.gradient(L,  model_vx.trainable_variables)
        pgrad_vy  = tape1.gradient(L,  model_vy.trainable_variables)
        pgrad_vz  = tape1.gradient(L,  model_vz.trainable_variables)
        pgrad_By  = tape1.gradient(L,  model_By.trainable_variables)
        pgrad_Bz  = tape1.gradient(L,  model_Bz.trainable_variables)
        pgrad_P   = tape1.gradient(L,   model_P.trainable_variables)

    #     # Save the parameters used in this epoch.
    #     phist_rho.append(
    #         np.hstack(
    #             (model_rho.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    #              model_rho.trainable_variables[1].numpy(),       # u (H,) row vector
    #              model_rho.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    #         )
    #     )
    #     phist_vx.append(
    #         np.hstack(
    #             (model_vx.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    #              model_vx.trainable_variables[1].numpy(),       # u (H,) row vector
    #              model_vx.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    #         )
    #     )
    #     phist_vy.append(
    #         np.hstack(
    #             (model_vy.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    #              model_vy.trainable_variables[1].numpy(),       # u (H,) row vector
    #              model_vy.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    #         )
    #     )
    #     phist_vz.append(
    #         np.hstack(
    #             (model_vz.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    #              model_vz.trainable_variables[1].numpy(),       # u (H,) row vector
    #              model_vz.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    #         )
    #     )
    #     phist_Bx.append(
    #         np.hstack(
    #             (model_Bx.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    #              model_Bx.trainable_variables[1].numpy(),       # u (H,) row vector
    #              model_Bx.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    #         )
    #     )
    #     phist_By.append(
    #         np.hstack(
    #             (model_By.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    #              model_By.trainable_variables[1].numpy(),       # u (H,) row vector
    #              model_By.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    #         )
    #     )
    #     phist_Bz.append(
    #         np.hstack(
    #             (model_Bz.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    #              model_Bz.trainable_variables[1].numpy(),       # u (H,) row vector
    #              model_Bz.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    #         )
    #     )
    #     phist_P.append(
    #         np.hstack(
    #             (model_P.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    #              model_P.trainable_variables[1].numpy(),       # u (H,) row vector
    #              model_P.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    #         )
    #     )

        # Update the parameters for this epoch.
        optimizer_rho.apply_gradients(zip(pgrad_rho, model_rho.trainable_variables))
        optimizer_vx.apply_gradients(zip( pgrad_vx,   model_vx.trainable_variables))
        optimizer_vy.apply_gradients(zip( pgrad_vy,   model_vy.trainable_variables))
        optimizer_vz.apply_gradients(zip( pgrad_vz,   model_vz.trainable_variables))
        optimizer_By.apply_gradients(zip( pgrad_By,   model_By.trainable_variables))
        optimizer_Bz.apply_gradients(zip( pgrad_Bz,   model_Bz.trainable_variables))
        optimizer_P.apply_gradients( zip( pgrad_P,     model_P.trainable_variables))

        if epoch % 1 == 0:
            # print("Ending epoch %s" % (epoch))
            print("Ending epoch %s, loss function = %f" % (epoch, L.numpy()))

    # # Save the parameters used in the last epoch.
    # phist_rho.append(
    #     np.hstack(
    #         (model_rho.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    #          model_rho.trainable_variables[1].numpy(),       # u (H,) row vector
    #          model_rho.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    #     )
    # )
    # phist_vx.append(
    #     np.hstack(
    #         (model_vx.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    #          model_vx.trainable_variables[1].numpy(),       # u (H,) row vector
    #          model_vx.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    #     )
    # )
    # phist_vy.append(
    #     np.hstack(
    #         (model_vy.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    #          model_vy.trainable_variables[1].numpy(),       # u (H,) row vector
    #          model_vy.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    #     )
    # )
    # phist_vz.append(
    #     np.hstack(
    #         (model_vz.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    #          model_vz.trainable_variables[1].numpy(),       # u (H,) row vector
    #          model_vz.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    #     )
    # )
    # phist_Bx.append(
    #     np.hstack(
    #         (model_Bx.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    #          model_Bx.trainable_variables[1].numpy(),       # u (H,) row vector
    #          model_Bx.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    #     )
    # )
    # phist_By.append(
    #     np.hstack(
    #         (model_By.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    #          model_By.trainable_variables[1].numpy(),       # u (H,) row vector
    #          model_By.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    #     )
    # )
    # phist_Bz.append(
    #     np.hstack(
    #         (model_Bz.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    #          model_Bz.trainable_variables[1].numpy(),       # u (H,) row vector
    #          model_Bz.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    #     )
    # )
    # phist_P.append(
    #     np.hstack(
    #         (model_P.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    #          model_P.trainable_variables[1].numpy(),       # u (H,) row vector
    #          model_P.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    #     )
    # )

    # Count the last epoch.
    n_epochs = epoch + 1

    t_stop = datetime.datetime.now()
    print("Training stopped at", t_stop)
    t_elapsed = t_stop - t_start
    print("Total training time was %s seconds." % t_elapsed.total_seconds())
    print("Epochs: %d" % n_epochs)
    print("Final value of loss function: %f" % losses[-1])
    print("converged = %s" % converged)

    # Save the loss function histories.
    np.savetxt(os.path.join(output_dir, 'losses_rho.dat'), np.array(losses_rho))
    np.savetxt(os.path.join(output_dir, 'losses_vx.dat'),  np.array(losses_vx))
    np.savetxt(os.path.join(output_dir, 'losses_vy.dat'),  np.array(losses_vy))
    np.savetxt(os.path.join(output_dir, 'losses_vz.dat'),  np.array(losses_vz))
    np.savetxt(os.path.join(output_dir, 'losses_By.dat'),  np.array(losses_By))
    np.savetxt(os.path.join(output_dir, 'losses_Bz.dat'),  np.array(losses_Bz))
    np.savetxt(os.path.join(output_dir, 'losses_P.dat'),   np.array(losses_P))
    np.savetxt(os.path.join(output_dir, 'losses.dat'),     np.array(losses))

    # # Save the parameter histories.
    # np.savetxt(os.path.join(output_dir, 'phist_rho.dat'), np.array(phist_rho))
    # np.savetxt(os.path.join(output_dir, 'phist_vx.dat'),  np.array(phist_vx))
    # np.savetxt(os.path.join(output_dir, 'phist_vy.dat'),  np.array(phist_vy))
    # np.savetxt(os.path.join(output_dir, 'phist_vz.dat'),  np.array(phist_vz))
    # np.savetxt(os.path.join(output_dir, 'phist_Bx.dat'),  np.array(phist_Bx))
    # np.savetxt(os.path.join(output_dir, 'phist_By.dat'),  np.array(phist_By))
    # np.savetxt(os.path.join(output_dir, 'phist_Bz.dat'),  np.array(phist_Bz))
    # np.savetxt(os.path.join(output_dir, 'phist_P.dat'),   np.array(phist_P))

    # Compute and save the trained results at training points.
    with tf.GradientTape(persistent=True) as tape:

        # Compute the network outputs at the training points.
        rho_train = model_rho(xt)
        vx_train  = model_vx( xt)
        vy_train  = model_vy( xt)
        vz_train  = model_vz( xt)
        By_train  = model_By( xt)
        Bz_train  = model_Bz( xt)
        P_train   = model_P(  xt)

    # Compute gradients here if needed.
    # drho_dx_train = tape.gradient(rho_train, x)
    np.savetxt(os.path.join(output_dir, "rho_train.dat"), rho_train.numpy().reshape((n_train,)))
    np.savetxt(os.path.join(output_dir, "vx_train.dat"),   vx_train.numpy().reshape((n_train,)))
    np.savetxt(os.path.join(output_dir, "vy_train.dat"),   vy_train.numpy().reshape((n_train,)))
    np.savetxt(os.path.join(output_dir, "vz_train.dat"),   vz_train.numpy().reshape((n_train,)))
    np.savetxt(os.path.join(output_dir, "By_train.dat"),   By_train.numpy().reshape((n_train,)))
    np.savetxt(os.path.join(output_dir, "Bz_train.dat"),   Bz_train.numpy().reshape((n_train,)))
    np.savetxt(os.path.join(output_dir, "P_train.dat"),     P_train.numpy().reshape((n_train,)))


if __name__ == "__main__":
    """Begin main program."""
    main()
