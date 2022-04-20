#!/usr/bin/env python

"""Use a set of neural networks to solve the 1-D MHD equations.

This program will use a set of neural networks to solve the coupled partial
differential equations of one-dimensional ideal MHD.

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
description = "Solve the 1-D MHD equations with a set of neural networks."

# Default identifier string for a run: "1dmhd" + time in seconds.
default_problem = "static"

# Name of system information file.
system_information_file = "system_information.txt"

# Name of hyperparameter record file.
hyperparameter_file = "hyperparameters.py"

# Name of optimizer used for training.
optimizer_name = "Adam"

# Initial parameter ranges
w0_range = [-0.1, 0.1]
u0_range = [-0.1, 0.1]
v0_range = [-0.1, 0.1]

# Number of dimensions
m = 2

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
    return xt_train


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


# Define the differential equations using TensorFlow operations.

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
# del_Y is the list of gradients [del_rho, del_vx, del_vy, del_vz, del_By, del_Bz, del_p]

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
    (del_rho, del_vx, del_vy, del_vz, del_By, del_Bz, del_P) = del_Y
    drho_dx = del_rho[:, 0]
    drho_dt = del_rho[:, 1]
    dvx_dx  =  del_vx[:, 0]
    dvx_dt  =  del_vx[:, 1]
    dBy_dx  =  del_By[:, 0]
    dBz_dx  =  del_Bz[:, 0]
    dP_dx   =   del_P[:, 0]
    dPtot_dx = dP_dx + By*dBy_dx + Bz*dBz_dx
    G = (
        rho*dvx_dt + drho_dt*vx
        + rho*2*vx*dvx_dx + drho_dx*vx**2 + dPtot_dx
    )
    return G

# @tf.function
def pde_vy(xt, Y, del_Y):
    """Differential equation for vy."""
    x = xt[:, 0]
    t = xt[:, 1]
    (rho, vx, vy, vz, By, Bz, P) = Y
    (del_rho, del_vx, del_vy, del_vz, del_By, del_Bz, del_P) = del_Y
    Bx = p.Bx_0
    drho_dx = del_rho[:, 0]
    drho_dt = del_rho[:, 1]
    dvx_dx  =  del_vx[:, 0]
    dvy_dx  =  del_vy[:, 0]
    dvy_dt  =  del_vy[:, 1]
    dBy_dx  =  del_By[:, 0]
    G = (
        rho*dvy_dt + drho_dt*vy
        + rho*vx*dvy_dx + rho*dvx_dx*vy + drho_dx*vx*vy
        - Bx*dBy_dx
    )
    return G

# @tf.function
def pde_vz(xt, Y, del_Y):
    """Differential equation for vz."""
    x = xt[:, 0]
    t = xt[:, 1]
    (rho, vx, vy, vz, By, Bz, P) = Y
    (del_rho, del_vx, del_vy, del_vz, del_By, del_Bz, del_P) = del_Y
    Bx = p.Bx_0
    drho_dx = del_rho[:, 0]
    drho_dt = del_rho[:, 1]
    dvx_dx  =  del_vx[:, 0]
    dvz_dx  =  del_vz[:, 0]
    dvz_dt  =  del_vz[:, 1]
    dBz_dx  =  del_Bz[:, 0]
    G = (
        rho*dvz_dt + drho_dt*vz
        + rho*vx*dvz_dx + rho*dvx_dx*vz + drho_dx*vx*vz
        - Bx*dBz_dx
    )
    return G

# @tf.function
def pde_By(xt, Y, del_Y):
    """Differential equation for By."""
    x = xt[:, 0]
    t = xt[:, 1]
    (rho, vx, vy, vz, By, Bz, P) = Y
    (del_rho, del_vx, del_vy, del_vz, del_By, del_Bz, del_P) = del_Y
    Bx = p.Bx_0
    dvx_dx = del_vx[:, 0]
    dvy_dx = del_vy[:, 0]
    dBy_dx = del_By[:, 0]
    dBy_dt = del_By[:, 1]
    G = dBy_dt + By*dvx_dx + dBy_dx*vx - Bx*dvy_dx
    return G

# @tf.function
def pde_Bz(xt, Y, del_Y):
    """Differential equation for Bz."""
    x = xt[:, 0]
    t = xt[:, 1]
    (rho, vx, vy, vz, By, Bz, P) = Y
    (del_rho, del_vx, del_vy, del_vz, del_By, del_Bz, del_P) = del_Y
    Bx = p.Bx_0
    dvx_dx  = del_vx[:, 0]
    dvz_dx  = del_vz[:, 0]
    dBz_dx  = del_Bz[:, 0]
    dBz_dt  = del_Bz[:, 1]
    G = dBz_dt + Bz*dvx_dx + dBz_dx*vx - Bx*dvz_dx
    return G

# @tf.function
def pde_P(xt, Y, del_Y):
    """Differential equation for P (actually E)."""
    x = xt[:, 0]
    t = xt[:, 1]
    (rho, vx, vy, vz, By, Bz, P) = Y
    (del_rho, del_vx, del_vy, del_vz, del_By, del_Bz, del_P) = del_Y
    Bx = p.Bx_0
    drho_dx  = del_rho[:, 0]
    drho_dt  = del_rho[:, 1]
    dvx_dx   =  del_vx[:, 0]
    dvx_dt   =  del_vx[:, 1]
    dvy_dx   =  del_vy[:, 0]
    dvy_dt   =  del_vy[:, 1]
    dvz_dx   =  del_vz[:, 0]
    dvz_dt   =  del_vz[:, 1]
    dBy_dx   =  del_By[:, 0]
    dBy_dt   =  del_By[:, 1]
    dBz_dx   =  del_Bz[:, 0]
    dBz_dt   =  del_Bz[:, 1]
    dP_dx    =   del_P[:, 0]
    dP_dt    =   del_P[:, 1]
    Ptot = P + 0.5*(Bx**2 + By**2 + Bz**2)
    dPtot_dx = dP_dx + By*dBy_dx + Bz*dBz_dx
    E = (
        P/(p.gamma - 1)
        + 0.5*rho*(vx**2 + vy**2 + vz**2)
        + 0.5*(Bx**2 + By**2 + Bz**2)
    )
    dE_dx = (
        dP_dx/(p.gamma - 1)
        + rho*(vx*dvx_dx + vy*dvy_dx + vz*dvz_dx)
        + drho_dx*0.5*(vx**2 + vy**2  + vz**2)
        + By*dBy_dx + Bz*dBz_dx
    )
    dE_dt = (
        dP_dt/(p.gamma - 1)
        + rho*(vx*dvx_dt + vy*dvy_dt + vz*dvz_dt)
        + drho_dt*0.5*(vx**2 + vy**2  + vz**2)
        + By*dBy_dt + Bz*dBz_dt
    )
    G = (
        dE_dt + (E + Ptot)*dvx_dx + (dE_dx + dPtot_dx)*vx
        - Bx*(Bx*dvx_dx + By*dvy_dx + dBy_dx*vy + Bz*dvz_dx + dBz_dx*vz)
    )
    return G


# Define the trial functions.

# Placeholders for boundary coordinate pairs.
x0t = None
x1t = None

# @tf.function
def Ytrial_rho(xt, N):
    """Trial solution for rho."""
    x = xt[:, 0]
    t = xt[:, 1]
    f0 = p.f0_rho
    f1 = p.f1_rho
    g0 = p.g0_rho
    A = (1 - x)*f0(xt) + x*f1(xt) + (1 - t)*(g0(xt) - ((1 - x)*g0(x0t) + x*g0(x1t)))
    P = x*(1 - x)*t
    Y = A + P*N[:, 0]
    return Y

# @tf.function
def Ytrial_vx(xt, N):
    """Trial solution for vx."""
    x = xt[:, 0]
    t = xt[:, 1]
    f0 = p.f0_vx
    f1 = p.f1_vx
    g0 = p.g0_vx
    A = (1 - x)*f0(xt) + x*f1(xt) + (1 - t)*(g0(xt) - ((1 - x)*g0(x0t) + x*g0(x1t)))
    P = x*(1 - x)*t
    Y = A + P*N[:, 0]
    return Y

# @tf.function
def Ytrial_vy(xt, N):
    """Trial solution for vy."""
    x = xt[:, 0]
    t = xt[:, 1]
    f0 = p.f0_vy
    f1 = p.f1_vy
    g0 = p.g0_vy
    A = (1 - x)*f0(xt) + x*f1(xt) + (1 - t)*(g0(xt) - ((1 - x)*g0(x0t) + x*g0(x1t)))
    P = x*(1 - x)*t
    Y = A + P*N[:, 0]
    return Y

# @tf.function
def Ytrial_vz(xt, N):
    """Trial solution for vz."""
    x = xt[:, 0]
    t = xt[:, 1]
    f0 = p.f0_vz
    f1 = p.f1_vz
    g0 = p.g0_vz
    A = (1 - x)*f0(xt) + x*f1(xt) + (1 - t)*(g0(xt) - ((1 - x)*g0(x0t) + x*g0(x1t)))
    P = x*(1 - x)*t
    Y = A + P*N[:, 0]
    return Y


# @tf.function
def Ytrial_By(xt, N):
    """Trial solution for By."""
    x = xt[:, 0]
    t = xt[:, 1]
    f0 = p.f0_By
    f1 = p.f1_By
    g0 = p.g0_By
    A = (1 - x)*f0(xt) + x*f1(xt) + (1 - t)*(g0(xt) - ((1 - x)*g0(x0t) + x*g0(x1t)))
    P = x*(1 - x)*t
    Y = A + P*N[:, 0]
    return Y

# @tf.function
def Ytrial_Bz(xt, N):
    """Trial solution for Bz."""
    x = xt[:, 0]
    t = xt[:, 1]
    f0 = p.f0_Bz
    f1 = p.f1_Bz
    g0 = p.g0_Bz
    A = (1 - x)*f0(xt) + x*f1(xt) + (1 - t)*(g0(xt) - ((1 - x)*g0(x0t) + x*g0(x1t)))
    P = x*(1 - x)*t
    Y = A + P*N[:, 0]
    return Y

# @tf.function
def Ytrial_P(xt, N):
    """Trial solution for P."""
    x = xt[:, 0]
    t = xt[:, 1]
    f0 = p.f0_P
    f1 = p.f1_P
    g0 = p.g0_P
    A = (1 - x)*f0(xt) + x*f1(xt) + (1 - t)*(g0(xt) - ((1 - x)*g0(x0t) + x*g0(x1t)))
    P = x*(1 - x)*t
    Y = A + P*N[:, 0]
    return Y


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
    n_train = nx_train*nt_train
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
    xt_train = create_training_data(nx_train, nt_train)
    np.savetxt(os.path.join(output_dir, "xt_train.dat"), xt_train)

    # Create copies of the training data with all x = 0.
    x0t_train = xt_train.copy()
    x0t_train[:, 0] = 0
    np.savetxt(os.path.join(output_dir, "x0t_train.dat"), x0t_train)

    # Create copies of the training data with all x = 1.
    x1t_train = xt_train.copy()
    x1t_train[:, 0] = 1
    np.savetxt(os.path.join(output_dir, "x1t_train.dat"), x1t_train)

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
    x0t_train_var = tf.Variable(x0t_train)
    global x0t, x1t
    x0t = x0t_train_var
    x1t_train_var = tf.Variable(x1t_train)
    x1t = x1t_train_var

    # Clear the convergence flag to start.
    converged = False

    t_start = datetime.datetime.now()
    print("Training started at", t_start, max_epochs)

    for epoch in range(max_epochs):

        # Run the forward pass.
        with tf.GradientTape(persistent=True) as tape1:
            with tf.GradientTape(persistent=True) as tape0:

                # Compute the network outputs at the training points.
                N_rho = model_rho(xt)
                N_vx  = model_vx( xt)
                N_vy  = model_vy( xt)
                N_vz  = model_vz( xt)
                N_By  = model_By( xt)
                N_Bz  = model_Bz( xt)
                N_P   = model_P(  xt)

                # Compute the trial solutions.
                rho = Ytrial_rho(xt, N_rho)
                vx  = Ytrial_vx( xt, N_vx)
                vy  = Ytrial_vy( xt, N_vy)
                vz  = Ytrial_vz( xt, N_vz)
                By  = Ytrial_By( xt, N_By)
                Bz  = Ytrial_Bz( xt, N_Bz)
                P   = Ytrial_P(  xt, N_P)

            # Compute the gradients of the trial solutions wrt inputs.
            del_rho = tape0.gradient(rho, xt)
            del_vx  = tape0.gradient(vx,  xt)
            del_vy  = tape0.gradient(vy,  xt)
            del_vz  = tape0.gradient(vz,  xt)
            del_By  = tape0.gradient(By,  xt)
            del_Bz  = tape0.gradient(Bz,  xt)
            del_P   = tape0.gradient(P,   xt)

            # Compute the estimates of the differential equations.
            Y = [rho, vx, vy, vz, By, Bz, P]
            del_Y = [del_rho, del_vx, del_vy, del_vz, del_By, del_Bz, del_P]
            G_rho = pde_rho(xt, Y, del_Y)
            G_vx  =  pde_vx(xt, Y, del_Y)
            G_vy  =  pde_vy(xt, Y, del_Y)
            G_vz  =  pde_vz(xt, Y, del_Y)
            G_By  =  pde_By(xt, Y, del_Y)
            G_Bz  =  pde_Bz(xt, Y, del_Y)
            G_P   =   pde_P(xt, Y, del_Y)

            # Compute the loss functions.
            L_rho = tf.math.sqrt(tf.reduce_sum(G_rho**2)/n_train)
            L_vx  = tf.math.sqrt(tf.reduce_sum(G_vx**2) /n_train)
            L_vy  = tf.math.sqrt(tf.reduce_sum(G_vy**2) /n_train)
            L_vz  = tf.math.sqrt(tf.reduce_sum(G_vz**2) /n_train)
            L_By  = tf.math.sqrt(tf.reduce_sum(G_By**2) /n_train)
            L_Bz  = tf.math.sqrt(tf.reduce_sum(G_Bz**2) /n_train)
            L_P   = tf.math.sqrt(tf.reduce_sum(G_P**2)  /n_train)
            L = L_rho + L_vx + L_vy + L_vz + L_By + L_Bz + L_P

        # Save the current losses.
        losses_rho.append(L_rho.numpy())
        losses_vx.append( L_vx.numpy())
        losses_vy.append( L_vy.numpy())
        losses_vz.append( L_vz.numpy())
        losses_By.append( L_By.numpy())
        losses_Bz.append( L_Bz.numpy())
        losses_P.append(  L_P.numpy())
        losses.append(    L.numpy())

        # Check for convergence.
        if epoch > 1:
            loss_delta = losses[-1] - losses[-2]
            if abs(loss_delta) <= tol:
                converged = True
                break

        # Compute the gradient of the loss function wrt the network parameters.
        pgrad_rho = tape1.gradient(L, model_rho.trainable_variables)
        pgrad_vx  = tape1.gradient(L,  model_vx.trainable_variables)
        pgrad_vy  = tape1.gradient(L,  model_vy.trainable_variables)
        pgrad_vz  = tape1.gradient(L,  model_vz.trainable_variables)
        pgrad_By  = tape1.gradient(L,  model_By.trainable_variables)
        pgrad_Bz  = tape1.gradient(L,  model_Bz.trainable_variables)
        pgrad_P   = tape1.gradient(L,   model_P.trainable_variables)

        # Save the parameters used in this epoch.
        phist_rho.append(
            np.hstack(
                (model_rho.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
                 model_rho.trainable_variables[1].numpy(),       # u (H,) row vector
                 model_rho.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
            )
        )
        phist_vx.append(
            np.hstack(
                (model_vx.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
                 model_vx.trainable_variables[1].numpy(),       # u (H,) row vector
                 model_vx.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
            )
        )
        phist_vy.append(
            np.hstack(
                (model_vy.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
                 model_vy.trainable_variables[1].numpy(),       # u (H,) row vector
                 model_vy.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
            )
        )
        phist_vz.append(
            np.hstack(
                (model_vz.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
                 model_vz.trainable_variables[1].numpy(),       # u (H,) row vector
                 model_vz.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
            )
        )
        phist_By.append(
            np.hstack(
                (model_By.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
                 model_By.trainable_variables[1].numpy(),       # u (H,) row vector
                 model_By.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
            )
        )
        phist_Bz.append(
            np.hstack(
                (model_Bz.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
                 model_Bz.trainable_variables[1].numpy(),       # u (H,) row vector
                 model_Bz.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
            )
        )
        phist_P.append(
            np.hstack(
                (model_P.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
                 model_P.trainable_variables[1].numpy(),       # u (H,) row vector
                 model_P.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
            )
        )

        # Update the parameters for this epoch.
        optimizer_rho.apply_gradients(zip(pgrad_rho, model_rho.trainable_variables))
        optimizer_vx.apply_gradients(zip( pgrad_vx,   model_vx.trainable_variables))
        optimizer_vy.apply_gradients(zip( pgrad_vy,   model_vy.trainable_variables))
        optimizer_vz.apply_gradients(zip( pgrad_vz,   model_vz.trainable_variables))
        optimizer_By.apply_gradients(zip( pgrad_By,   model_By.trainable_variables))
        optimizer_Bz.apply_gradients(zip( pgrad_Bz,   model_Bz.trainable_variables))
        optimizer_P.apply_gradients( zip( pgrad_P,     model_P.trainable_variables))

        if epoch % 1 == 0:
            print("Ending epoch %s, loss function = %f" % (epoch, L.numpy()))

    # Save the parameters used in the last epoch.
    phist_rho.append(
        np.hstack(
            (model_rho.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
             model_rho.trainable_variables[1].numpy(),       # u (H,) row vector
             model_rho.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
        )
    )
    phist_vx.append(
        np.hstack(
            (model_vx.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
             model_vx.trainable_variables[1].numpy(),       # u (H,) row vector
             model_vx.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
        )
    )
    phist_vy.append(
        np.hstack(
            (model_vy.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
             model_vy.trainable_variables[1].numpy(),       # u (H,) row vector
             model_vy.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
        )
    )
    phist_vz.append(
        np.hstack(
            (model_vz.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
             model_vz.trainable_variables[1].numpy(),       # u (H,) row vector
             model_vz.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
        )
    )
    phist_By.append(
        np.hstack(
            (model_By.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
             model_By.trainable_variables[1].numpy(),       # u (H,) row vector
             model_By.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
        )
    )
    phist_Bz.append(
        np.hstack(
            (model_Bz.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
             model_Bz.trainable_variables[1].numpy(),       # u (H,) row vector
             model_Bz.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
        )
    )
    phist_P.append(
        np.hstack(
            (model_P.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
             model_P.trainable_variables[1].numpy(),       # u (H,) row vector
             model_P.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
        )
    )

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

    # Save the parameter histories.
    np.savetxt(os.path.join(output_dir, 'phist_rho.dat'), np.array(phist_rho))
    np.savetxt(os.path.join(output_dir, 'phist_vx.dat'),  np.array(phist_vx))
    np.savetxt(os.path.join(output_dir, 'phist_vy.dat'),  np.array(phist_vy))
    np.savetxt(os.path.join(output_dir, 'phist_vz.dat'),  np.array(phist_vz))
    np.savetxt(os.path.join(output_dir, 'phist_By.dat'),  np.array(phist_By))
    np.savetxt(os.path.join(output_dir, 'phist_Bz.dat'),  np.array(phist_Bz))
    np.savetxt(os.path.join(output_dir, 'phist_P.dat'),   np.array(phist_P))

    # Compute and save the trained results at training points.
    with tf.GradientTape(persistent=True) as tape:

        # Compute the network outputs at the training points.
        N_rho = model_rho(xt)
        N_vx  = model_vx( xt)
        N_vy  = model_vy( xt)
        N_vz  = model_vz( xt)
        N_By  = model_By( xt)
        N_Bz  = model_Bz( xt)
        N_P   = model_P(  xt)

        # Compute the trial solutions.
        rho_train = Ytrial_rho(xt, N_rho)
        vx_train  = Ytrial_vx( xt, N_vx)
        vy_train  = Ytrial_vy( xt, N_vy)
        vz_train  = Ytrial_vz( xt, N_vz)
        By_train  = Ytrial_By( xt, N_By)
        Bz_train  = Ytrial_Bz( xt, N_Bz)
        P_train   = Ytrial_P(  xt, N_P)

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
