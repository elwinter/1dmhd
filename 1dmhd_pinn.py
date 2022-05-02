#!/usr/bin/env python

"""Use a set of neural networks to solve the 1-D MHD equations.

This program will use a set of neural networks to solve the coupled partial
differential equations of one-dimensional ideal MHD.

This code uses the PINN method.

The problem is set up as a 2-point boundary value problem in x, and
an initial value problem in t.

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


# Program constants

# Program description.
description = "Solve the 1-D MHD equations with a set of neural networks, using the PINN method."

# Default activation function to use in hidden nodes.
default_activation = "sigmoid"

# Default learning rate.
default_learning_rate = 0.01

# Default maximum number of training epochs.
default_max_epochs = 10

# Default number of hidden nodes per layer.
default_H = 10

# Default number of layers in the fully-connected network, each with H nodes.
default_n_layers = 1

# Default number of training points in the t-dimension.
default_nt_train = 11

# Default number of training points in the x-dimension.
default_nx_train = 11

# Default problem name.
default_problem = "static"

# Default random number generator seed.
default_seed = 0

# Default absolute tolerance for consecutive loss function values to indicate
# convergence.
default_tolerance = 1e-6

# Default normalized weight to apply to the boundary condition loss.
default_w_bc = 0.0

# Name of system information file.
system_information_file = "system_information.txt"

# Name of hyperparameter record file, as an importable Python module.
hyperparameter_file = "hyperparameters.py"

# Name of problem record file, as an importable Python module.
problem_record_file = "problem.py"

# Initial parameter ranges
w0_range = [-0.1, 0.1]
u0_range = [-0.1, 0.1]
v0_range = [-0.1, 0.1]


# Program global variables.

# Global object to hold the problem definition.
p = None


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
        "-a", "--activation", type=str, default=default_activation,
        help="Print debugging output (default: %(default)s)."
    )
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
        "--noconvcheck", action="store_true", default=False,
        help="Do not perform convergence check (default: %(default)s)."
    )
    parser.add_argument(
        "--n_hid", type=int, default=default_H,
        help="Number of hidden nodes per layer (default: %(default)s)"
    )
    parser.add_argument(
        "--n_layers", type=int, default=default_n_layers,
        help="Number of hidden layers (default: %(default)s)"
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
        "--seed", type=int, default=default_seed,
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
    parser.add_argument(
        "-w", "--w_bc", type=float, default=default_w_bc,
        help="Weight for boundary loss (default: %(default)s)."
    )
    return parser


def create_output_directory(path="."):
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


def save_system_information(output_dir="."):
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


def save_hyperparameters(args, output_dir="."):
    """Save the neural network hyperparameters.
    
    Print a record of the hyperparameters of the neural networks in the
    specified directory.

    Parameters
    ----------
    args : dict
        Dictionary of command-line arguments.
    output_dir : str
        Path to directory to contain the report.

    Returns
    -------
    None
    """
    path = os.path.join(output_dir, hyperparameter_file)
    with open(path, "w") as f:
        f.write("n_layers = %s\n" % repr(args.n_layers))
        f.write("H = %s\n" % repr(args.n_hid))
        f.write("w0_range = %s\n" % repr(w0_range))
        f.write("u0_range = %s\n" % repr(u0_range))
        f.write("v0_range = %s\n" % repr(v0_range))
        f.write("activation = %s\n" % repr(args.activation))
        f.write("learning_rate = %s\n" % repr(args.learning_rate))
        f.write("max_epochs = %s\n" % repr(args.max_epochs))
        f.write("nx_train = %s\n" % repr(args.nx_train))
        f.write("nt_train = %s\n" % repr(args.nt_train))
        f.write("random_seed = %s\n" % repr(args.seed))
        f.write("tol = %s\n" % repr(args.tolerance))


def save_problem_definition(args, output_dir="."):
    """Save the problem parameters for the run.
    
    Print a record of the problem description.

    Parameters
    ----------
    args : dict
        Dictionary of command-line arguments.
    output_dir : str
        Path to directory to contain the report.

    Returns
    -------
    None
    """
    path = os.path.join(output_dir, problem_record_file)
    with open(path, "w") as f:
        f.write("problem_name = %s\n" % repr(args.problem))


def create_training_data(nx, nt):
    """Create the training data.
    
    Create and return a set of training data of points evenly spaced in x and
    t. Flatten the data to a list of pairs of points. Also return copies
    of the data containing only internal points, and only boundary points.
    
    Parameters
    ----------
    nx, nt : int
        Number of points in x- and t-dimensions.
    
    Returns
    -------
    xt : np.ndarray, shape (nx*nt, 2)
        Array of all [x, t] points.
    xt_in : np.ndarray, shape ((nx - 2)*(nt - 1)), 2)
        Array of all [x, t] points.
    xt_bc : np.ndarray, shape (2*nt + nx - 2, 2)
        Array of all [x, t] points.
    """
    # Create the array of all training points (x, t), looping over t then x.
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    X = np.repeat(x, nt)
    T = np.tile(t, nx)
    xt = np.vstack([X, T]).T

    # Now split the training data into two groups - inside the BC, and on the BC.
    # Initialize the mask to keep everything.
    mask = np.ones(len(xt), dtype=bool)
    # Mask off the points at x = 0.
    # mask[:nt] = False
    # Mask off the points at x = 1.
    # mask[-nt:] = False
    # Mask off the points at t = 0.
    mask[::nt] = False
    # Keep t = t_max inside.
    # mask[nt_train - 1::nx_train] = False
    xt_in = xt[mask]
    mask = np.logical_not(mask)
    xt_bc = xt[mask]
    return xt, xt_in, xt_bc


def build_model(n_layers, H, activation="sigmoid"):
    """Build a multi-layer neural network model.

    Build a fully-connected, multi-layer neural network with single output.
    Each layer will have H hidden nodes.

    Parameters
    ----------
    n_layers : int
        Number of hidden layers to create.
    H : int
        Number of nodes to use in each hidden layer.
    activation : str
        Name of activation function to use.

    Returns
    -------
    model : tf.keras.Sequential
        The neural network.
    """
    layers = []
    for i in range(n_layers):
        hidden_layer = tf.keras.layers.Dense(
            units=H, use_bias=True,
            activation=tf.keras.activations.deserialize(activation),
            kernel_initializer=tf.keras.initializers.RandomUniform(*w0_range),
            bias_initializer=tf.keras.initializers.RandomUniform(*u0_range)
        )
        layers.append(hidden_layer)
    output_layer = tf.keras.layers.Dense(
        units=1,
        activation=tf.keras.activations.linear,
        kernel_initializer=tf.keras.initializers.RandomUniform(*v0_range),
        use_bias=False,
    )
    layers.append(output_layer)
    model = tf.keras.Sequential(layers)
    return model


# Define the differential equations using TensorFlow operations.

# These equations are taken from:

# https://www.csun.edu/~jb715473/examples/mhd1d.htm

# The original equations are:

# For 1-D flow
# div(B) = 0 = dBx/dx + dBy/dy + dBz/dz
# Bx is constant,
# dBy/dy + dBz/dz = 0 => dBy/dy = - dBz/dz

# The general form of each differential equation is (d are
# partial derivatives)

#     dU/dt + dF/dx = 0

#     U = (rho, rho*vx, rho*vy, rho*vz, By, Bz, E)

#           / rho*vx                       \
#          |  rho*vx**2 + Ptot - Bx**2      |
#          |  rho*vx*vy - Bx*By             |
#      F = |  rho*vx*vz - Bx*Bz             |
#          |  By*vx - Bx*vy                 |
#          |  Bz*vx - Bx*vz                 |
#           \ (E + Ptot)*vx - Bx*(B dot v) /

#     Ptot = P + B**2/2

#     P = (gamma - 1)*(E - rho*v**2/2 - B**2/2)

# xt is the tf.Variable [x, t] of all of the training points.
# Y is the list of tf.Variable [rho, P, vx, vy, vz, By, Bz]
# del_Y is the list of gradients [del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz]

# @tf.function
def pde_rho(xt, Y, del_Y):
    """Differential equation for rho."""
    n = xt.shape[0]
    x = tf.reshape(xt[:, 0], (n, 1))
    t = tf.reshape(xt[:, 1], (n, 1))
    (rho, P, vx, vy, vz, By, Bz) = Y
    (del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz) = del_Y
    drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
    drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
    dvx_dx  = tf.reshape(del_vx[:, 0], (n, 1))
    G = drho_dt + rho*dvx_dx + drho_dx*vx
    return G

# @tf.function
def pde_P(xt, Y, del_Y):
    """Differential equation for P (actually E)."""
    n = xt.shape[0]
    x = tf.reshape(xt[:, 0], (n, 1))
    t = tf.reshape(xt[:, 1], (n, 1))
    (rho, P, vx, vy, vz, By, Bz) = Y
    (del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz) = del_Y
    Bx = p.Bx_0
    drho_dx  = tf.reshape(del_rho[:, 0], (n, 1))
    drho_dt  = tf.reshape(del_rho[:, 1], (n, 1))
    dvx_dx   =  tf.reshape(del_vx[:, 0], (n, 1))
    dvx_dt   =  tf.reshape(del_vx[:, 1], (n, 1))
    dvy_dx   =  tf.reshape(del_vy[:, 0], (n, 1))
    dvy_dt   =  tf.reshape(del_vy[:, 1], (n, 1))
    dvz_dx   =  tf.reshape(del_vz[:, 0], (n, 1))
    dvz_dt   =  tf.reshape(del_vz[:, 1], (n, 1))
    dBx_dx   = 0
    dBy_dx   =  tf.reshape(del_By[:, 0], (n, 1))
    dBy_dt   =  tf.reshape(del_By[:, 1], (n, 1))
    dBz_dx   =  tf.reshape(del_Bz[:, 0], (n, 1))
    dBz_dt   =  tf.reshape(del_Bz[:, 1], (n, 1))
    dP_dx    =   tf.reshape(del_P[:, 0], (n, 1))
    dP_dt    =   tf.reshape(del_P[:, 1], (n, 1))
    Ptot = P + 0.5*(Bx**2 + By**2 + Bz**2)
    # dBx_dx and dBx_dt are 0.
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
        - Bx*(Bx*dvx_dx + dBx_dx*vx + By*dvy_dx + dBy_dx*vy + Bz*dvz_dx + dBz_dx*vz)
        - dBx_dx*(Bx*vx + By*vy + Bz*vz)
    )
    return G

# @tf.function
def pde_vx(xt, Y, del_Y):
    """Differential equation for vx."""
    n = xt.shape[0]
    x = tf.reshape(xt[:, 0], (n, 1))
    t = tf.reshape(xt[:, 1], (n, 1))
    (rho, P, vx, vy, vz, By, Bz) = Y
    (del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz) = del_Y
    Bx = p.Bx_0
    drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
    drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
    dvx_dx  =  tf.reshape(del_vx[:, 0], (n, 1))
    dvx_dt  =  tf.reshape(del_vx[:, 1], (n, 1))
    dBx_dx  = 0.0
    dBy_dx  =  tf.reshape(del_By[:, 0], (n, 1))
    dBz_dx  =  tf.reshape(del_Bz[:, 0], (n, 1))
    dP_dx   =   tf.reshape(del_P[:, 0], (n, 1))
    dPtot_dx = dP_dx + Bx*dBx_dx + By*dBy_dx + Bz*dBz_dx
    G = (
        rho*dvx_dt + drho_dt*vx
        + rho*2*vx*dvx_dx + drho_dx*vx**2 + dPtot_dx - 2*Bx*dBx_dx
    )
    return G

# @tf.function
def pde_vy(xt, Y, del_Y):
    """Differential equation for vy."""
    n = xt.shape[0]
    x = tf.reshape(xt[:, 0], (n, 1))
    t = tf.reshape(xt[:, 1], (n, 1))
    (rho, P, vx, vy, vz, By, Bz) = Y
    (del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz) = del_Y
    Bx = p.Bx_0
    drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
    drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
    dvx_dx  =  tf.reshape(del_vx[:, 0], (n, 1))
    dvy_dx  =  tf.reshape(del_vy[:, 0], (n, 1))
    dvy_dt  =  tf.reshape(del_vy[:, 1], (n, 1))
    dBx_dx  = 0.0
    dBy_dx  =  tf.reshape(del_By[:, 0], (n, 1))
    G = (
        rho*dvy_dt + drho_dt*vy
        + rho*(vx*dvy_dx + dvx_dx*vy) + drho_dx*vx*vy
        - Bx*dBy_dx - dBx_dx*By
    )
    return G

# @tf.function
def pde_vz(xt, Y, del_Y):
    """Differential equation for vz."""
    n = xt.shape[0]
    x = tf.reshape(xt[:, 0], (n, 1))
    t = tf.reshape(xt[:, 1], (n, 1))
    (rho, P, vx, vy, vz, By, Bz) = Y
    (del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz) = del_Y
    Bx = p.Bx_0
    drho_dx = tf.reshape(del_rho[:, 0], (n, 1))
    drho_dt = tf.reshape(del_rho[:, 1], (n, 1))
    dvx_dx  =  tf.reshape(del_vx[:, 0], (n, 1))
    dvz_dx  =  tf.reshape(del_vz[:, 0], (n, 1))
    dvz_dt  =  tf.reshape(del_vz[:, 1], (n, 1))
    dBx_dx  = 0.0
    dBz_dx  =  tf.reshape(del_Bz[:, 0], (n, 1))
    G = (
        rho*dvz_dt + drho_dt*vz
        + rho*(vx*dvz_dx + dvx_dx*vz) + drho_dx*vx*vz
        - Bx*dBz_dx - dBx_dx*Bz
    )
    return G

# @tf.function
def pde_By(xt, Y, del_Y):
    """Differential equation for By."""
    n = xt.shape[0]
    x = tf.reshape(xt[:, 0], (n, 1))
    t = tf.reshape(xt[:, 1], (n, 1))
    (rho, P, vx, vy, vz, By, Bz) = Y
    (del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz) = del_Y
    Bx = p.Bx_0
    dvx_dx = tf.reshape(del_vx[:, 0], (n, 1))
    dvy_dx = tf.reshape(del_vy[:, 0], (n, 1))
    dBx_dx = 0
    dBy_dx = tf.reshape(del_By[:, 0], (n, 1))
    dBy_dt = tf.reshape(del_By[:, 1], (n, 1))
    G = dBy_dt + By*dvx_dx + dBy_dx*vx - Bx*dvy_dx - dBx_dx*vy
    return G

# @tf.function
def pde_Bz(xt, Y, del_Y):
    """Differential equation for Bz."""
    n = xt.shape[0]
    x = tf.reshape(xt[:, 0], (n, 1))
    t = tf.reshape(xt[:, 1], (n, 1))
    (rho, P, vx, vy, vz, By, Bz) = Y
    (del_rho, del_P, del_vx, del_vy, del_vz, del_By, del_Bz) = del_Y
    Bx = p.Bx_0
    dvx_dx  = tf.reshape(del_vx[:, 0], (n, 1))
    dvz_dx  = tf.reshape(del_vz[:, 0], (n, 1))
    dBx_dx = 0
    dBz_dx  = tf.reshape(del_Bz[:, 0], (n, 1))
    dBz_dt  = tf.reshape(del_Bz[:, 1], (n, 1))
    G = dBz_dt + Bz*dvx_dx + dBz_dx*vx - Bx*dvz_dx - dBx_dx*vz
    return G


def main():
    """Begin main program."""

    # Set up the command-line parser.
    parser = create_command_line_parser()

    # Parse the command-line arguments.
    args = parser.parse_args()
    activation = args.activation
    debug = args.debug
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    H = args.n_hid
    noconvcheck = args.noconvcheck
    n_layers = args.n_layers
    nt_train = args.nt_train
    nx_train = args.nx_train
    problem = args.problem
    seed = args.seed
    tol = args.tolerance
    verbose = args.verbose
    w_bc = args.w_bc
    if debug:
        print("args = %s" % args)

    # Import the problem to solve.
    global p
    if verbose:
        print("Importing definition for problem '%s'." % problem)
    p = import_module(problem)

    # Set up the output directory under the current directory.
    output_dir = os.path.join(".", problem)
    create_output_directory(output_dir)

    # Record system information, network parameters, and problem definition.
    if verbose:
        print("Recording system information, model hyperparameters, and problem definition.")
    save_system_information(output_dir)
    save_hyperparameters(args, output_dir)
    save_problem_definition(args, output_dir)

    # Create and save the training data.
    if verbose:
        print("Creating and saving training data.")
    xt_train, xt_train_in, xt_train_bc = create_training_data(
        nx_train, nt_train
    )
    np.savetxt(os.path.join(output_dir, "xt_train.dat"), xt_train)
    n_train = len(xt_train)
    np.savetxt(os.path.join(output_dir, "xt_train_in.dat"), xt_train_in)
    n_train_in = len(xt_train_in)
    np.savetxt(os.path.join(output_dir, "xt_train_bc.dat"), xt_train_bc)
    n_train_bc = len(xt_train_bc)
    assert n_train == n_train_in + n_train_bc

    # Compute the boundary condition values.
    if verbose:
        print("Computing boundary conditions.")
    bc = p.compute_boundary_conditions(xt_train_bc)
    rho_bc0 = tf.Variable(bc[:, 0].reshape(n_train_bc, 1), dtype="float32")
    P_bc0   = tf.Variable(bc[:, 1].reshape(n_train_bc, 1), dtype="float32")
    vx_bc0  = tf.Variable(bc[:, 2].reshape(n_train_bc, 1), dtype="float32")
    vy_bc0  = tf.Variable(bc[:, 3].reshape(n_train_bc, 1), dtype="float32")
    vz_bc0  = tf.Variable(bc[:, 4].reshape(n_train_bc, 1), dtype="float32")
    By_bc0  = tf.Variable(bc[:, 5].reshape(n_train_bc, 1), dtype="float32")
    Bz_bc0  = tf.Variable(bc[:, 6].reshape(n_train_bc, 1), dtype="float32")

    # Compute the weight for the interior points.
    w_in = 1.0 - w_bc

    # Build the models.
    if verbose:
        print("Creating neural networks.")
    model_rho = build_model(n_layers, H, activation)
    model_P   = build_model(n_layers, H, activation)
    model_vx  = build_model(n_layers, H, activation)
    model_vy  = build_model(n_layers, H, activation)
    model_vz  = build_model(n_layers, H, activation)
    model_By  = build_model(n_layers, H, activation)
    model_Bz  = build_model(n_layers, H, activation)

    # Create the optimizer.
    if verbose:
        print("Creating optimizer.")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Train the models.

    # Create history variables.
    losses_rho = []
    losses_P   = []
    losses_vx  = []
    losses_vy  = []
    losses_vz  = []
    losses_By  = []
    losses_Bz  = []
    losses     = []

    # Set the random number seed for reproducibility.
    tf.random.set_seed(seed)

    # Rename the training data Variables for convenience.
    xt_train_var = tf.Variable(xt_train, dtype="float32")
    xt = xt_train_var
    xt_train_in_var = tf.Variable(xt_train_in, dtype="float32")
    xt_in = xt_train_in_var
    xt_train_bc_var = tf.Variable(xt_train_bc, dtype="float32")
    xt_bc = xt_train_bc_var

    # Clear the convergence flag to start.
    converged = False

    t_start = datetime.datetime.now()
    if verbose:
        print("Training started at", t_start, max_epochs)

    for epoch in range(max_epochs):

        # Run the forward pass.
        with tf.GradientTape(persistent=True) as tape1:
            with tf.GradientTape(persistent=True) as tape0:

                # Compute the network outputs within the domain.
                rho_in = model_rho(xt_in)
                P_in   = model_P(  xt_in)
                vx_in  = model_vx( xt_in)
                vy_in  = model_vy( xt_in)
                vz_in  = model_vz( xt_in)
                By_in  = model_By( xt_in)
                Bz_in  = model_Bz( xt_in)

                # Compute the network outputs on the boundaries.
                rho_bc = model_rho(xt_bc)
                P_bc   = model_P(  xt_bc)
                vx_bc  = model_vx( xt_bc)
                vy_bc  = model_vy( xt_bc)
                vz_bc  = model_vz( xt_bc)
                By_bc  = model_By( xt_bc)
                Bz_bc  = model_Bz( xt_bc)

            # Compute the gradients of the network outputs wrt inputs at the interior training points.
            del_rho_in = tape0.gradient(rho_in, xt_in)
            del_P_in   = tape0.gradient(P_in,   xt_in)
            del_vx_in  = tape0.gradient(vx_in,  xt_in)
            del_vy_in  = tape0.gradient(vy_in,  xt_in)
            del_vz_in  = tape0.gradient(vz_in,  xt_in)
            del_By_in  = tape0.gradient(By_in,  xt_in)
            del_Bz_in  = tape0.gradient(Bz_in,  xt_in)

            # Compute the estimates of the differential equations at the interior training points.
            Y_in = [rho_in, P_in, vx_in, vy_in, vz_in, By_in, Bz_in]
            del_Y_in = [del_rho_in, del_P_in, del_vx_in, del_vy_in, del_vz_in, del_By_in, del_Bz_in]
            G_rho_in = pde_rho(xt_in, Y_in, del_Y_in)
            G_P_in   =   pde_P(xt_in, Y_in, del_Y_in)
            G_vx_in  =  pde_vx(xt_in, Y_in, del_Y_in)
            G_vy_in  =  pde_vy(xt_in, Y_in, del_Y_in)
            G_vz_in  =  pde_vz(xt_in, Y_in, del_Y_in)
            G_By_in  =  pde_By(xt_in, Y_in, del_Y_in)
            G_Bz_in  =  pde_Bz(xt_in, Y_in, del_Y_in)

            # Compute the errors in the computed BC.
            E_rho_bc = rho_bc - rho_bc0
            E_P_bc   = P_bc - P_bc0
            E_vx_bc  = vx_bc - vx_bc0
            E_vy_bc  = vy_bc - vy_bc0
            E_vz_bc  = vz_bc - vz_bc0
            E_By_bc  = By_bc - By_bc0
            E_Bz_bc  = Bz_bc - Bz_bc0

            # Compute the loss functions for the interior training points.
            L_rho_in = tf.math.sqrt(tf.reduce_sum(G_rho_in**2)/n_train_in)
            L_P_in   = tf.math.sqrt(tf.reduce_sum(G_P_in**2)  /n_train_in)
            L_vx_in  = tf.math.sqrt(tf.reduce_sum(G_vx_in**2) /n_train_in)
            L_vy_in  = tf.math.sqrt(tf.reduce_sum(G_vy_in**2) /n_train_in)
            L_vz_in  = tf.math.sqrt(tf.reduce_sum(G_vz_in**2) /n_train_in)
            L_By_in  = tf.math.sqrt(tf.reduce_sum(G_By_in**2) /n_train_in)
            L_Bz_in  = tf.math.sqrt(tf.reduce_sum(G_Bz_in**2) /n_train_in)
            L_in = L_rho_in + L_P_in + L_vx_in + L_vy_in + L_vz_in + L_By_in + L_Bz_in

            # Compute the loss functions for the boundary points.
            L_rho_bc = tf.math.sqrt(tf.reduce_sum(E_rho_bc**2)/n_train_bc)
            L_P_bc   = tf.math.sqrt(tf.reduce_sum(E_P_bc**2)  /n_train_bc)
            L_vx_bc  = tf.math.sqrt(tf.reduce_sum(E_vx_bc**2) /n_train_bc)
            L_vy_bc  = tf.math.sqrt(tf.reduce_sum(E_vy_bc**2) /n_train_bc)
            L_vz_bc  = tf.math.sqrt(tf.reduce_sum(E_vz_bc**2) /n_train_bc)
            L_By_bc  = tf.math.sqrt(tf.reduce_sum(E_By_bc**2) /n_train_bc)
            L_Bz_bc  = tf.math.sqrt(tf.reduce_sum(E_Bz_bc**2) /n_train_bc)
            L_bc = L_rho_bc + L_P_bc + L_vx_bc + L_vy_bc + L_vz_bc + L_By_bc + L_Bz_bc

            # Compute the weighted total losses.
            L_rho = w_in*L_rho_in + w_bc*L_rho_bc
            L_P =   w_in*L_P_in   + w_bc*L_P_bc
            L_vx =  w_in*L_vx_in  + w_bc*L_vx_bc
            L_vy =  w_in*L_vy_in  + w_bc*L_vy_bc
            L_vz =  w_in*L_vz_in  + w_bc*L_vz_bc
            L_By =  w_in*L_By_in  + w_bc*L_By_bc
            L_Bz =  w_in*L_Bz_in  + w_bc*L_Bz_bc
            L =     w_in*L_in     + w_bc*L_bc

        # Save the current losses.
        losses_rho.append(L_rho.numpy())
        losses_P.append(  L_P.numpy())
        losses_vx.append( L_vx.numpy())
        losses_vy.append( L_vy.numpy())
        losses_vz.append( L_vz.numpy())
        losses_By.append( L_By.numpy())
        losses_Bz.append( L_Bz.numpy())
        losses.append(    L.numpy())

        # Check for convergence.
        if not noconvcheck:
            if epoch > 1:
                loss_delta = losses[-1] - losses[-2]
                if abs(loss_delta) <= tol:
                    converged = True
                    break

        # Compute the gradient of the loss function wrt the network parameters.
        pgrad_rho = tape1.gradient(L, model_rho.trainable_variables)
        pgrad_P   = tape1.gradient(L,   model_P.trainable_variables)
        pgrad_vx  = tape1.gradient(L,  model_vx.trainable_variables)
        pgrad_vy  = tape1.gradient(L,  model_vy.trainable_variables)
        pgrad_vz  = tape1.gradient(L,  model_vz.trainable_variables)
        pgrad_By  = tape1.gradient(L,  model_By.trainable_variables)
        pgrad_Bz  = tape1.gradient(L,  model_Bz.trainable_variables)

        # Update the parameters for this epoch.
        optimizer.apply_gradients(zip(pgrad_rho, model_rho.trainable_variables))
        optimizer.apply_gradients(zip(pgrad_P,   model_P.trainable_variables))
        optimizer.apply_gradients(zip(pgrad_vx,  model_vx.trainable_variables))
        optimizer.apply_gradients(zip(pgrad_vy,  model_vy.trainable_variables))
        optimizer.apply_gradients(zip(pgrad_vz,  model_vz.trainable_variables))
        optimizer.apply_gradients(zip(pgrad_By,  model_By.trainable_variables))
        optimizer.apply_gradients(zip(pgrad_Bz,  model_Bz.trainable_variables))

        if verbose and epoch % 1 == 0:
            print("Ending epoch %s, loss function = %f" % (epoch, L.numpy()))

    # Count the last epoch.
    n_epochs = epoch + 1

    t_stop = datetime.datetime.now()
    t_elapsed = t_stop - t_start
    if verbose:
        print("Training stopped at", t_stop)
        print("Total training time was %s seconds." % t_elapsed.total_seconds())
        print("Epochs: %d" % n_epochs)
        print("Final value of loss function: %f" % losses[-1])
        print("converged = %s" % converged)

    # Save the loss function histories.
    if verbose:
        print("Saving loss function histories.")
    np.savetxt(os.path.join(output_dir, 'losses_rho.dat'), np.array(losses_rho))
    np.savetxt(os.path.join(output_dir, 'losses_P.dat'),   np.array(losses_P))
    np.savetxt(os.path.join(output_dir, 'losses_vx.dat'),  np.array(losses_vx))
    np.savetxt(os.path.join(output_dir, 'losses_vy.dat'),  np.array(losses_vy))
    np.savetxt(os.path.join(output_dir, 'losses_vz.dat'),  np.array(losses_vz))
    np.savetxt(os.path.join(output_dir, 'losses_By.dat'),  np.array(losses_By))
    np.savetxt(os.path.join(output_dir, 'losses_Bz.dat'),  np.array(losses_Bz))
    np.savetxt(os.path.join(output_dir, 'losses.dat'),     np.array(losses))

    # Compute and save the trained results at training points.
    if verbose:
        print("Computing and saving trained results.")
    with tf.GradientTape(persistent=True) as tape:

        # Compute the trial solutions.
        rho_train = model_rho(xt)
        P_train   = model_P(  xt)
        vx_train  = model_vx( xt)
        vy_train  = model_vy( xt)
        vz_train  = model_vz( xt)
        By_train  = model_By( xt)
        Bz_train  = model_Bz( xt)

    np.savetxt(os.path.join(output_dir, "rho_train.dat"), rho_train.numpy().reshape((n_train,)))
    np.savetxt(os.path.join(output_dir, "P_train.dat"),     P_train.numpy().reshape((n_train,)))
    np.savetxt(os.path.join(output_dir, "vx_train.dat"),   vx_train.numpy().reshape((n_train,)))
    np.savetxt(os.path.join(output_dir, "vy_train.dat"),   vy_train.numpy().reshape((n_train,)))
    np.savetxt(os.path.join(output_dir, "vz_train.dat"),   vz_train.numpy().reshape((n_train,)))
    np.savetxt(os.path.join(output_dir, "By_train.dat"),   By_train.numpy().reshape((n_train,)))
    np.savetxt(os.path.join(output_dir, "Bz_train.dat"),   Bz_train.numpy().reshape((n_train,)))


if __name__ == "__main__":
    """Begin main program."""
    main()
