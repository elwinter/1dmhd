"""Common code for PINN software.

This module provides a set of standard functions used by all of the programs
in the PINN family of programs.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard modules.
import argparse
import datetime
import os
import platform
import shutil
import sys

# Import 3rd-party modules.
import numpy as np
import tensorflow as tf


# Default values for command-line arguments

# Default activation function to use in hidden nodes.
default_activation = "sigmoid"

# Default learning rate.
default_learning_rate = 0.01

# Default maximum number of training epochs.
default_max_epochs = 10

# Default number of hidden nodes per layer.
default_n_hid = 10

# Default number of layers in the fully-connected network, each with n_hid
# nodes.
default_n_layers = 1

# Default number of training points in the x-dimension.
default_nx_train = 11

# Default number of validation points in the x-dimension.
default_nx_val = 101

# Default TensorFlow precision for computations.
default_precision = "float32"

# Default random number generator seed.
default_seed = 0

# Default absolute tolerance for consecutive loss function values to indicate
# convergence.
default_tolerance = 1e-6

# Default normalized weight to apply to the boundary condition loss function.
default_w_bc = 0.0


# Module constants

# Name of file to hold the system information report.
system_information_file = "system_information.txt"

# Name of file to hold the network hyperparameters, as an importable Python
# module.
hyperparameter_file = "hyperparameters.py"

# Initial parameter ranges
w0_range = [-0.1, 0.1]  # Hidden layer weights
u0_range = [-0.1, 0.1]  # Hidden layer biases
v0_range = [-0.1, 0.1]  # Output layer weights


def create_command_line_argument_parser(description, default_problem):
    """Create the common command-line argument parser.

    Create the common command-line argument parser.

    Parameters
    ----------
    description : str
        Description string for program creating this command-line parser.
    default_problem : str
        Name of module containing the default problem definition

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser for command-line arguments.
    """
    parser = argparse.ArgumentParser(description)
    parser.add_argument(
        "-a", "--activation", type=str, default=default_activation,
        help="Print debugging output (default: %(default)s)."
    )
    parser.add_argument(
        "--convcheck", action="store_true",
        help="Perform convergence check (default: %(default)s)."
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
        "--no-convcheck", dest="convcheck", action="store_false",
        help="Do not perform convergence check (default: %(default)s)."
    )
    parser.add_argument(
        "--no-save_model", dest="save_model", action="store_false",
        help="Do not save the trained model (default: %(default)s)."
    )
    parser.add_argument(
        "--no-save_weights", dest="save_weights", action="store_false",
        help="Do not save the model weights at each epoch "
             "(default: %(default)s)."
    )
    parser.add_argument(
        "--n_hid", type=int, default=default_n_hid,
        help="Number of hidden nodes per layer (default: %(default)s)"
    )
    parser.add_argument(
        "--n_layers", type=int, default=default_n_layers,
        help="Number of hidden layers (default: %(default)s)"
    )
    parser.add_argument(
        "--nx_train", type=int, default=default_nx_train,
        help="Number of equally-spaced training points in x dimension "
             "(default: %(default)s)"
    )
    parser.add_argument(
        "--nx_val", type=int, default=default_nx_val,
        help="Number of equally-spaced validation points in x dimension "
             "(default: %(default)s)"
    )
    parser.add_argument(
        "--precision", type=str, default=default_precision,
        help="Precision to use in TensorFlow solution (default: %(default)s)"
    )
    parser.add_argument(
        "--problem", type=str, default=default_problem,
        help="Name of problem to solve (default: %(default)s)"
    )
    parser.add_argument(
        "--save_model", action="store_true",
        help="Save the trained model (default: %(default)s)."
    )
    parser.add_argument(
        "--save_weights", action="store_true",
        help="Save the model weights at each epoch (default: %(default)s)."
    )
    parser.add_argument(
        "--seed", type=int, default=default_seed,
        help="Seed for random number generator (default: %(default)s)"
    )
    parser.add_argument(
        "--tolerance", type=float, default=default_tolerance,
        help="Absolute loss function convergence tolerance "
             "(default: %(default)s)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Print verbose output (default: %(default)s)."
    )
    parser.add_argument(
        "-w", "--w_bc", type=float, default=default_w_bc,
        help="Normalized weight for boundary condition loss function "
             "(default: %(default)s)."
    )
    parser.set_defaults(convcheck=True)
    parser.set_defaults(save_model=True)
    parser.set_defaults(save_weights=False)
    return parser


def create_output_directory(path):
    """Create an output directory for the results.

    Create the specified directory. Do nothing if it already exists.

    Parameters
    ----------
    path : str
        Path to directory to create.

    Returns
    -------
    None
    """
    try:
        os.mkdir(path)
    except FileExistsError:
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
        f.write("Python implementation: %s\n" %
                platform.python_implementation())
        f.write("Python file: %s\n" % __file__)
        f.write("NumPy version: %s\n" % np.__version__)
        f.write("TensorFlow version: %s\n" % tf.__version__)


def save_problem_definition(problem, output_dir):
    """Save the problem definition for the run.

    Copy the problem definition file to the output directory.

    Parameters
    ----------
    problem : module
        Imported module object for problem definition.
    output_dir : str
        Path to directory to contain the copy of the problem definition file.

    Returns
    -------
    None
    """
    # Copy the problem definition file to the output directory.
    shutil.copy(problem.__file__, output_dir)


def build_model(n_layers, n_hidden, activation):
    """Build a multi-layer neural network model.

    Build a fully-connected, multi-layer neural network with single output.
    Each layer will have H hidden nodes. Each hidden node has weights and
    a bias, and uses the specified activation function.

    The number of inputs is determined when the network is first used.

    Parameters
    ----------
    n_layers : int
        Number of hidden layers to create.
    n_hidden : int
        Number of nodes to use in each hidden layer.
    activation : str
        Name of activation function (from TensorFlow) to use.

    Returns
    -------
    model : tf.keras.Sequential
        The neural network.
    """
    layers = []
    for _ in range(n_layers):
        hidden_layer = tf.keras.layers.Dense(
            units=n_hidden, use_bias=True,
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


def save_hyperparameters(args, output_dir):
    """Save the neural network hyperparameters.

    Print a record of the hyperparameters of the neural network in the
    specified directory, as an importable python module.

    Parameters
    ----------
    args : dict
        Dictionary of command-line arguments.
    output_dir : str
        Path to directory to contain the report.

    Returns
    -------
    path : str
        Path to hyperparameter file.
    """
    path = os.path.join(output_dir, hyperparameter_file)
    with open(path, "w") as f:
        f.write("activation = %s\n" % repr(args.activation))
        f.write("convcheck = %s\n" % repr(args.convcheck))
        f.write("learning_rate = %s\n" % repr(args.learning_rate))
        f.write("max_epochs = %s\n" % repr(args.max_epochs))
        f.write("H = %s\n" % repr(args.n_hid))
        f.write("n_layers = %s\n" % repr(args.n_layers))
        f.write("nx_train = %s\n" % repr(args.nx_train))
        f.write("nx_val = %s\n" % repr(args.nx_val))
        f.write("precision = %s\n" % repr(args.precision))
        f.write("random_seed = %s\n" % repr(args.seed))
        f.write("tolerance = %s\n" % repr(args.tolerance))
        f.write("w_bc = %s\n" % repr(args.w_bc))
        f.write("w0_range = %s\n" % repr(w0_range))
        f.write("u0_range = %s\n" % repr(u0_range))
        f.write("v0_range = %s\n" % repr(v0_range))
    return path
