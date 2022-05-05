#!/usr/bin/env python

"""Use a set of neural networks to solve simple ODE.

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

# Default number of training points in the x-dimension.
default_nx_train = 11

# Default number of validation points in the x-dimension.
default_nx_val = 101

# Default TF precision for computations.
default_precision = "float32"

# Default problem name.
default_problem = "linear"

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
        "--nx_train", type=int, default=default_nx_train,
        help="Number of equally-spaced training points in x dimension (default: %(default)s)"
    )
    parser.add_argument(
        "--nx_val", type=int, default=default_nx_val,
        help="Number of equally-spaced validation points in x dimension (default: %(default)s)"
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
        f.write("random_seed = %s\n" % repr(args.seed))
        f.write("tol = %s\n" % repr(args.tolerance))
        f.write("precision = %s\n" % repr(args.precision))


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
    # nt_train = args.nt_train
    nx_train = args.nx_train
    nx_val = args.nx_val
    problem = args.problem
    seed = args.seed
    tol = args.tolerance
    verbose = args.verbose
    w_bc = args.w_bc
    if debug:
        print("args = %s" % args)

    # Set the backend TensorFlow precision.
    tf.keras.backend.set_floatx(args.precision)

    # Import the problem to solve.
    global p
    if verbose:
        print("Importing definition for problem '%s'." % problem)
    p = import_module(problem)
    if debug:
        print("p = %s" % p)

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
    x_train, x_train_in, x_train_bc = p.create_training_data(nx_train)
    np.savetxt(os.path.join(output_dir, "x_train.dat"), x_train)
    n_train = len(x_train)
    np.savetxt(os.path.join(output_dir, "x_train_in.dat"), x_train_in)
    n_train_in = len(x_train_in)
    np.savetxt(os.path.join(output_dir, "x_train_bc.dat"), x_train_bc)
    n_train_bc = len(x_train_bc)
    assert n_train == n_train_in + n_train_bc

    # Compute the initial condition value.
    if verbose:
        print("Computing boundary conditions.")
    ic = tf.Variable([[p.ic]], dtype=args.precision)

    # Compute the weight for the interior points.
    w_in = 1.0 - w_bc

    # Build the models.
    if verbose:
        print("Creating neural network.")
    model = build_model(n_layers, H, activation)

    # Create the optimizer.
    if verbose:
        print("Creating optimizer.")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Train the models.

    # Create history variables.
    losses_y = []
    losses = []

    # Set the random number seed for reproducibility.
    tf.random.set_seed(seed)

    # Rename the training data Variables for convenience.
    x_train_var = tf.Variable(x_train.reshape(n_train, 1), dtype=args.precision)
    x = x_train_var
    x_train_in_var = tf.Variable(x_train_in.reshape(n_train_in, 1), dtype=args.precision)
    x_in = x_train_in_var
    x_train_bc_var = tf.Variable(x_train_bc.reshape(n_train_bc, 1), dtype=args.precision)
    x_bc = x_train_bc_var

    # Clear the convergence flag to start.
    converged = False

    t_start = datetime.datetime.now()
    if verbose:
        print("Training started at %s, max_epochs = %s" % (t_start, max_epochs))

    for epoch in range(max_epochs):

        # Run the forward pass.
        with tf.GradientTape(persistent=True) as tape1:
            with tf.GradientTape(persistent=True) as tape0:

                # Compute the network outputs within the domain.
                y_in = model(x_in)

                # Compute the network outputs on the boundaries.
                y_bc = model(x_bc)

            # Compute the gradients of the network outputs wrt inputs at the interior training points.
            del_y_in = tape0.gradient(y_in, x_in)

            # Compute the estimate of the differential equation at the interior training points.
            Y_in = [y_in]
            del_Y_in = [del_y_in]
            G_y_in = p.differential_equation(x_in, Y_in, del_Y_in)

            # Compute the errors in the computed initial condition.
            E_y_bc = y_bc - p.ic

            # Compute the loss functions for the interior training points.
            L_y_in = tf.math.sqrt(tf.reduce_sum(G_y_in**2)/n_train_in)
            L_in = L_y_in

            # Compute the loss functions for the boundary points.
            L_y_bc = tf.math.sqrt(tf.reduce_sum(E_y_bc**2)/n_train_bc)
            L_bc = L_y_bc

            # Compute the weighted total losses.
            L_y = w_in*L_y_in + w_bc*L_y_bc
            L = w_in*L_in + w_bc*L_bc

        # Save the current losses.
        losses_y.append(L_y.numpy())
        losses.append(L.numpy())

        # Check for convergence.
        if not noconvcheck:
            if epoch > 1:
                loss_delta = losses[-1] - losses[-2]
                if abs(loss_delta) <= tol:
                    converged = True
                    break

        # Compute the gradient of the loss function wrt the network parameters.
        pgrad_y = tape1.gradient(L, model.trainable_variables)

        # Update the parameters for this epoch.
        optimizer.apply_gradients(zip(pgrad_y, model.trainable_variables))

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
    np.savetxt(os.path.join(output_dir, 'losses_y.dat'), np.array(losses_y))
    np.savetxt(os.path.join(output_dir, 'losses.dat'),     np.array(losses))

    # Compute and save the trained results at training points.
    if verbose:
        print("Computing and saving trained results.")
    with tf.GradientTape(persistent=True) as tape:
        y_train = model(x)
    np.savetxt(os.path.join(output_dir, "y_train.dat"), y_train.numpy().reshape((n_train,)))

    # Compute and save the trained results at validation points.
    if verbose:
        print("Computing and saving validation results.")
    x_val, x_val_in, x_val_bc = p.create_training_data(nx_val)
    n_val = len(x_val)
    n_val_in = len(x_val_in)
    n_val_bc = len(x_val_bc)
    assert n_val_in + n_val_bc == n_val
    np.savetxt(os.path.join(output_dir, "x_val.dat"), x_val)
    np.savetxt(os.path.join(output_dir, "x_val_in.dat"), x_val_in)
    np.savetxt(os.path.join(output_dir, "x_val_bc.dat"), x_val_bc)
    x_val = tf.Variable(x_val.reshape(n_val, 1), dtype=args.precision)
    with tf.GradientTape(persistent=True) as tape:
        y_val = model(x_val)
    np.savetxt(os.path.join(output_dir, "y_val.dat"), y_val.numpy().reshape((n_val,)))

if __name__ == "__main__":
    """Begin main program."""
    main()
