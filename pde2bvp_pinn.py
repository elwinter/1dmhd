#!/usr/bin/env python

"""Use a neural network to solve a 2nd-order PDE BVP.

This program will use a neural network to solve a 2nd-order partial
differential equation boundary value problem.

This code uses the PINN method.

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
description = "Solve a 2nd-order PDE BVP using the PINN method."

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

# Default number of training points in the y-dimension.
default_ny_train = 11

# Default number of validation points in the x-dimension.
default_nx_val = 101

# Default number of validation points in the y-dimension.
default_ny_val = 101

# Default TF precision for computations.
default_precision = "float32"

# Default problem name.
default_problem = "lagaris05"

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
        help="Do not save the model weights at each epoch (default: %(default)s)."
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
        "--ny_train", type=int, default=default_ny_train,
        help="Number of equally-spaced training points in y dimension (default: %(default)s)"
    )
    parser.add_argument(
        "--nx_val", type=int, default=default_nx_val,
        help="Number of equally-spaced validation points in x dimension (default: %(default)s)"
    )
    parser.add_argument(
        "--ny_val", type=int, default=default_ny_val,
        help="Number of equally-spaced validation points in y dimension (default: %(default)s)"
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
    parser.set_defaults(convcheck=True)
    parser.set_defaults(save_model=True)
    parser.set_defaults(save_weights=False)
    return parser


def create_output_directory(path="."):
    """Create an output directory for the results.
    
    Create the specified directory. Skip if it already exists.

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
        f.write("ny_train = %s\n" % repr(args.ny_train))
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
    convcheck = args.convcheck
    debug = args.debug
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    H = args.n_hid
    n_layers = args.n_layers
    nx_train = args.nx_train
    ny_train = args.ny_train
    nx_val = args.nx_val
    ny_val = args.ny_val
    problem = args.problem
    save_model = args.save_model
    save_weights = args.save_weights
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
    xy_train, xy_train_in, xy_train_bc = p.create_training_data(nx_train, ny_train)
    np.savetxt(os.path.join(output_dir, "xy_train.dat"), xy_train)
    n_train = len(xy_train)
    np.savetxt(os.path.join(output_dir, "xy_train_in.dat"), xy_train_in)
    n_train_in = len(xy_train_in)
    np.savetxt(os.path.join(output_dir, "xy_train_bc.dat"), xy_train_bc)
    n_train_bc = len(xy_train_bc)
    assert n_train == n_train_in + n_train_bc

    # Compute the boundary condition values.
    if verbose:
        print("Computing boundary conditions.")
    bc = p.compute_boundary_conditions(xy_train_bc)
    bc = bc.reshape((n_train_bc, 1))
    bc = tf.Variable(bc, dtype=args.precision)

    # Compute the weight for the interior points.
    w_in = 1.0 - w_bc

    # Build the model.
    if verbose:
        print("Creating neural network.")
    model = build_model(n_layers, H, activation)

    # Create the optimizer.
    if verbose:
        print("Creating optimizer.")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Train the models.

    # Create history variables.
    losses = []

    # Set the random number seed for reproducibility.
    tf.random.set_seed(seed)

    # Rename the training data Variables for convenience.
    xy_train_var = tf.Variable(xy_train.reshape(n_train, 2), dtype=args.precision)
    xy = xy_train_var
    xy_train_in_var = tf.Variable(xy_train_in.reshape(n_train_in, 2), dtype=args.precision)
    xy_in = xy_train_in_var
    xy_train_bc_var = tf.Variable(xy_train_bc.reshape(n_train_bc, 2), dtype=args.precision)
    xy_bc = xy_train_bc_var

    # Clear the convergence flag to start.
    converged = False

    t_start = datetime.datetime.now()
    if verbose:
        print("Training started at %s, max_epochs = %s" % (t_start, max_epochs))

    for epoch in range(max_epochs):

        # Run the forward pass.
        with tf.GradientTape(persistent=True) as tape_param:
            with tf.GradientTape(persistent=True) as tape2:
                with tf.GradientTape(persistent=True) as tape1:

                    # Compute the network outputs within the domain.
                    Y_in = model(xy_in)

                    # Compute the network outputs on the boundaries.
                    Y_bc = model(xy_bc)

                # Compute the first derivatives at the interior training
                # points.
                delY_in = tape1.gradient(Y_in, xy_in)

            # Compute the second derivatives at the interior training points.
            del2Y_jac_in = tape2.jacobian(delY_in, xy_in)
            d2Y_dx2_in = tf.reshape(tf.linalg.tensor_diag_part(del2Y_jac_in[:, 0, :, 0]), (n_train_in, 1))
            d2Y_dy2_in = tf.reshape(tf.linalg.tensor_diag_part(del2Y_jac_in[:, 1, :, 1]), (n_train_in, 1))
            del2Y_in = [d2Y_dx2_in, d2Y_dy2_in]

            # Compute the estimate of the differential equation at the interior training points.
            G_in = p.differential_equation(xy_in, Y_in, delY_in, del2Y_in)

            # Compute the errors in the computed boundary conditions.
            E_bc = Y_bc - bc

            # Compute the loss functions for the interior training points.
            L_in = tf.math.sqrt(tf.reduce_sum(G_in**2)/n_train_in)

            # Compute the loss functions for the boundary points.
            L_bc = tf.math.sqrt(tf.reduce_sum(E_bc**2)/n_train_bc)

            # Compute the weighted total losses.
            L = w_in*L_in + w_bc*L_bc

        # Save the current losses.
        losses.append(L.numpy())

        # Save the current model weights.
        if save_weights:
            model.save_weights(os.path.join(output_dir, "weights", "weights_%06d" % epoch))

        # Check for convergence.
        if convcheck:
            if epoch > 1:
                loss_delta = losses[-1] - losses[-2]
                if abs(loss_delta) <= tol:
                    converged = True
                    break

        # Compute the gradient of the loss function wrt the network parameters.
        pgrad = tape_param.gradient(L, model.trainable_variables)

        # Update the parameters for this epoch.
        optimizer.apply_gradients(zip(pgrad, model.trainable_variables))

        if verbose and epoch % 1 == 0:
            print("Ending epoch %s, loss function = (%f, %f, %f)" % (epoch, L.numpy(), L_in.numpy(), L_bc.numpy()))

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

    # Save the loss function history.
    if verbose:
        print("Saving loss function history.")
    np.savetxt(os.path.join(output_dir, 'losses.dat'),     np.array(losses))

    # Compute and save the trained results at training points.
    if verbose:
        print("Computing and saving trained results.")
    # with tf.GradientTape(persistent=True) as tape:
    Y_train = model(xy)
    np.savetxt(os.path.join(output_dir, "Y_train.dat"), Y_train.numpy().reshape((n_train,)))

    # Compute and save the trained results at validation points.
    if verbose:
        print("Computing and saving validation results.")
    xy_val, xy_val_in, xy_val_bc = p.create_training_data(nx_val, ny_val)
    n_val = len(xy_val)
    n_val_in = len(xy_val_in)
    n_val_bc = len(xy_val_bc)
    assert n_val_in + n_val_bc == n_val
    np.savetxt(os.path.join(output_dir, "xy_val.dat"), xy_val)
    np.savetxt(os.path.join(output_dir, "xy_val_in.dat"), xy_val_in)
    np.savetxt(os.path.join(output_dir, "xy_val_bc.dat"), xy_val_bc)
    xy_val = tf.Variable(xy_val.reshape(n_val, 2), dtype=args.precision)
    # with tf.GradientTape(persistent=True) as tape:
    Y_val = model(xy_val)
    np.savetxt(os.path.join(output_dir, "Y_val.dat"), Y_val.numpy().reshape((n_val,)))

    # Save the trained model.
    if save_model:
        model.save(os.path.join(output_dir, "model"))


if __name__ == "__main__":
    """Begin main program."""
    main()
