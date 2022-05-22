#!/usr/bin/env python

"""Use a neural network to solve a 1st-order PDE BVP.

This program will use a neural network to solve a 1st-order partial
differential equation boundary value problem.

This code uses the PINN method.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard Python modules.
import datetime
from importlib import import_module
import os

# Import 3rd-party modules.
import numpy as np
import tensorflow as tf

# Import project modules.
import common


# Program constants

# Program description.
description = "Solve a 1st-order PDE BVP using the PINN method."

# Default number of training points in the y-dimension.
default_ny_train = 11

# Default number of validation points in the y-dimension.
default_ny_val = 101

# Default problem name.
default_problem = "transport"

# Name of hyperparameter record file, as an importable Python module.
hyperparameter_file = "hyperparameters.py"


# Program global variables.

# Global object to hold the problem definition.
p = None


def create_command_line_argument_parser():
    """Create the command-line argument parser.

    Create the command-line argument parser.

    Parameters
    ----------
    None

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser for command-line arguments.
    """
    # Create the common command-line parser.
    parser = common.create_common_command_line_argument_parser(
        description, default_problem
    )

    # Add program-specific command-line arguments.
    parser.add_argument(
        "--ny_train", type=int, default=default_ny_train,
        help="Number of equally-spaced training points in y dimension (default: %(default)s)"
    )
    parser.add_argument(
        "--ny_val", type=int, default=default_ny_val,
        help="Number of equally-spaced validation points in y dimension (default: %(default)s)"
    )
    return parser


def save_hyperparameters(args, output_dir):
    """Save the neural network hyperparameters.
    
    Print a record of the hyperparameters of the neural networks in the
    specified directory, as an importable python module.

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
        f.write("activation = %s\n" % repr(args.activation))
        f.write("convcheck = %s\n" % repr(args.convcheck))
        f.write("learning_rate = %s\n" % repr(args.learning_rate))
        f.write("max_epochs = %s\n" % repr(args.max_epochs))
        f.write("H = %s\n" % repr(args.n_hid))
        f.write("n_layers = %s\n" % repr(args.n_layers))
        f.write("nx_train = %s\n" % repr(args.nx_train))
        f.write("nx_val = %s\n" % repr(args.nx_val))
        f.write("ny_train = %s\n" % repr(args.ny_train))
        f.write("ny_val = %s\n" % repr(args.ny_val))
        f.write("precision = %s\n" % repr(args.precision))
        f.write("random_seed = %s\n" % repr(args.seed))
        f.write("tolerance = %s\n" % repr(args.tolerance))
        f.write("w_bc = %s\n" % repr(args.w_bc))


def main():
    """Begin main program."""

    # Set up the command-line parser.
    parser = create_command_line_argument_parser()

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
    nx_val = args.nx_val
    ny_train = args.ny_train
    ny_val = args.ny_val
    precision = args.precision
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
    tf.keras.backend.set_floatx(precision)

    # Import the problem to solve.
    global p
    if verbose:
        print("Importing definition for problem '%s'." % problem)
    p = import_module(problem)
    if debug:
        print("p = %s" % p)

    # Set up the output directory under the current directory.
    output_dir = os.path.join(".", problem)
    common.create_output_directory(output_dir)

    # Record system information, network parameters, and problem definition.
    if verbose:
        print("Recording system information, model hyperparameters, and problem definition.")
    common.save_system_information(output_dir)
    save_hyperparameters(args, output_dir)
    common.save_problem_definition(p, output_dir)

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
    bc = tf.Variable(bc, dtype=precision)

    # Compute the weight for the interior points.
    w_in = 1.0 - w_bc

    # Build the model.
    if verbose:
        print("Creating neural network.")
    model = common.build_model(n_layers, H, activation)

    # Create the optimizer.
    if verbose:
        print("Creating optimizer.")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Train the model.

    # Create history variables.
    losses = []

    # Set the random number seed for reproducibility.
    tf.random.set_seed(seed)

    # Rename the training data Variables for convenience.
    xy_train_var = tf.Variable(xy_train, dtype=precision)
    xy = xy_train_var
    xy_train_in_var = tf.Variable(xy_train_in, dtype=precision)
    xy_in = xy_train_in_var
    xy_train_bc_var = tf.Variable(xy_train_bc, dtype=precision)
    xy_bc = xy_train_bc_var

    # Clear the convergence flag to start.
    converged = False

    t_start = datetime.datetime.now()
    if verbose:
        print("Training started at %s, max_epochs = %s" % (t_start, max_epochs))

    for epoch in range(max_epochs):

        # Run the forward pass.
        with tf.GradientTape(persistent=True) as tape_param:
            with tf.GradientTape(persistent=True) as tape1:

                # Compute the network outputs within the domain.
                Y_in = model(xy_in)

                # Compute the network outputs on the boundaries.
                Y_bc = model(xy_bc)

            # Compute the first derivatives at the interior training
            # points.
            delY_in = tape1.gradient(Y_in, xy_in)

            # Compute the estimate of the differential equation at the interior
            # training points.
            G_in = p.differential_equation(xy_in, Y_in, delY_in)

            # Compute the errors in the computed boundary conditions.
            E_bc = Y_bc - bc

            # Compute the loss function for the interior training points.
            L_in = tf.math.sqrt(tf.reduce_sum(G_in**2)/n_train_in)

            # Compute the loss function for the boundary points.
            L_bc = tf.math.sqrt(tf.reduce_sum(E_bc**2)/n_train_bc)

            # Compute the weighted total loss function.
            L = w_in*L_in + w_bc*L_bc

        # Save the current loss.
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
    xy_val = tf.Variable(xy_val.reshape(n_val, 2), dtype=precision)
    # with tf.GradientTape(persistent=True) as tape:
    Y_val = model(xy_val)
    np.savetxt(os.path.join(output_dir, "Y_val.dat"), Y_val.numpy().reshape((n_val,)))

    # Save the trained model.
    if save_model:
        model.save(os.path.join(output_dir, "model"))


if __name__ == "__main__":
    """Begin main program."""
    main()
