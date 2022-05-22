#!/usr/bin/env python

"""Use a neural network to solve a 2nd-order PDE BVP.

This program will use a neural network to solve a 2nd-order partial
differential equation boundary value problem.

This code assumes the problem has the solution values specified at the first
and last values of the independent variable.

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

# Import project-specific modules.
import common


# Program constants

# Program description.
description = "Solve a 2nd-order PDE BVP using the PINN method."

# Default number of training points in the y-dimension.
default_ny_train = 11

# Default number of validation points in the y-dimension.
default_ny_val = 101

# Default problem name.
default_problem = "lagaris05"


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
    parser = common.create_command_line_argument_parser(
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
    path = common.save_hyperparameters(args, output_dir)
    with open(path, "a") as f:
        f.write("ny_train = %s\n" % repr(args.ny_train))
        f.write("ny_val = %s\n" % repr(args.ny_val))


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
    tolerance = args.tolerance
    verbose = args.verbose
    w_bc = args.w_bc
    if debug:
        print("args = %s" % args)

    # Set the backend TensorFlow precision.
    tf.keras.backend.set_floatx(precision)

    # Import the problem to solve.
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
    # These are each 2-D NumPy arrays.
    # Shapes are (n_train, 2), (n_train_in, 2), (n_train_bc, 2)
    xy_train, xy_train_in, xy_train_bc = p.create_training_data(nx_train, ny_train)
    np.savetxt(os.path.join(output_dir, "xy_train.dat"), xy_train)
    n_train = len(xy_train)
    np.savetxt(os.path.join(output_dir, "xy_train_in.dat"), xy_train_in)
    n_train_in = len(xy_train_in)
    np.savetxt(os.path.join(output_dir, "xy_train_bc.dat"), xy_train_bc)
    n_train_bc = len(xy_train_bc)
    assert n_train_bc == 2*(nx_train + ny_train - 2)
    assert n_train == n_train_in + n_train_bc

    # Compute the boundary condition values.
    if verbose:
        print("Computing boundary conditions.")
    # This is a 1-D NumPy array, of length n_train_bc.
    bc = p.compute_boundary_conditions(xy_train_bc)
    # Reshape to a 2-D NumPy array, shape (n_train_bc, 1).
    bc = bc.reshape((n_train_bc, 1))
    # Convert to a Tensor, shape (n_train_bc, 1).
    bc = tf.Variable(bc, dtype=precision)
    if debug:
        print("bc = %s" % bc)

    # Compute the normalized weight for the interior points.
    w_in = 1.0 - w_bc
    if debug:
        print("w_in = %s" % w_in)

    # Build the model.
    if verbose:
        print("Creating neural network.")
    model = common.build_model(n_layers, H, activation)
    if debug:
        print("model = %s" % model)

    # Create the optimizer.
    if verbose:
        print("Creating optimizer.")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if debug:
        print("optimizer = %s" % optimizer)

    # Train the model.

    # Create the loss function histories.
    losses = []
    losses_in = []
    losses_bc = []

    # Set the random number seed for reproducibility.
    tf.random.set_seed(seed)

    # Rename the training data Variables for convenience.
    # The 2-D NumPy arrays are converted to 2-D Variables.
    # Shape (n_train, 2)
    xy_train_var = tf.Variable(xy_train, dtype=precision)
    xy = xy_train_var
    # Shape (n_train_in, 2)
    xy_train_in_var = tf.Variable(xy_train_in, dtype=precision)
    xy_in = xy_train_in_var
    # Shape (n_train_bc, 2)
    xy_train_bc_var = tf.Variable(xy_train_bc, dtype=precision)
    xy_bc = xy_train_bc_var

    # Clear the convergence flag to start.
    converged = False

    t_start = datetime.datetime.now()
    if verbose:
        print("Training started at %s." % t_start)

    for epoch in range(max_epochs):

        # Run the forward pass.
        with tf.GradientTape(persistent=True) as tape_param:
            with tf.GradientTape(persistent=True) as tape2:
                with tf.GradientTape(persistent=True) as tape1:

                    # Compute the network outputs within the domain.
                    # Shape is (n_in, 1).
                    Y_in = model(xy_in)

                    # Compute the network outputs on the boundaries.
                    # Shape is (n_bc, 1).
                    Y_bc = model(xy_bc)

                # Compute the first derivatives at the interior training
                # points.
                # Shape is (n_in, 2).
                delY_in = tape1.gradient(Y_in, xy_in)

            # Compute the second derivatives at the interior training points.
            # Shape is (n_in, 1, n_in, 2).
            del2Y_jac_in = tape2.jacobian(delY_in, xy_in)
            # Shape is (n_in, 1).
            d2Y_dx2_in = tf.reshape(tf.linalg.tensor_diag_part(del2Y_jac_in[:, 0, :, 0]), (n_train_in, 1))
            # Shape is (n_in, 1).
            d2Y_dy2_in = tf.reshape(tf.linalg.tensor_diag_part(del2Y_jac_in[:, 1, :, 1]), (n_train_in, 1))
            # Shape is (n_in, 2).
            del2Y_in = tf.stack([d2Y_dx2_in[:, 0], d2Y_dy2_in[:, 0]], axis=1)

            # Compute the estimate of the differential equation at the
            # interior training points.
            # Shape is (n_in, 1).
            G_in = p.differential_equation(xy_in, Y_in, delY_in, del2Y_in)

            # Compute the errors in the computed boundary conditions.
            # Shape is (n_bc, 1).
            E_bc = Y_bc - bc

            # Compute the loss functions for the interior training points.
            # Shape is () (scalar).
            L_in = tf.math.sqrt(tf.reduce_sum(G_in**2)/n_train_in)

            # Compute the loss functions for the boundary points.
            # Shape is () (scalar).
            L_bc = tf.math.sqrt(tf.reduce_sum(E_bc**2)/n_train_bc)

            # Compute the weighted total losses.
            # Shape is () (scalar).
            L = w_in*L_in + w_bc*L_bc

        # Save the current losses.
        losses.append(L.numpy())
        losses_in.append(L_in.numpy())
        losses_bc.append(L_bc.numpy())

        # Save the current model weights.
        if save_weights:
            model.save_weights(os.path.join(output_dir, "weights", "weights_%06d" % epoch))

        # Check for convergence.
        if convcheck:
            if epoch > 1:
                loss_delta = losses[-1] - losses[-2]
                if abs(loss_delta) <= tolerance:
                    converged = True
                    break

        # Compute the gradient of the loss function wrt the network parameters.
        # The gradient object is a list of 3 or more Tensor objects. The shapes
        # of these Tensor objects are the same as the shapes of the elements of
        # the model.trainable_variables. More specifically:
        #   * The first Tensor is for the gradient of the loss function wrt the
        #     weights of the first hidden layer, and has shape (1, H).
        #   * The second Tensor is for the gradient of the loss function wrt
        #     the biases of the first hidden layer, and has shape (H,).
        #   * If n_layers > 1, each subsequent pair of Tensor objects
        #     represents the gradients of the loss function with respect to the
        #     weights and biases of the next hidden layer, with layer index
        #     increasing from the first hidden layer above the input layer, to
        #     the last hidden layer below the output layer. The weight gradient
        #     Tensor objects are shape (H, H), and the bias gradient Tensor
        #     objects are shape (H,).
        #   * The last Tensor is for the gradient of the loss function wrt the
        #     weights of the output layer, and has shape (H, 1).
        pgrad = tape_param.gradient(L, model.trainable_variables)

        # Update the parameters for this epoch.
        optimizer.apply_gradients(zip(pgrad, model.trainable_variables))

        if verbose and epoch % 1 == 0:
            print("Ending epoch %s, (L, L_in, L_bc) = (%f, %f, %f)" %
                  (epoch, L.numpy(), L_in.numpy(), L_bc.numpy()))

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
    np.savetxt(os.path.join(output_dir, 'losses.dat'), np.array(losses))
    np.savetxt(os.path.join(output_dir, 'losses_in.dat'), np.array(losses_in))
    np.savetxt(os.path.join(output_dir, 'losses_bc.dat'), np.array(losses_bc))

    # Compute and save the trained results at training points.
    if verbose:
        print("Computing and saving trained results.")
    with tf.GradientTape(persistent=True) as tape2:
        with tf.GradientTape(persistent=True) as tape1:
            # Shape (n_train, 1)
            Y_train = model(xy)
        # Shape (n_train, 2)
        delY_train = tape1.gradient(Y_train, xy)
    # Shape is (n_train, 1, n_train, 2).
    del2Y_jac_train = tape2.jacobian(delY_train, xy)
    # Shape is (n_train, 1).
    d2Y_dx2_train = tf.reshape(tf.linalg.tensor_diag_part(del2Y_jac_train[:, 0, :, 0]), (n_train, 1))
    # Shape is (n_train, 1).
    d2Y_dy2_train = tf.reshape(tf.linalg.tensor_diag_part(del2Y_jac_train[:, 1, :, 1]), (n_train, 1))
    # Shape is (n_train, 2).
    del2Y_train = tf.stack([d2Y_dx2_train[:, 0], d2Y_dy2_train[:, 0]], axis=1)
    np.savetxt(os.path.join(output_dir, "Y_train.dat"), Y_train.numpy().reshape((n_train,)))
    np.savetxt(os.path.join(output_dir, "delY_train.dat"), delY_train.numpy().reshape((n_train, 2)))
    np.savetxt(os.path.join(output_dir, "del2Y_train.dat"), del2Y_train.numpy().reshape((n_train, 2)))

    # Compute and save the trained results at validation points.
    if verbose:
        print("Computing and saving validation results.")
    # Shapes are (n_val, 2), (n_val_in, 2), (n_val_bc, 2).
    xy_val, xy_val_in, xy_val_bc = p.create_training_data(nx_val, ny_val)
    n_val = len(xy_val)
    n_val_in = len(xy_val_in)
    n_val_bc = len(xy_val_bc)
    assert n_val_bc == 2*(nx_val + ny_val - 2)
    assert n_val_in + n_val_bc == n_val
    np.savetxt(os.path.join(output_dir, "xy_val.dat"), xy_val)
    np.savetxt(os.path.join(output_dir, "xy_val_in.dat"), xy_val_in)
    np.savetxt(os.path.join(output_dir, "xy_val_bc.dat"), xy_val_bc)
    xy_val = tf.Variable(xy_val.reshape(n_val, 2), dtype=precision)
    with tf.GradientTape(persistent=True) as tape2:
        with tf.GradientTape(persistent=True) as tape1:
            # Shape (n_val, 1)
            Y_val = model(xy_val)
        # Shape (n_val, 2)
        delY_val = tape1.gradient(Y_val, xy_val)
    # Shape is (n_val, 1, n_val, 2).
    del2Y_jac_val = tape2.jacobian(delY_val, xy_val)
    # Shape is (n_val, 1).
    d2Y_dx2_val = tf.reshape(tf.linalg.tensor_diag_part(del2Y_jac_val[:, 0, :, 0]), (n_val, 1))
    # Shape is (n_val, 1).
    d2Y_dy2_val = tf.reshape(tf.linalg.tensor_diag_part(del2Y_jac_val[:, 1, :, 1]), (n_val, 1))
    # Shape is (n_val, 2).
    del2Y_val = tf.stack([d2Y_dx2_val[:, 0], d2Y_dy2_val[:, 0]], axis=1)
    np.savetxt(os.path.join(output_dir, "Y_val.dat"), Y_val.numpy().reshape((n_val,)))
    np.savetxt(os.path.join(output_dir, "delY_val.dat"), delY_val.numpy().reshape((n_val, 2)))
    np.savetxt(os.path.join(output_dir, "del2Y_val.dat"), del2Y_val.numpy().reshape((n_val, 2)))

    # Save the trained model.
    if save_model:
        model.save(os.path.join(output_dir, "model"))


if __name__ == "__main__":
    """Begin main program."""
    main()
