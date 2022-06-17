#!/usr/bin/env python

"""Use a set of neural networks to solve a set of coupled 1st-order PDE BVP.

This program will use a set of neural networks to solve a set of coupled
1st-order PDEs as a BVP.

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
description = "Solve a set of coupled 1st-order ODE BVP with the PINN method."

# Default number of training points in the y-dimension.
default_ny_train = 11

# Default number of validation points in the y-dimension.
default_ny_val = 101

# Default problem name.
default_problem = "static"


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
    tol = args.tolerance
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

    # Create the output directory, named after the problem, under the current
    # directory.
    output_dir = os.path.join(".", problem)
    common.create_output_directory(output_dir)

    # Record system information, network parameters, and problem definition.
    if verbose:
        print("Recording system information, model hyperparameters, and "
              "problem definition.")
    common.save_system_information(output_dir)
    save_hyperparameters(args, output_dir)
    common.save_problem_definition(p, output_dir)

    # Create and save the training data.
    if verbose:
        print("Creating and saving training data.")
    xy_train, xy_train_in, xy_train_bc = p.create_training_data(
        nx_train, ny_train
    )
    # Shape is (n_train, p.n_dim)
    np.savetxt(os.path.join(output_dir, "xy_train.dat"), xy_train)
    n_train = len(xy_train)
    # Shape is (n_train_in, p.n_dim)
    np.savetxt(os.path.join(output_dir, "xy_train_in.dat"), xy_train_in)
    n_train_in = len(xy_train_in)
    # Shape is (n_train_bc, p.n_dim)
    np.savetxt(os.path.join(output_dir, "xy_train_bc.dat"), xy_train_bc)
    n_train_bc = len(xy_train_bc)
    assert n_train == n_train_in + n_train_bc

    # Compute the boundary condition values.
    if verbose:
        print("Computing boundary conditions.")
    # This is a pair of 1-D NumPy arrays.
    # bc0 contains the 0th-order (Dirichlet) boundary conditions on the
    # solution.
    # shape (n_train_bc, p.n_var)
    bc0 = p.compute_boundary_conditions(xy_train_bc)
    # Convert to Tensor, shape (n_train_bc, p.n_var).
    bc0 = tf.Variable(bc0, dtype=precision)
    if debug:
        print("bc0 = %s" % bc0)

    # Build the models.
    if verbose:
        print("Creating neural networks.")
    models = []
    for i in range(p.n_var):
        model = common.build_model(n_layers, H, activation)
        models.append(model)
    if debug:
        print("models = %s" % models)

    # Create the optimizer.
    if verbose:
        print("Creating optimizer.")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if debug:
        print("optimizer = %s" % optimizer)

    # Train the models.

    # Create history variables.
    # Loss for each model for interior points.
    losses_model_in = []
    # Loss for each model for boundary condition points.
    losses_model_bc = []
    # Total loss for each model.
    losses_model = []
    # Total loss for all models for interior points.
    losses_in = []
    # Total loss for all models for boundary condition points.
    losses_bc = []
    # Total loss for all models.
    losses = []

    # Set the random number seed for reproducibility.
    tf.random.set_seed(seed)

    # Rename the training data Variables for convenience.
    # The 2-D NumPy arrays must be converted to TensorFlow.
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
        print("Training started at", t_start)

    for epoch in range(max_epochs):

        # Run the forward pass.
        with tf.GradientTape(persistent=True) as tape1:
            with tf.GradientTape(persistent=True) as tape0:

                # Compute the network outputs at the interior training points.
                # N_in is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list.
                # Each Tensor has shape (n_train_in, 1).
                N_in = [model(xy_in) for model in models]

                # Compute the network outputs at the boundary training points.
                # N_bc is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list.
                # Each Tensor has shape (n_train_bc, 1).
                N_bc = [model(xy_bc) for model in models]

            # Compute the gradients of the network outputs wrt inputs at the
            # interior training points.
            # delN_in is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape (n_train_in, p.n_dim).
            delN_in = [tape0.gradient(N, xy_in) for N in N_in]

            # Compute the estimates of the differential equations at the
            # interior training points.
            # G_in is a list of Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape (n_in, 1).
            G_in = [
                pde(xy_in, N_in, delN_in) for pde in p.differential_equations
            ]

            # Compute the loss function for the interior points for each
            # model, based on the values of the differential equations.
            # Lm_in is a list of Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape ().
            Lm_in = [tf.math.sqrt(tf.reduce_sum(G**2)/n_train_in) for G in G_in]

            # Compute the errors for the boundary points.
            # E_bc is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape (n_train_bc, 1).
            E_bc = []
            for i in range(p.n_var):
                E = N_bc[i] - tf.reshape(bc0[:, i], (n_train_bc, 1))
                E_bc.append(E)

            # Compute the loss functions for the boundary points for each
            # model.
            # Lm_bc is a list of Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape ().
            Lm_bc = [tf.math.sqrt(tf.reduce_sum(E**2)/n_train_bc) for E in E_bc]

            # Compute the total losses for each model.
            # Lm is a list of Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape ().
            Lm = [loss_in + loss_bc for (loss_in, loss_bc) in zip(Lm_in, Lm_bc)]

            # Compute the total loss for interior points for the model
            # collection.
            # Tensor shape ()
            L_in = tf.math.reduce_sum(Lm_in)

            # Compute the total loss for boundary points for the model
            # collection.
            # Tensor shape ()
            L_bc = tf.math.reduce_sum(Lm_bc)

            # Compute the total loss for all points for the model
            # collection.
            # Tensor shape ()
            L = L_in + L_bc

        # Save the current losses.
        # The per-model loss histories are lists of lists of Tensors.
        # Each sub-list has length p.n_var.
        # Each Tensor is shape ().
        losses_model_in.append(Lm_in)
        losses_model_bc.append(Lm_bc)
        losses_model.append(Lm)
        # The total loss histories are lists of scalars.
        losses_in.append(L_in.numpy())
        losses_bc.append(L_bc.numpy())
        losses.append(L.numpy())

        # Check for convergence.
        if epoch > 1:
            loss_delta = losses[-1] - losses[-2]
            if abs(loss_delta) <= tol:
                converged = True
                break

        # Compute the gradient of the loss function wrt the network parameters.
        # pgrad is a list of lists of Tensor objects.
        # There are p.n_var sub-lists in the top-level list.
        # There are 3 Tensors in each sub-list, with shapes:
        # Input weights: (H, p.n_dim)
        # Input biases: (H,)
        # Output weights: (H, 1)
        # Each Tensor is shaped based on model.trainable_variables.
        pgrad = [tape1.gradient(L, model.trainable_variables) for model in models]

        # Update the parameters for this epoch.
        for (g, m) in zip(pgrad, models):
            optimizer.apply_gradients(zip(g, m.trainable_variables))

        if verbose and epoch % 1 == 0:
            print("Ending epoch %s, loss function = %f" % (epoch, L.numpy()))
            # print("Ending epoch %s." % epoch)

    # Count the last epoch.
    n_epochs = epoch + 1

    t_stop = datetime.datetime.now()
    t_elapsed = t_stop - t_start
    if verbose:
        print("Training stopped at", t_stop)
        print("Total training time was %s seconds." % t_elapsed.total_seconds())
        print("Epochs: %d" % n_epochs)
        # print("Final value of loss function: %f" % losses[-1])
        print("converged = %s" % converged)

    # Convert the loss function histories to numpy arrays.
    losses_model_in = np.array(losses_model_in)
    losses_model_bc = np.array(losses_model_bc)
    losses_model = np.array(losses_model)
    losses_in = np.array(losses_in)
    losses_bc = np.array(losses_bc)
    losses = np.array(losses)

    # Save the loss function histories.
    if verbose:
        print("Saving loss function histories.")
    np.savetxt(os.path.join(output_dir, 'losses_model_in.dat'), np.array(losses_model_in))
    np.savetxt(os.path.join(output_dir, 'losses_model_bc.dat'), np.array(losses_model_bc))
    np.savetxt(os.path.join(output_dir, 'losses_model.dat'), np.array(losses_model))
    np.savetxt(os.path.join(output_dir, 'losses_in.dat'), np.array(losses_in))
    np.savetxt(os.path.join(output_dir, 'losses_bc.dat'), np.array(losses_bc))
    np.savetxt(os.path.join(output_dir, 'losses.dat'), np.array(losses))

    # Compute and save the trained results at training points.
    n_var = len(p.variable_names)
    if verbose:
        print("Computing and saving trained results.")
    with tf.GradientTape(persistent=True) as tape0:
        N_train = [model(xy) for model in models]
    delN_train = [tape0.gradient(N_train, xy) for N in N_train]
    for i in range(n_var):
        np.savetxt(os.path.join(output_dir, "%s_train.dat" % p.variable_names[i]),
                   N_train[i].numpy().reshape(n_train,))
        np.savetxt(os.path.join(output_dir, "del_%s_train.dat" % p.variable_names[i]),
                  delN_train[i].numpy().reshape(n_train, 2))

if __name__ == "__main__":
    """Begin main program."""
    main()
