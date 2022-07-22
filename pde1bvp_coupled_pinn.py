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
description = "Solve a set of coupled 1st-order PDE BVP using the PINN method."

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

    # Set up the output directory under the current directory.
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
    # These are each 2-D NumPy arrays.
    # Shapes are (n_train, 2), (n_train_in, 2), (n_train_bc, 2)
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

    # Compute the boundary condition values in normalized dimensionless form.
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

    # Compute the normalized weight for the interior points.
    w_all = 1.0 - w_bc
    if debug:
        print("w_all = %s" % w_all)

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
    losses_model_all = []
    # Loss for each model for boundary condition points.
    losses_model_bc = []
    # Total loss for each model.
    losses_model = []
    # Total loss for all models for interior points.
    losses_all = []
    # Total loss for all models for boundary condition points.
    losses_bc = []
    # Total loss for all models.
    losses = []

    # Set the random number seed for reproducibility.
    tf.random.set_seed(seed)

    # Rename the training data Variables for convenience.
    # The 2-D NumPy arrays must be converted to TensorFlow.
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
        print("Training started at", t_start)

    for epoch in range(max_epochs):

        # Run the forward pass.
        # tape0 is for computing gradients wrt network parameters.
        # tape1 is for computing 1st-order derivatives of outputs wrt inputs.
        with tf.GradientTape(persistent=True) as tape0:
            with tf.GradientTape(persistent=True) as tape1:

                # Compute the network outputs at all training points.
                # N_all is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list.
                # Each Tensor has shape (n_train, 1).
                N_all = [model(xy) for model in models]

                # Compute the network outputs at the boundary training points.
                # N_bc is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list.
                # Each Tensor has shape (n_train_bc, 1).
                N_bc = [model(xy_bc) for model in models]

            # Compute the gradients of the network outputs wrt inputs at all
            # training points.
            # delN_all is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape (n_train, p.n_dim).
            delN_all = [tape1.gradient(N, xy) for N in N_all]

            # Compute the estimates of the differential equations at all
            # training points.
            # G_all is a list of Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape (n_train, 1).
            G_all = [
                pde(xy, N_all, delN_all) for pde in p.differential_equations
            ]

            # Compute the loss function for all points for each
            # model, based on the values of the differential equations.
            # Lm_all is a list of Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape ().
            Lm_all = [tf.math.sqrt(tf.reduce_sum(G**2)/n_train) for G in G_all]

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
            Lm = [w_all*loss_all + w_bc*loss_bc for (loss_all, loss_bc) in zip(Lm_all, Lm_bc)]

            # Compute the total loss for all points for the model
            # collection.
            # Tensor shape ()
            L_all = tf.math.reduce_sum(Lm_all)

            # Compute the total loss for boundary points for the model
            # collection.
            # Tensor shape ()
            L_bc = tf.math.reduce_sum(Lm_bc)

            # Compute the total loss for all points for the model
            # collection.
            # Tensor shape ()
            L = w_all*L_all + w_bc*L_bc

        # Save the current losses.
        # The per-model loss histories are lists of lists of Tensors.
        # Each sub-list has length p.n_var.
        # Each Tensor is shape ().
        losses_model_all.append(Lm_all)
        losses_model_bc.append(Lm_bc)
        losses_model.append(Lm)
        # The total loss histories are lists of scalars.
        losses_all.append(L_all.numpy())
        losses_bc.append(L_bc.numpy())
        losses.append(L.numpy())

        # Save the current model weights.
        if save_weights:
            for (i, model) in enumerate(models):
                model.save_weights(
                    os.path.join(output_dir, "weights_" + p.variable_names[i],
                                 "weights_%06d" % epoch)
                )

        # Check for convergence.
        if convcheck:
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
        pgrad = [tape0.gradient(L, model.trainable_variables) for model in models]

        # Update the parameters for this epoch.
        for (g, m) in zip(pgrad, models):
            optimizer.apply_gradients(zip(g, m.trainable_variables))

        if verbose and epoch % 1 == 0:
            print("Ending epoch %s, (L, L_all, L_bc) = (%f, %f, %f)" %
                  (epoch, L.numpy(), L_all.numpy(), L_bc.numpy()))

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

    # Convert the loss function histories to numpy arrays.
    losses_model_all = np.array(losses_model_all)
    losses_model_bc = np.array(losses_model_bc)
    losses_model = np.array(losses_model)
    losses_all = np.array(losses_all)
    losses_bc = np.array(losses_bc)
    losses = np.array(losses)

    # Save the loss function histories.
    if verbose:
        print("Saving loss function histories.")
    np.savetxt(os.path.join(output_dir, 'losses_model_all.dat'), losses_model_all)
    np.savetxt(os.path.join(output_dir, 'losses_model_bc.dat'), losses_model_bc)
    np.savetxt(os.path.join(output_dir, 'losses_model.dat'), losses_model)
    np.savetxt(os.path.join(output_dir, 'losses_all.dat'), losses_all)
    np.savetxt(os.path.join(output_dir, 'losses_bc.dat'), losses_bc)
    np.savetxt(os.path.join(output_dir, 'losses.dat'), losses)

    # Compute and save the trained results at training points.
    if verbose:
        print("Computing and saving trained results.")
    # Shapes are ???
    with tf.GradientTape(persistent=True) as tape1:
        N_train = [model(xy) for model in models]
    delN_train = [tape1.gradient(N, xy) for N in N_train]
    for i in range(p.n_var):
        np.savetxt(os.path.join(output_dir, "%s_train.dat" %
                   p.variable_names[i]), tf.reshape(N_train[i], (n_train,)))
        np.savetxt(os.path.join(output_dir, "del_%s_train.dat" %
                   p.variable_names[i]), delN_train[i])

    # Compute and save the trained results at validation points.
    # if verbose:
    #     print("Computing and saving validation results.")
    # Shapes are (n_val, 2), (n_val_in, 2), (n_val_bc, 2).
    # Shapes are (n_val, 2), (n_val_in, 2), (n_val_bc, 2).
    # xy_val, xy_val_in, xy_val_bc = p.create_training_data(nx_val, ny_val)
    # n_val = len(xy_val)
    # n_val_in = len(xy_val_in)
    # n_val_bc = len(xy_val_bc)
    # assert n_val_bc == 2*(nx_val + ny_val - 2)
    # assert n_val_in + n_val_bc == n_val
    # np.savetxt(os.path.join(output_dir, "xy_val.dat"), xy_val)
    # np.savetxt(os.path.join(output_dir, "xy_val_in.dat"), xy_val_in)
    # np.savetxt(os.path.join(output_dir, "xy_val_bc.dat"), xy_val_bc)
    # xy_val = tf.Variable(xy_val.reshape(n_val, 2), dtype=precision)
    # with tf.GradientTape(persistent=True) as tape1:
    #     N_val = [model(xy_val) for model in models]
    # delN_val = [tape1.gradient(N, xy_val) for N in N_val]
    # for i in range(p.n_var):
    #     np.savetxt(os.path.join(output_dir, "%s_val.dat" %
    #                p.variable_names[i]), tf.reshape(N_val[i], (n_val,)))
    #     np.savetxt(os.path.join(output_dir, "del_%s_val.dat" %
    #                p.variable_names[i]), delN_val[i])

    # Save the trained models.
    if save_model:
        for (i, model) in enumerate(models):
            model.save(os.path.join(output_dir, "model_" + p.variable_names[i]))

if __name__ == "__main__":
    """Begin main program."""
    main()
