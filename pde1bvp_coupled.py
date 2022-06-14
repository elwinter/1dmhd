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
description = "Solve the a set of coupled 1st-order ODE BVP with the PINN method."

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
    xy_train, xy_train_in, xy_train_bc = p.create_training_data(
        nx_train, ny_train
    )
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
    # This is a pair of 1-D NumPy arrays.
    # bc0 contains boundary conditions on the solution.
    # shape (n_train_bc, p.n_var)
    bc0 = p.compute_boundary_conditions(xy_train_bc)
    # Convert to Tensors, shape (n_train_bc, 1).
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
                N_in = [model(xy_in) for model in models]
                # N is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list.
                # Each Tensor has shape (n_train_in, 1).

                # Compute the network outputs at the boundary training points.
                N_bc = [model(xy_bc) for model in models]
                # N is a list of tf.Tensor objects.
                # There are p.n_var Tensors in the list.
                # Each Tensor has shape (n_train_bc, 1).

            # Compute the gradients of the network outputs wrt inputs at the
            # interior training points.
            # Shape is (n_in, p.n_var).
            delN_in = [tape0.gradient(N, xy_in) for N in N_in]
            # del is a list of tf.Tensor objects.
            # There are p.n_var Tensors in the list.
            # Each Tensor has shape (n_train_in, 2).

            # Compute the estimates of the differential equations at the
            # interior training points.
            # Shape is (n_in, p.n_var).
            G_in = [
                pde(xy_in, N_in, delN_in) for pde in p.differential_equations
            ]
#             del_rho = tape0.gradient(rho, xt)
#             del_vx  = tape0.gradient(vx,  xt)
#             del_vy  = tape0.gradient(vy,  xt)
#             del_vz  = tape0.gradient(vz,  xt)
#             del_By  = tape0.gradient(By,  xt)
#             del_Bz  = tape0.gradient(Bz,  xt)
#             del_P   = tape0.gradient(P,   xt)

#             # Compute the estimates of the differential equations.
#             Y = [rho, vx, vy, vz, By, Bz, P]
#             del_Y = [del_rho, del_vx, del_vy, del_vz, del_By, del_Bz, del_P]
#             G_rho = pde_rho(xt, Y, del_Y)
#             G_vx  =  pde_vx(xt, Y, del_Y)
#             G_vy  =  pde_vy(xt, Y, del_Y)
#             G_vz  =  pde_vz(xt, Y, del_Y)
#             G_By  =  pde_By(xt, Y, del_Y)
#             G_Bz  =  pde_Bz(xt, Y, del_Y)
#             G_P   =   pde_P(xt, Y, del_Y)

#             # Compute the loss functions.
#             L_rho = tf.math.sqrt(tf.reduce_sum(G_rho**2)/n_train)
#             L_vx  = tf.math.sqrt(tf.reduce_sum(G_vx**2) /n_train)
#             L_vy  = tf.math.sqrt(tf.reduce_sum(G_vy**2) /n_train)
#             L_vz  = tf.math.sqrt(tf.reduce_sum(G_vz**2) /n_train)
#             L_By  = tf.math.sqrt(tf.reduce_sum(G_By**2) /n_train)
#             L_Bz  = tf.math.sqrt(tf.reduce_sum(G_Bz**2) /n_train)
#             L_P   = tf.math.sqrt(tf.reduce_sum(G_P**2)  /n_train)
#             L = L_rho + L_vx + L_vy + L_vz + L_By + L_Bz + L_P

#         # Save the current losses.
#         losses_rho.append(L_rho.numpy())
#         losses_vx.append( L_vx.numpy())
#         losses_vy.append( L_vy.numpy())
#         losses_vz.append( L_vz.numpy())
#         losses_By.append( L_By.numpy())
#         losses_Bz.append( L_Bz.numpy())
#         losses_P.append(  L_P.numpy())
#         losses.append(    L.numpy())

# #         # Check for convergence.
# #         # if epoch > 1:
# #         #     loss_delta = losses[-1] - losses[-2]
# #         #     if abs(loss_delta) <= tol:
# #         #         converged = True
# #         #         break

#         # Compute the gradient of the loss function wrt the network parameters.
#         pgrad_rho = tape1.gradient(L, model_rho.trainable_variables)
#         pgrad_vx  = tape1.gradient(L,  model_vx.trainable_variables)
#         pgrad_vy  = tape1.gradient(L,  model_vy.trainable_variables)
#         pgrad_vz  = tape1.gradient(L,  model_vz.trainable_variables)
#         pgrad_By  = tape1.gradient(L,  model_By.trainable_variables)
#         pgrad_Bz  = tape1.gradient(L,  model_Bz.trainable_variables)
#         pgrad_P   = tape1.gradient(L,   model_P.trainable_variables)

#         # Update the parameters for this epoch.
#         optimizer.apply_gradients(zip(pgrad_rho, model_rho.trainable_variables))
#         optimizer.apply_gradients(zip( pgrad_vx, model_vx.trainable_variables))
#         optimizer.apply_gradients(zip( pgrad_vy, model_vy.trainable_variables))
#         optimizer.apply_gradients(zip( pgrad_vz, model_vz.trainable_variables))
#         optimizer.apply_gradients(zip( pgrad_By, model_By.trainable_variables))
#         optimizer.apply_gradients(zip( pgrad_Bz, model_Bz.trainable_variables))
#         optimizer.apply_gradients( zip( pgrad_P, model_P.trainable_variables))

        if verbose and epoch % 1 == 0:
            # print("Ending epoch %s, loss function = %f" % (epoch, L.numpy()))
            print("Ending epoch %s." % epoch)

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

#     # Save the loss function histories.
#     if verbose:
#         print("Saving loss function histories.")
#     np.savetxt(os.path.join(output_dir, 'losses_rho.dat'), np.array(losses_rho))
#     np.savetxt(os.path.join(output_dir, 'losses_vx.dat'),  np.array(losses_vx))
#     np.savetxt(os.path.join(output_dir, 'losses_vy.dat'),  np.array(losses_vy))
#     np.savetxt(os.path.join(output_dir, 'losses_vz.dat'),  np.array(losses_vz))
#     np.savetxt(os.path.join(output_dir, 'losses_By.dat'),  np.array(losses_By))
#     np.savetxt(os.path.join(output_dir, 'losses_Bz.dat'),  np.array(losses_Bz))
#     np.savetxt(os.path.join(output_dir, 'losses_P.dat'),   np.array(losses_P))
#     np.savetxt(os.path.join(output_dir, 'losses.dat'),     np.array(losses))

#     # Compute and save the trained results at training points.
#     if verbose:
#         print("Computing and saving trained results.")
#     with tf.GradientTape(persistent=True) as tape:

#         # Compute the network outputs at the training points.
#         N_rho = model_rho(xt)
#         N_vx  = model_vx( xt)
#         N_vy  = model_vy( xt)
#         N_vz  = model_vz( xt)
#         N_By  = model_By( xt)
#         N_Bz  = model_Bz( xt)
#         N_P   = model_P(  xt)

#         # Compute the trial solutions.
#         rho_train = Ytrial_rho(xt, N_rho)
#         vx_train  = Ytrial_vx( xt, N_vx)
#         vy_train  = Ytrial_vy( xt, N_vy)
#         vz_train  = Ytrial_vz( xt, N_vz)
#         By_train  = Ytrial_By( xt, N_By)
#         Bz_train  = Ytrial_Bz( xt, N_Bz)
#         P_train   = Ytrial_P(  xt, N_P)

#     np.savetxt(os.path.join(output_dir, "rho_train.dat"), rho_train.numpy().reshape((n_train,)))
#     np.savetxt(os.path.join(output_dir, "vx_train.dat"),   vx_train.numpy().reshape((n_train,)))
#     np.savetxt(os.path.join(output_dir, "vy_train.dat"),   vy_train.numpy().reshape((n_train,)))
#     np.savetxt(os.path.join(output_dir, "vz_train.dat"),   vz_train.numpy().reshape((n_train,)))
#     np.savetxt(os.path.join(output_dir, "By_train.dat"),   By_train.numpy().reshape((n_train,)))
#     np.savetxt(os.path.join(output_dir, "Bz_train.dat"),   Bz_train.numpy().reshape((n_train,)))
#     np.savetxt(os.path.join(output_dir, "P_train.dat"),     P_train.numpy().reshape((n_train,)))


if __name__ == "__main__":
    """Begin main program."""
    main()
