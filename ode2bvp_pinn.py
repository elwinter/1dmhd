#!/usr/bin/env python

"""Use a neural network to solve a 2nd-order ODE BVP.

This program will use a neural network to solve a 2nd-order ordinary
differential equation boundary value problem.

This code assumes the problem has the solution value specified at the first
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

# Import TensorFlow.
import tensorflow as tf

# Import project modules.
import common


# Program constants

# Program description.
description = "Solve a 2nd-order ODE BVP using the PINN method."

# Default problem name.
default_problem = "lagaris03bvp"

# Name of file to hold the system information.
system_information_file = "system_information.txt"

# Name of file to hold the network hyperparameters, as an importable Python
# module.
hyperparameter_file = "hyperparameters.py"

# Initial parameter ranges
w0_range = [-0.1, 0.1]  # Hidden layer weights
u0_range = [-0.1, 0.1]  # Hidden layer biases
v0_range = [-0.1, 0.1]  # Output layer weights


# Program global variables.

# Global object to hold the imported problem definition module.
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
        f.write("precision = %s\n" % repr(args.precision))
        f.write("random_seed = %s\n" % repr(args.seed))
        f.write("tolerance = %s\n" % repr(args.tolerance))
        f.write("w_bc = %s\n" % repr(args.w_bc))
        f.write("w0_range = %s\n" % repr(w0_range))
        f.write("u0_range = %s\n" % repr(u0_range))
        f.write("v0_range = %s\n" % repr(v0_range))


def build_model(n_layers, H, activation):
    """Build a multi-layer neural network model.

    Build a fully-connected, multi-layer neural network with single output.
    Each layer will have H hidden nodes. Each hidden node has weights and
    a bias, and uses the specified activation function.

    The number of inputs is determined when the network is first used.

    Parameters
    ----------
    n_layers : int
        Number of hidden layers to create.
    H : int
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
    tf.keras.backend.set_floatx(args.precision)

    # Import the problem to solve.
    global p
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
        print("Recording system information, model hyperparameters, and problem definition.")
    common.save_system_information(output_dir)
    save_hyperparameters(args, output_dir)
    common.save_problem_definition(p, output_dir)

    # Create and save the training data.
    if verbose:
        print("Creating and saving training data.")
    # These are each 1-D NumPy arrays.
    x_train, x_train_in, x_train_bc = p.create_training_data(nx_train)
    np.savetxt(os.path.join(output_dir, "x_train.dat"), x_train)
    n_train = len(x_train)
    np.savetxt(os.path.join(output_dir, "x_train_in.dat"), x_train_in)
    n_train_in = len(x_train_in)
    np.savetxt(os.path.join(output_dir, "x_train_bc.dat"), x_train_bc)
    n_train_bc = len(x_train_bc)
    assert n_train_bc == 2
    assert n_train == n_train_in + n_train_bc

    # Compute the boundary condition values.
    if verbose:
        print("Computing boundary conditions.")
    # This is a 1-D NumPy array, of length 2.
    bc = p.compute_boundary_conditions(x_train_bc)
    # Reshape to a 2-D NumPy array, shape (2, 1).
    bc = bc.reshape((n_train_bc, 1))
    # Convert to a Tensor, shape (2, 1).
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
    model = build_model(n_layers, H, activation)
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
    # The 1-D NumPy arrays must be reshaped to shape (n_train, 1) since
    # TensorFlow needs these variables to have 2 dimensions.
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
        print("Training started at %s." % t_start)

    for epoch in range(max_epochs):

        # Run the forward pass.
        with tf.GradientTape(persistent=True) as tape_param:
            with tf.GradientTape(persistent=True) as tape2:
                with tf.GradientTape(persistent=True) as tape1:

                    # Compute the network outputs within the domain.
                    # Shape is (n_in, 1).
                    y_in = model(x_in)

                    # Compute the network output on the boundary.
                    # Shape is (n_bc, 1) == (2, 1).
                    y_bc = model(x_bc)

                # Compute the first derivatives at the interior training
                # points.
                # Shape is (n_in, 1).
                dy_dx_in = tape1.gradient(y_in, x_in)

            # Compute the second derivatives at the interior training points.
            #  Shape is (n_in, 1).
            d2y_dx2_in = tape2.gradient(dy_dx_in, x_in)

            # Compute the estimate of the differential equation at the
            # interior training points.
            # Shape is (n_in, 1).
            G_in = p.differential_equation(x_in, y_in, dy_dx_in, d2y_dx2_in)

            # Compute the errors in the computed boundary conditions.
            # Shape is (n_bc, 1) == (2, 1).
            E_bc = y_bc - bc

            # Compute the loss function for the interior training points.
            # Shape is () (scalar).
            L_in = tf.math.sqrt(tf.reduce_sum(G_in**2)/n_train_in)

            # Compute the loss function for the boundary points.
            # Shape is () (scalar).
            L_bc = tf.math.sqrt(tf.reduce_sum(E_bc**2)/n_train_bc)

            # Compute the weighted total loss.
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
            y_train = model(x)
        # Shape (n_train, 1)
        dy_dx_train = tape1.gradient(y_train, x)
    # Shape (n_train, 1)
    d2y_dx2_train = tape2.gradient(dy_dx_train, x)
    # Reshape the 2-D Tensor objects to 1-D NumPy arrays.
    np.savetxt(os.path.join(output_dir, "y_train.dat"), y_train.numpy().reshape((n_train,)))
    np.savetxt(os.path.join(output_dir, "dy_dx_train.dat"), dy_dx_train.numpy().reshape((n_train,)))
    np.savetxt(os.path.join(output_dir, "d2y_dx2_train.dat"), d2y_dx2_train.numpy().reshape((n_train,)))

    # Compute and save the trained results at validation points.
    if verbose:
        print("Computing and saving validation results.")
    # These are NumPy arrays, not Tensor objects.
    # Shapes (n_val,), (n_val_in), (n_val_bc,) == (1,).
    x_val, x_val_in, x_val_bc = p.create_training_data(nx_val)
    n_val = len(x_val)
    n_val_in = len(x_val_in)
    n_val_bc = len(x_val_bc)  # Should be 1.
    assert n_val_in + n_val_bc == n_val
    np.savetxt(os.path.join(output_dir, "x_val.dat"), x_val)
    np.savetxt(os.path.join(output_dir, "x_val_in.dat"), x_val_in)
    np.savetxt(os.path.join(output_dir, "x_val_bc.dat"), x_val_bc)
    # Reshape to 2-D Tensor, shape (n_val, 1).
    x_val = tf.Variable(x_val.reshape(n_val, 1), dtype=precision)
    with tf.GradientTape(persistent=True) as tape2:
        with tf.GradientTape(persistent=True) as tape1:
            # Shape (n_val, 1)
            y_val = model(x_val)
        # Shape (n_val, 1)
        dy_dx_val = tape1.gradient(y_val, x_val)
    d2y_dx2_val = tape2.gradient(dy_dx_val, x_val)
    np.savetxt(os.path.join(output_dir, "y_val.dat"), y_val.numpy().reshape((n_val,)))
    np.savetxt(os.path.join(output_dir, "dy_dx_val.dat"), dy_dx_val.numpy().reshape((n_val,)))
    np.savetxt(os.path.join(output_dir, "d2y_dx2_val.dat"), d2y_dx2_val.numpy().reshape((n_val,)))

    # Save the trained model.
    if save_model:
        model.save(os.path.join(output_dir, "model"))


if __name__ == "__main__":
    """Begin main program."""
    main()
