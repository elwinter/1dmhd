#!/usr/bin/env python

"""Use a pair of neural networks to solve 2 coupled ODE IVP.

This program will use a pair of neural networks to solve 2 coupled
1st-order ODE IVP.

This code uses the PINN method.

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard Python modules.
import datetime
from importlib import import_module
import os
import platform
import shutil
import sys

# Import 3rd-party modules.
import numpy as np

# Import TensorFlow.
import tensorflow as tf

# Import project modules.
import common


# Program constants

# Program description.
description = "Solve 2 coupled 1st-order ODE IVP, using the PINN method."


# Default problem name.
default_problem = "lagaris04"

# Name of system information file.
system_information_file = "system_information.txt"

# Name of hyperparameter record file, as an importable Python module.
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
    except:
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
        f.write("Python implementation: %s\n" % platform.python_implementation())
        f.write("Python file: %s\n" % __file__)
        f.write("NumPy version: %s\n" % np.__version__)
        f.write("TensorFlow version: %s\n" % tf.__version__)


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
    shutil.copy(p.__file__, output_dir)


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
    create_output_directory(output_dir)

    # Record system information, network parameters, and problem definition.
    if verbose:
        print("Recording system information, model hyperparameters, and problem definition.")
    save_system_information(output_dir)
    save_hyperparameters(args, output_dir)
    save_problem_definition(p, output_dir)

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
    assert n_train_bc == 1
    assert n_train == n_train_in + n_train_bc

    # Compute the initial condition values.
    if verbose:
        print("Computing initial conditions.")
    # This is a 1-D NumPy array, of length 1.
    bc1 = p.compute_boundary_conditions_1(x_train_bc)
    bc2 = p.compute_boundary_conditions_2(x_train_bc)
    # Reshape to a 2-D NumPy array, shape (1, 1).
    bc1 = bc1.reshape((n_train_bc, 1))
    bc2 = bc2.reshape((n_train_bc, 1))
    # Convert to a Tensor, shape (1, 1).
    bc1 = tf.Variable(bc1, dtype=precision)
    bc2 = tf.Variable(bc2, dtype=precision)
    if debug:
        print("bc1 = %s" % bc1)
        print("bc2 = %s" % bc2)

    # Compute the normalized weight for the interior points.
    w_in = 1.0 - w_bc
    if debug:
        print("w_in = %s" % w_in)

    # Build the models.
    if verbose:
        print("Creating neural networks.")
    model1 = build_model(n_layers, H, activation)
    model2 = build_model(n_layers, H, activation)
    if debug:
        print("model1 = %s" % model1)
        print("model2 = %s" % model2)

    # Create the optimizer.
    if verbose:
        print("Creating optimizer.")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if debug:
        print("optimizer = %s" % optimizer)

    # Train the models.

    # Create the loss function histories.
    losses = []
    losses_1 = []
    losses_2 = []
    losses_in = []
    losses_in_1 = []
    losses_in_2 = []
    losses_bc = []
    losses_bc_1 = []
    losses_bc_2 = []

    # Set the random number seed for reproducibility.
    tf.random.set_seed(seed)

    # Rename the training data Variables for convenience.
    x_train_var = tf.Variable(x_train.reshape(n_train, 1), dtype=args.precision)
    # The 1-D NumPy arrays must be reshaped to shape (n_train, 1) since
    # TensorFlow needs these variables to have 2 dimensions.
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
        with tf.GradientTape(persistent=True) as tape1:
            with tf.GradientTape(persistent=True) as tape0:

                # Compute the network outputs within the domain.
                # Shape is (n_in, 1).
                y1_in = model1(x_in)
                y2_in = model2(x_in)

                # Compute the network outputs on the boundaries.
                # Shape is (n_bc, 1) == (1, 1).
                y1_bc = model1(x_bc)
                y2_bc = model2(x_bc)

            # Compute the gradients of the network outputs wrt inputs at the
            # interior training points.
            # Shape is (n_in, 1).
            dy1_dx_in = tape0.gradient(y1_in, x_in)
            dy2_dx_in = tape0.gradient(y2_in, x_in)

            # Compute the estimates of the differential equations at the
            # interior training points.
            # Shape is (n_in, 1).
            y_in = [y1_in, y2_in]
            dy_dx_in = [dy1_dx_in, dy2_dx_in]
            G_y1_in = p.differential_equation_1(x_in, y_in, dy_dx_in)
            G_y2_in = p.differential_equation_2(x_in, y_in, dy_dx_in)

            # Compute the errors in the computed initial condition.
            # Shape is (n_bc, 1) == (1, 1).
            E_y1_bc = y1_bc - bc1
            E_y2_bc = y2_bc - bc2

            # Compute the loss functions for the interior training points.
            # Shape is () (scalar).
            L_y1_in = tf.math.sqrt(tf.reduce_sum(G_y1_in**2)/n_train_in)
            L_y2_in = tf.math.sqrt(tf.reduce_sum(G_y2_in**2)/n_train_in)
            L_in = L_y1_in + L_y2_in

            # Compute the loss functions for the initial points.
            # Shape is () (scalar).
            L_y1_bc = tf.math.sqrt(tf.reduce_sum(E_y1_bc**2)/n_train_bc)
            L_y2_bc = tf.math.sqrt(tf.reduce_sum(E_y2_bc**2)/n_train_bc)
            L_bc = L_y1_bc + L_y2_bc

            # Compute the weighted total losses.
            # Shape is () (scalar).
            L_y1 = w_in*L_y1_in + w_bc*L_y1_bc
            L_y2 = w_in*L_y2_in + w_bc*L_y2_bc
            L = w_in*L_in + w_bc*L_bc

        # Save the current losses.
        losses.append(L.numpy())
        losses_1.append(L_y1.numpy())
        losses_2.append(L_y2.numpy())
        losses_in.append(L_in.numpy())
        losses_in_1.append(L_y1_in.numpy())
        losses_in_2.append(L_y2_in.numpy())
        losses_bc.append(L_bc.numpy())
        losses_bc_1.append(L_y1_bc.numpy())
        losses_bc_2.append(L_y2_bc.numpy())

        # Save the current model weights.
        if save_weights:
            model1.save_weights(os.path.join(output_dir, "weights1", "weights_%06d" % epoch))
            model2.save_weights(os.path.join(output_dir, "weights2", "weights_%06d" % epoch))

        # Check for convergence.
        if convcheck:
            if epoch > 1:
                loss_delta = losses[-1] - losses[-2]
                if abs(loss_delta) <= tolerance:
                    converged = True
                    break

        # Compute the gradients of the loss function wrt the network parameters.
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
        pgrad_y1 = tape1.gradient(L, model1.trainable_variables)
        pgrad_y2 = tape1.gradient(L, model2.trainable_variables)

        # Update the parameters for this epoch.
        optimizer.apply_gradients(zip(pgrad_y1, model1.trainable_variables))
        optimizer.apply_gradients(zip(pgrad_y2, model2.trainable_variables))

        if verbose and epoch % 1 == 0:
            print("Ending epoch %s, (L, L_in, L_bc, L_y1, L_y1_in, L_y1_bc, L_y2, L_y2_in, L_y2_bc) = (%f, %f, %f, %f, %f, %f, %f, %f, %f)" %
                  (epoch, L.numpy(), L_in.numpy(), L_bc.numpy(), L_y1.numpy(), L_y1_in.numpy(), L_y1_bc.numpy(), L_y2.numpy(), L_y2_in.numpy(), L_y2_bc.numpy()))

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
    np.savetxt(os.path.join(output_dir, 'losses_1.dat'), np.array(losses_1))
    np.savetxt(os.path.join(output_dir, 'losses_2.dat'), np.array(losses_2))
    np.savetxt(os.path.join(output_dir, 'losses_in.dat'), np.array(losses_in))
    np.savetxt(os.path.join(output_dir, 'losses_in_1.dat'), np.array(losses_in_1))
    np.savetxt(os.path.join(output_dir, 'losses_in_2.dat'), np.array(losses_in_2))
    np.savetxt(os.path.join(output_dir, 'losses_bc.dat'), np.array(losses_bc))
    np.savetxt(os.path.join(output_dir, 'losses_bc_1.dat'), np.array(losses_bc_1))
    np.savetxt(os.path.join(output_dir, 'losses_bc_2.dat'), np.array(losses_bc_2))

    # Compute and save the trained results at training points.
    if verbose:
        print("Computing and saving trained results.")
    with tf.GradientTape(persistent=True) as tape:
        # Shape (n_train, 1)
        y1_train = model1(x)
        y2_train = model2(x)
    # Shape (n_train, 1)
    dy1_dx_train = tape.gradient(y1_train, x)
    dy2_dx_train = tape.gradient(y2_train, x)
    # Reshape the 2-D Tensor objects to 1-D NumPy arrays.
    np.savetxt(os.path.join(output_dir, "y1_train.dat"), y1_train.numpy().reshape((n_train,)))
    np.savetxt(os.path.join(output_dir, "y2_train.dat"), y2_train.numpy().reshape((n_train,)))
    np.savetxt(os.path.join(output_dir, "dy1_dx_train.dat"), dy1_dx_train.numpy().reshape((n_train,)))
    np.savetxt(os.path.join(output_dir, "dy2_dx_train.dat"), dy2_dx_train.numpy().reshape((n_train,)))

    # Compute and save the trained results at validation points.
    if verbose:
        print("Computing and saving validation results.")
    # Shapes (n_val,), (n_val_in), (n_val_bc,) == (1,).
    x_val, x_val_in, x_val_bc = p.create_training_data(nx_val)
    n_val = len(x_val)
    n_val_in = len(x_val_in)
    n_val_bc = len(x_val_bc)
    assert n_val_in + n_val_bc == n_val
    np.savetxt(os.path.join(output_dir, "x_val.dat"), x_val)
    np.savetxt(os.path.join(output_dir, "x_val_in.dat"), x_val_in)
    np.savetxt(os.path.join(output_dir, "x_val_bc.dat"), x_val_bc)
    # Reshape to 2-D Tensor, shape (n_val, 1).
    x_val = tf.Variable(x_val.reshape(n_val, 1), dtype=args.precision)
    with tf.GradientTape(persistent=True) as tape:
        # Shape (n_val, 1)
        y1_val = model1(x_val)
        y2_val = model2(x_val)
    # Shape (n_val, 1)
    dy1_dx_val = tape.gradient(y1_val, x_val)
    dy2_dx_val = tape.gradient(y2_val, x_val)
    np.savetxt(os.path.join(output_dir, "y1_val.dat"), y1_val.numpy().reshape((n_val,)))
    np.savetxt(os.path.join(output_dir, "y2_val.dat"), y2_val.numpy().reshape((n_val,)))
    np.savetxt(os.path.join(output_dir, "dy1_dx_val.dat"), dy1_dx_val.numpy().reshape((n_val,)))
    np.savetxt(os.path.join(output_dir, "dy2_dx_val.dat"), dy2_dx_val.numpy().reshape((n_val,)))

    # Save the trained model.
    if save_model:
        model1.save(os.path.join(output_dir, "model1"))
        model2.save(os.path.join(output_dir, "model2"))


if __name__ == "__main__":
    """Begin main program."""
    main()
