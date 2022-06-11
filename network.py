import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from layer import Dense, Conv1D, Conv2D
from activations import activations
from losses import losses
from regularization import regularizations


class Network:

    def __init__(self, config):
        self.loss = losses[config["loss"]]
        self.J_loss = losses["J_" + config["loss"]]
        self.learning_rate = config["learning_rate"]
        self.weight_regularization_rate = config["weight_regularization_rate"]
        self.weight_regularization_function = regularizations[config["weight_regularization_type"]]
        self.J_weight_regularization_function = regularizations["J_" + config["weight_regularization_type"]]

        self.layers = []
        input_size = int(config["image_width"]) ** 2 # Keep track of the size of the input and update during each layer
        for i in range(config["layer_amount"]): # Add one layer object with the given parameters from the config
            if config["layer" + str(i + 1) + "_type"] == "dense":
                if i == 0:
                    input_nodes = input_size
                elif config["layer" + str(i) + "_type"] == "dense":
                    input_nodes = int(config["layer" + str(i) + "_nodes"])
                else:
                    input_nodes = int(config["layer" + str(i) + "_output_filter_amount"]) * input_size
                output_nodes = int(config["layer" + str(i + 1) + "_nodes"])
                activation = config["layer" + str(i + 1) + "_activation"]
                lower_weight_range = float(config["layer" + str(i + 1) + "_lower_weight_range"])
                upper_weight_range = float(config["layer" + str(i + 1) + "_upper_weight_range"])
                self.layers.append(Dense(input_nodes, output_nodes, activation, lower_weight_range, upper_weight_range))

            elif config["layer" + str(i + 1) + "_type"] == "conv1d":
                if i == 0:
                    input_filter_amount = 1
                else:
                    input_filter_amount = int(config["layer" + str(i) + "_output_filter_amount"])
                kernel_size = int(config["layer" + str(i + 1) + "_kernel_size"])
                output_filter_amount = int(config["layer" + str(i + 1) + "_output_filter_amount"])
                stride = int(config["layer" + str(i + 1) + "_stride"])
                mode = config["layer" + str(i + 1) + "_mode"]
                activation = config["layer" + str(i + 1) + "_activation"]
                lower_weight_range = float(config["layer" + str(i + 1) + "_lower_weight_range"])
                upper_weight_range = float(config["layer" + str(i + 1) + "_upper_weight_range"])
                self.layers.append(Conv1D(kernel_size, output_filter_amount, input_filter_amount, stride, mode, activation, lower_weight_range, upper_weight_range))

                if mode == "full":
                    padding = 2
                elif mode == "same":
                    padding = 1
                elif mode == "valid":
                    padding = 0
                input_size = int((input_size - kernel_size + 2 * padding) / stride + 1)

            elif config["layer" + str(i + 1) + "_type"] == "conv2d":
                if i == 0:
                    input_filter_amount = 1
                else:
                    input_filter_amount = int(config["layer" + str(i) + "_output_filter_amount"])
                kernel_size = int(config["layer" + str(i + 1) + "_kernel_size"])
                output_filter_amount = int(config["layer" + str(i + 1) + "_output_filter_amount"])
                stride = int(config["layer" + str(i + 1) + "_stride"])
                mode = config["layer" + str(i + 1) + "_mode"]
                activation = config["layer" + str(i + 1) + "_activation"]
                lower_weight_range = float(config["layer" + str(i + 1) + "_lower_weight_range"])
                upper_weight_range = float(config["layer" + str(i + 1) + "_upper_weight_range"])
                self.layers.append(Conv2D(kernel_size, output_filter_amount, input_filter_amount, stride, mode, activation, lower_weight_range, upper_weight_range))

                if mode == "full":
                    padding = 2
                elif mode == "same":
                    padding = 1
                elif mode == "valid":
                    padding = 0
                input_size = int((int(input_size ** 0.5) - kernel_size + 2 * padding) / stride + 1) ** 2

        # Add one additional activation at the end, typically a softmax function
        self.output_activation = activations[config["output_activation"]]
        self.J_output_activation = activations["J_" + config["output_activation"]]


    # Send the input x through the network by sending it through each of the layers
    # At the end, send it through the final activation function
    def forward_pass(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return self.output_activation(x)


    # Iterate over every training case from the forward minibatch, stored in the columns of the y_pred
    # Compute the initial jacobian R from the loss function
    # Then multiply in the jacobian of the final activation function
    # Iterate over the layers in the backwards direction and multiply in the jacobian from each layer
    # After the backpropagation has finished, update the weights of each layer
    # A notable difference from the forward pass is that the data is now numpy arrays with dimension 1 instead of column-vectors
    # Also, only one case is sent through at a time instead of the whole minibatch
    def backward_pass(self, y_pred, y_true):
        for i in range(y_pred.shape[1]):
            R = self.J_loss(y_pred[:, i], y_true[:, i])
            R = np.dot(R, self.J_output_activation(self.layers[-1].y[:, i]))

            for j in range(len(self.layers) - 1, -1, -1):
                R = self.layers[j].backward(R, i, self.weight_regularization_rate, self.J_weight_regularization_function)
        
        for i in range(len(self.layers)):
            self.layers[i].update_weights(self.learning_rate)
            
    
    # Create a random sample of the minibatch without repetitions
    # Create two empty arrays to store the batch
    # Generate a list of indices from the whole set
    # Then generate batch_size indices and put the case in the minibatch
    # After an index is selected, put it to the back of the list, and subtract 1 from the possible list of indices
    # This makes it so an index can only be chosen once
    def get_minibatch(self, x, y, batch_size):
        x_minibatch = np.zeros((x.shape[0], batch_size))
        y_minibatch = np.zeros((y.shape[0], batch_size))

        available_indices = np.arange(0, x.shape[1])

        for i in range(batch_size):
            index_index = np.random.randint(0, x.shape[1] - i)
            index = available_indices[index_index]
            available_indices[index_index], available_indices[-1 - i] = available_indices[-1 - i], available_indices[index_index]
            x_minibatch[:, i] = x[:, index]
            y_minibatch[:, i] = y[:, index]
            
        return x_minibatch, y_minibatch


    # Fit the training data to the network
    # Compute the train and validation losses for each epoch
    # Compute the test loss at the end
    # Plot the training and validation loss
    # For every epoch, get a random minibatch from the training data
    # Send it through the network with the forward() method
    # Then use the predicted data in the backwards pass
    def fit(self, x_train, y_train, x_val, y_val, x_test, y_test, epochs, batch_size, verbose, visualize_kernels):
        train_losses = np.zeros(epochs)
        val_losses = np.zeros(epochs)

        pbar = tqdm(range(epochs))
        for epoch in pbar:
            x_batch, y_batch = self.get_minibatch(x_train, y_train, batch_size)

            y_pred = self.forward_pass(x_batch)
            train_loss = self.loss(y_pred, y_batch)
            train_losses[epoch] = np.average(train_loss)

            """for layer in self.layers:
                train_loss += self.weight_regularization_rate * self.weight_regularization_function(layer.w)
                train_loss += self.weight_regularization_rate * self.weight_regularization_function(layer.b)"""

            self.backward_pass(y_pred, y_batch)

            y_val_pred = self.forward_pass(x_val)
            val_loss = self.loss(y_val_pred, y_val)
            val_losses[epoch] = np.average(val_loss)

            pbar.set_postfix({"train_loss": train_losses[epoch], "val_loss": val_losses[epoch]})

            if verbose and (epoch % 10) == 0:
                print("Network input:")
                print(x_batch)
                print()
                print("Network output:")
                print(y_pred)
                print()
                print("Target values:")
                print(y_batch)
                print()

        y_test_pred = self.forward_pass(x_test)
        test_loss = self.loss(y_test_pred, y_test)
        test_loss = np.average(test_loss)
        print("Test loss:", test_loss)
        
        plt.plot(np.arange(1, epochs + 1), train_losses, label="Training loss")
        plt.plot(np.arange(1, epochs + 1), val_losses, label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.show()

        if visualize_kernels:
            for layer in self.layers:
                try:
                    layer.visualize_kernels()
                except:
                    pass
