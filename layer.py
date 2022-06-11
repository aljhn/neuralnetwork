import numpy as np
import matplotlib.pyplot as plt
from activations import activations


# Code taken from: https://matplotlib.org/stable/gallery/specialty_plots/hinton_demo.html
def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size, facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


class Dense:

    def __init__(self, input_nodes, output_nodes, activation, lower_weight_range, upper_weight_range):
        self.w = np.random.uniform(lower_weight_range, upper_weight_range, (input_nodes, output_nodes))
        self.b = np.random.uniform(lower_weight_range, upper_weight_range, (output_nodes, 1))

        self.delta_w = np.zeros(self.w.shape)
        self.delta_b = np.zeros(self.b.shape)

        self.activation = activations[activation]
        self.J_activation = activations["J_" + activation]
        

    # Run the minibatch x through the layer
    # x is a matrix where each column is a data case
    # Compute z by multiplying w.T by x, transpose because datacases are columns vectors and add the bias b
    def forward(self, x):
        if x.ndim >= 3:
            x.shape = (self.w.shape[0], x.shape[-1])
        self.x = x
        self.z  = np.dot(self.w.T, self.x) + self.b # numpy auto-broadcasts on addition
        self.y = self.activation(self.z)
        return self.y

    
    # z = w * x
    # y = f(z)
    # R = J_L_y
    def backward(self, R, i, weight_regularization_rate, weight_regularization_function):
        J_y_z = self.J_activation(self.z[:, i])

        # Compute the jacobian from the loss function to the weights
        # The R matrix is the jacobian from to the loss function to the output of the current layer
        J_y_w = np.outer(self.x[:, i], np.diag(J_y_z)) # 
        J_L_w = R * J_y_w # Performs the multiplcation needed to make the dimensions work out, found in the lecture slides
        self.delta_w += J_L_w + weight_regularization_rate * weight_regularization_function(self.w)

        # Then compute the jacobian from the loss function to the bias
        J_L_b = np.dot(R, J_y_z) # z differentiated with b is the identity matrix, does not need to add to the equation
        J_L_b.shape = (J_L_b.shape[0], 1) # Reshape the vector into a matrix with one column to make the dimensions correct
        self.delta_b += J_L_b + weight_regularization_rate * weight_regularization_function(self.b)
        
        # Finally compute the jacobian from the loss function to the input of the network
        # This is also the same as to the output of the previous layer, so is the next R
        J_y_x = np.dot(J_y_z, self.w.T)
        R = np.dot(R, J_y_x)
        return R

    
    # Add the deltas multiplied with the learning rate to the weights and biases of the network
    # Called after the backward pass has finished computing the deltas
    # Manually set the deltas to zero afterwards
    def update_weights(self, learning_rate):
        self.w -= learning_rate * self.delta_w
        self.b -= learning_rate * self.delta_b

        self.delta_w = np.zeros(self.w.shape)
        self.delta_b = np.zeros(self.b.shape)


class Conv1D:

    def __init__(self, kernel_size, output_filter_amount, input_filter_amount, stride, mode, activation, lower_weight_range, upper_weight_range):
        self.kernel_size = kernel_size
        self.output_filter_amount = output_filter_amount
        self.input_filter_amount = input_filter_amount
        self.filters = np.random.uniform(lower_weight_range, upper_weight_range, (output_filter_amount, input_filter_amount, kernel_size))
        self.delta = np.zeros_like(self.filters)
        
        self.stride = stride
        if mode == "full":
            self.padding = 2
        elif mode == "same":
            self.padding = 1
        elif mode == "valid":
            self.padding = 0

        self.activation = activations[activation]
        self.J_activation = activations["J_" + activation]


    # Run the minibatch x through the layer
    # The output z is a 3-dimensional numpy array
    # The dimensions corresponds to the filters, values and data cases in that order
    # The output is computed by doing convolution between each filter and the input (actually cross-correlation, but they are equivalent for this purpose)
    def forward(self, x):
        if x.ndim == 2:
            x.shape = (1, x.shape[0], x.shape[1])
        elif x.ndim == 4:
            x.shape = (x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        self.x = x

        # The output size is based on: input image size, kernel size, padding (mode) and the stride
        output_size = int((x.shape[1] - self.kernel_size + 2 * self.padding) / self.stride + 1)
        self.z = np.zeros((self.output_filter_amount, output_size, x.shape[2]))
        
        for i in range(x.shape[2]):
            for j in range(output_size):
                for k in range(self.kernel_size):
                    index = j * self.stride - self.padding + k
                    if index < 0 or index >= x.shape[1]:
                        continue
                    self.z[:, j, i] += np.sum(self.filters[:, :, k] * self.x[:, index, i], axis=1)

        self.y = self.activation(self.z)
        return self.y

    # Convolution gradient equations found here: https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509
    def backward(self, R, i, weight_regularization_rate, weight_regularization_function):
        # Start by finding the jacobian from the loss function to the outputs from the convolution by multiplying with the activation jacobian
        # Assume that R is always a 1 dimensional vector
        # Then reshape J_L_z to fit with the convolution channels
        J_y_z = self.J_activation(self.z[:, :, i].flatten())
        J_L_z = np.dot(R, J_y_z)
        J_L_z.shape = (self.output_filter_amount, -1)

        # Convolve self.x[:, :, i] and J_L_z to get the delta
        for j in range(self.output_filter_amount):
            for k in range(self.kernel_size):
                for l in range(self.x[:, :, i].shape[1]):
                    index = l * self.stride - self.padding + k
                    if index < 0 or index >= J_L_z.shape[1]:
                        continue
                    self.delta[j, :, k] += self.x[:, l, i] * J_L_z[j, index]

        # Convolve J_L_z and the filters to get the jacobian to the previous layer
        J_L_x = np.zeros_like(self.x[:, :, i])
        for j in range(self.input_filter_amount):
            for k in range(J_L_x.shape[1]):
                for l in range(self.kernel_size):
                    index = l * self.stride - self.padding - 1 + k
                    if index < 0 or index >= J_L_z.shape[1]:
                        continue
                    J_L_x[j, k] += np.sum(J_L_z[:, index] * self.filters[:, j, self.kernel_size - 1 - l])

        return J_L_x.flatten()

    
    # Update the kernels after the backward pass has finished for every case in the minibatch
    # Add the deltas multiplied with the learning rate
    # Finally reset the deltas to zero
    def update_weights(self, learning_rate):
        self.filters -= learning_rate * self.delta
        self.delta = np.zeros(self.filters.shape)
    

    # Visualize kernels with the hinton function
    # One plot is every kernel for that output channel
    # Each kernels is a column in the plot
    def visualize_kernels(self):
        for i in range(self.output_filter_amount):
            hinton(self.filters[i, :, :])
            plt.show()


class Conv2D:

    def __init__(self, kernel_size, output_filter_amount, input_filter_amount, stride, mode, activation, lower_weight_range, upper_weight_range):
        self.kernel_size = kernel_size
        self.output_filter_amount = output_filter_amount
        self.input_filter_amount = input_filter_amount
        self.filters = np.random.uniform(lower_weight_range, upper_weight_range, (output_filter_amount, input_filter_amount, kernel_size, kernel_size))
        self.delta = np.zeros(self.filters.shape)
        
        self.stride = stride
        if mode == "full":
            self.padding = 2
        elif mode == "same":
            self.padding = 1
        elif mode == "valid":
            self.padding = 0

        self.activation = activations[activation]
        self.J_activation = activations["J_" + activation]


    # Same as Conv1D, but with an extra dimension
    def forward(self, x):
        if x.ndim == 2:
            x.shape = (1, int(x.shape[0] ** 0.5), int(x.shape[0] ** 0.5), x.shape[1])
        elif x.ndim == 3:
            x.shape = (x.shape[0], int(x.shape[1] ** 0.5), int(x.shape[1] ** 0.5), x.shape[2])
        self.x = x

        output_size = int((x.shape[1] - self.kernel_size + 2 * self.padding) / self.stride + 1)
        self.z = np.zeros((self.output_filter_amount, output_size, output_size, x.shape[3]))
        
        for i in range(x.shape[3]):
            for j in range(output_size):
                for k in range(output_size):
                    s = np.zeros(self.output_filter_amount)
                    for l in range(self.kernel_size):
                        for m in range(self.kernel_size):
                            index1 = j * self.stride - self.padding + l
                            index2 = k * self.stride - self.padding + m
                            if index1 < 0 or index1 >= x.shape[1] or index2 < 0 or index2 >= x.shape[2]:
                                continue
                            s += np.sum(self.filters[:, :, l, m] * self.x[:, index1, index2, i], axis=1)
                    self.z[:, j, k, i] = s

        self.y = self.activation(self.z)
        return self.y


    # Same as Conv1D, but with an extra dimension
    def backward(self, R, i, weight_regularization_rate, weight_regularization_function):
        J_y_z = self.J_activation(self.z[:, :, :, i].flatten())
        J_L_z = np.dot(R, J_y_z)
        size = int((J_L_z.shape[0] / self.output_filter_amount) ** 0.5)
        J_L_z.shape = (self.output_filter_amount, size, size)

        # Convolve self.x[:, :, i] and J_L_z to get the delta
        for j in range(self.output_filter_amount):
            for k in range(self.kernel_size):
                for l in range(self.kernel_size):
                    for m in range(self.x[:, :, :, i].shape[1]):
                        for n in range(self.x[:, :, :, i].shape[2]):
                            index1 = m * self.stride - self.padding + k
                            index2 = n * self.stride - self.padding + l
                            if index1 < 0 or index1 >= J_L_z.shape[1] or index2 < 0 or index2 >= J_L_z.shape[2]:
                                continue
                            self.delta[j, :, k, l] += self.x[:, m, n, i] * J_L_z[j, index1, index2]

        # Convolve J_L_z and the filters to get the jacobian to the previous layer
        J_L_x = np.zeros_like(self.x[:, :, :, i])
        for j in range(self.input_filter_amount):
            for k in range(J_L_x.shape[1]):
                for l in range(J_L_x.shape[2]):
                    for m in range(self.kernel_size):
                        for n in range(self.kernel_size):
                            index1 = m * self.stride - self.padding - 1 + k
                            index1 = n * self.stride - self.padding - 1 + l
                            if index1 < 0 or index1 >= J_L_z.shape[1] or index2 < 0 or index2 >= J_L_z.shape[2]:
                                continue
                            J_L_x[j, k, l] += np.sum(J_L_z[:, index1, index2] * self.filters[:, j, self.kernel_size - 1 - m, self.kernel_size - 1 - n])

        return J_L_x.flatten()

    
    # Update the kernels after the backward pass has finished for every case in the minibatch
    # Add the deltas multiplied with the learning rate
    # Finally reset the deltas to zero
    def update_weights(self, learning_rate):
        self.filters -= learning_rate * self.delta
        self.delta = np.zeros(self.filters.shape)


    # Visualize kernels with the hinton function
    # One plot is one kernel
    def visualize_kernels(self):
        for i in range(self.output_filter_amount):
            for j in range(self.input_filter_amount):
                hinton(self.filters[i, j, :, :])
                plt.show()
