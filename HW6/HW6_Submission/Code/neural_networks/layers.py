"""
Author: Sophia Sanborn, Sagnik Bhattacharya
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas, github.com/sagnibak
"""

import numpy as np
from abc import ABC, abstractmethod

from neural_networks.activations import initialize_activation
from neural_networks.weights import initialize_weights
from collections import OrderedDict

from typing import Callable, List, Literal, Tuple, Union


class Layer(ABC):
    """Abstract class defining the `Layer` interface."""

    def __init__(self):
        self.activation = None

        self.n_in = None
        self.n_out = None

        self.parameters = {}
        self.cache = {}
        self.gradients = {}

        super().__init__()

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    def clear_gradients(self) -> None:
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.gradients.items()}
        )

    def forward_with_param(
        self, param_name: str, X: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val: np.ndarray) -> np.ndarray:
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward

    def _get_parameters(self) -> List[np.ndarray]:
        return [b for a, b in self.parameters.items()]

    def _get_cache(self) -> List[np.ndarray]:
        return [b for a, b in self.cache.items()]

    def _get_gradients(self) -> List[np.ndarray]:
        return [b for a, b in self.gradients.items()]


def initialize_layer(
    name: str,
    activation: str = None,
    weight_init: str = None,
    n_out: int = None,
    kernel_shape: Tuple[int, int] = None,
    stride: int = None,
    pad: int = None,
    mode: str = None,
    keep_dim: str = "first",
) -> Layer:
    """Factory function for layers."""
    if name == "fully_connected":
        return FullyConnected(
            n_out=n_out, activation=activation, weight_init=weight_init,
        )

    elif name == "conv2d":
        return Conv2D(
            n_out=n_out,
            activation=activation,
            kernel_shape=kernel_shape,
            stride=stride,
            pad=pad,
            weight_init=weight_init,
        )

    elif name == "pool2d":
        return Pool2D(kernel_shape=kernel_shape, mode=mode, stride=stride, pad=pad)

    elif name == "flatten":
        return Flatten(keep_dim=keep_dim)

    else:
        raise NotImplementedError("Layer type {} is not implemented".format(name))


class FullyConnected(Layer):
    """A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(
        self, n_out: int, activation: str, weight_init="xavier_uniform"
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)

        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init, activation=activation)

    #GOOD
    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###
        
        W_shape = (self.n_in,) + (self.n_out,)
        W = self.init_weights(W_shape)
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b}) # DO NOT CHANGE THE KEYS
        self.cache = OrderedDict({"Z": [], "X": []}) # cache for backprop
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)}) # parameter gradients initialized to zero
                                                                                     # MUST HAVE THE SAME KEYS AS `self.parameters`
        ### END YOUR CODE ###

    #GOOD
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim)
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)

        ### BEGIN YOUR CODE ###
        Z = np.matmul(X, self.parameters["W"])
        Z = Z + + self.parameters["b"]
        a = self.activation(Z)
        self.cache = {"X" : X, "Z" : Z}
        ### END YOUR CODE ###

        return a

    #GOOD
    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdY  gradient of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        ### BEGIN YOUR CODE ###
        X, Z = self.cache["X"], self.cache["Z"]
        XT = X.T
        a = self.activation.backward(Z, dLdY)
        b = a.dot(self.parameters["W"].T)
        c = np.sum(a, axis=0, keepdims=True)
        d = XT.dot(a)
        self.gradients = {"W" : d, "b" : c}
        ### END YOUR CODE ###

        return b


class Conv2D(Layer):
    """Convolutional layer for inputs with 2 spatial dimensions."""

    def __init__(
        self,
        n_out: int,
        kernel_shape: Tuple[int, int],
        activation: str,
        stride: int = 1,
        pad: str = "same",
        weight_init: str = "xavier_uniform",
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad

        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int, int, int]) -> None:
        """Initialize all layer parameters and determine padding."""
        self.n_in = X_shape[3]

        W_shape = self.kernel_shape + (self.n_in,) + (self.n_out,)
        W = self.init_weights(W_shape)
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b}) # DO NOT CHANGE THE KEYS
        self.cache = OrderedDict({"Z": [], "X": []}) # cache for backprop
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)}) # parameter gradients initialized to zero
                                                                                     # MUST HAVE THE SAME KEYS AS `self.parameters`

        if self.pad == "same":
            self.pad = ((W_shape[0] - 1) // 2, (W_shape[1] - 1) // 2)
        elif self.pad == "valid":
            self.pad = (0, 0)
        elif isinstance(self.pad, int):
            self.pad = (self.pad, self.pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layer. This layer convolves the input
        `X` with a filter of weights, adds a bias term, and applies an activation
        function to compute the output. This layer also supports padding and
        integer strides. Intermediates necessary for the backward pass are stored
        in the cache.

        Parameters
        ----------
        X  input with shape (batch_size, in_rows, in_cols, in_channels)

        Returns
        -------
        output feature maps with shape (batch_size, out_rows, out_cols, out_channels)
        """
        """
        if self.n_in is None:
            self._init_parameters(X.shape)

        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)
        """
        ### BEGIN YOUR CODE ###
        if self.n_in is None:
            self._init_parameters(X.shape)

        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)

        a1 = 2 * self.pad[0]
        a = in_rows - kernel_height + a1
        b1 = self.stride
        o_r = ((a) // b1)
        o_r = o_r + 1

        a2 = 2 * self.pad[1]
        a3 = in_cols - kernel_width + a2
        o_c = (a3) // self.stride
        o_c = o_c + 1

        xp1 = ((0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]), (0, 0))
        xp = np.pad(X, xp1, mode='constant')

        o = np.zeros((n_examples, o_r, o_c, out_channels))

        for i in range(0, o_r):
            for j in range(0, o_c):
                h_s = j * self.stride
                h_e = h_s + kernel_width
                
                v_s = i * self.stride
                v_e = v_s + kernel_height


                X = xp[:, v_s:v_e, h_s:h_e, :]

                o1 = np.tensordot(X, W, axes=([1, 2, 3], [0, 1, 2]))
                o[:, i, j, :] = o1 + b
                Z = o

        self.cache = {"X": X, "Z": Z}

        return self.activation(o)
        ### END YOUR CODE ###  

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for conv layer. Computes the gradients of the output
        with respect to the input feature maps as well as the filter weights and
        biases.

        Parameters
        ----------
        dLdY  gradient of loss with respect to output of this layer
              shape (batch_size, out_rows, out_cols, out_channels)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, in_channels)
        """
        ### BEGIN YOUR CODE ###
        #TODO: DO
        ### END YOUR CODE ###  

class Pool2D(Layer):
    """Pooling layer, implements max and average pooling."""

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        mode: str = "max",
        stride: int = 1,
        pad: Union[int, Literal["same"], Literal["valid"]] = 0,
    ) -> None:

        if type(kernel_shape) == int:
            kernel_shape = (kernel_shape, kernel_shape)

        self.kernel_shape = kernel_shape
        self.stride = stride

        if pad == "same":
            self.pad = ((kernel_shape[0] - 1) // 2, (kernel_shape[1] - 1) // 2)
        elif pad == "valid":
            self.pad = (0, 0)
        elif isinstance(pad, int):
            self.pad = (pad, pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

        self.mode = mode

        if mode == "max":
            self.pool_fn = np.max
            self.arg_pool_fn = np.argmax
        elif mode == "average":
            self.pool_fn = np.mean

        self.cache = {
            "out_rows": [],
            "out_cols": [],
            "X_pad": [],
            "p": [],
            "pool_shape": [],
        }
        self.parameters = {}
        self.gradients = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: use the pooling function to aggregate local information
        in the input. This layer typically reduces the spatial dimensionality of
        the input while keeping the number of feature maps the same.

        As with all other layers, please make sure to cache the appropriate
        information for the backward pass.

        Parameters
        ----------
        X  input array of shape (batch_size, in_rows, in_cols, channels)

        Returns
        -------
        pooled array of shape (batch_size, out_rows, out_cols, channels)
        """
        ### BEGIN YOUR CODE ###
        k_h, k_w = self.kernel_shape
        s_h = self.stride
        s_w = self.stride
        p_h, p_r = self.pad

        b_s, i_c, i_r, i_chan = X.shape

        c1 = ((0, 0), (p_h, p_h), (p_r, p_r), (0, 0))
        xp = np.pad(X, c1, mode='constant')

        a1 = i_c + (2 * p_h) - k_h
        o_c = (a1) // s_h + 1
        
        b1 = i_r + (2 * p_r) - k_w
        o_r = (b1) // s_w + 1

        p = np.zeros((b_s, o_c, o_r, i_chan))

        for i in range(0, b_s):
            for j in range(0, o_c):
                for k in range(0, o_r):
                    for l in range(0, i_chan):
                        h_s = j * s_h
                        h_e = h_s + k_h
                        
                        w_s = k * s_w
                        w_e = w_s + k_w

                        x_s = xp[i, h_s:h_e, w_s:w_e, l]
                        p[i, j, k, l] = self.pool_fn(x_s)

        self.cache['out_rows'] = o_c
        self.cache['out_cols'] = o_r
        self.cache['X_pad'] = xp
        self.cache['pool_shape'] = self.kernel_shape
        return p
        ### END YOUR CODE ###

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for pooling layer.

        Parameters
        ----------
        dLdY  gradient of loss with respect to the output of this layer
              shape (batch_size, out_rows, out_cols, channels)

        Returns
        -------
        gradient of loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, channels)
        """
        #TODO: Do
        ### BEGIN YOUR CODE ###
        # perform a backward pass

        ### END YOUR CODE ###

        
    
class Flatten(Layer):
    """Flatten the input array."""

    def __init__(self, keep_dim: str = "first") -> None:
        super().__init__()

        self.keep_dim = keep_dim
        self._init_params()

    def _init_params(self):
        self.X = []
        self.gradients = {}
        self.parameters = {}
        self.cache = {"in_dims": []}

    def forward(self, X: np.ndarray, retain_derived: bool = True) -> np.ndarray:
        self.cache["in_dims"] = X.shape

        if self.keep_dim == -1:
            return X.flatten().reshape(1, -1)

        rs = (X.shape[0], -1) if self.keep_dim == "first" else (-1, X.shape[-1])
        return X.reshape(*rs)

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        in_dims = self.cache["in_dims"]
        dX = dLdY.reshape(in_dims)
        return dX