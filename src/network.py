## @package network
#  Package responsible for the neural network definition.
#
#  This package provides functions to define a multi-layer perceptron according to a network configuration file.
#  It also provides separate functions to define the loss and the training algorithm.
import tensorflow as tf
import numpy as np


DTYPE = tf.float32


class MLP:
    def __init__(self, conf):
        self.conf = conf["network"]
        self.layers = [tf.placeholder(shape=(None, self.conf["layers_sizes"][0]), dtype=DTYPE)]
        self.weights = []
        self.biases = []
        for out_size, batch_norm, dropout, activation in zip(self.conf["layers_sizes"][1:], self.conf["batch_norm"], self.conf["dropout"], self.conf["activation_functions"]):
            self.add_layer(self.layers[-1], out_size, batch_norm, dropout, activation)
        self.inp = self.layers[0]
        self.out = self.layers[-1]

    def add_layer(self, inp, out_size, batch_norm, dropout, activation):
        in_size = tf.shape(self.layers[-1])[-1]
        W = ???
        B = None if batch_norm else ???
        dropout = ???
        out = tf.matmul(inp, W) if B is None else tf.matmul(inp, W) + B
        out = tf.keras.batch_normalization(out, ???)
        out = out if activation is None else activation(out)
        self.layers.append(out)
        self.weights.append(W)
        self.biases.append(B)
