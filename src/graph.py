## @package graph
#  Package responsible for the tensorflow graph definition.
#
#  This package provides functions to define a multi-layer perceptron according to the arguments passed.
#  It also provides separate functions to define the loss and the training algorithm.
import tensorflow as tf
import numpy as np
import logging as log


logger = log.getLogger("classifier")


DTYPE = tf.float32


## Class for a Multilayer perceptron
# defines and store the layers of the network
# implements batch normalization and dropout
class MLP:
    ## Constructor
    #  @param args arguments passed to the command line
    def __init__(self, args):
        logger.debug("Defining multilayer perceptron")
        self.layers_dims = np.array(args.layers_dims)
        self.batch_norm = np.array(args.batch_norms)
        self.activations = np.array(args.activations)
        self.dropout_probs = np.array(args.dropout_probs)
        assert(len(self.batch_norm) ==
               len(self.activations) ==
               len(self.dropout_probs) ==
               self.layers_dims - 1)
        self.n_layers = len(self.batch_norm)
        self._net_constructed_once = False
        self._define_variables()

    def _define_variables(self):
        self.weights = []
        self.biases = []
        for in_size, out_size, batch_norm in zip(self.layers_dims, self.layers_dims[1:], self.batch_norm):
            random = tf.truncated_normal(stddev=0.01, shape=(in_size, out_size))
            W = tf.Variable(random)
            B = None if batch_norm else tf.Variable(tf.zeros(shape=(1, out_size)))
            self.weights.append(W)
            self.biases.append(B)

    def __call__(self, features, training=False):
        out = features
        for i in range(self.n_layers):
            out = self._make_layer(out, i, training)
        self._net_constructed_once = True
        return out

    ## Private function for adding layers to the network
    # @param inp input tensor
    # @param out_size size of the new layer
    # @param batch_norm bool stating if batch normalization should be used
    # @param dropout droupout probability. Set to 0 to disable
    # @param activation activation function
    def _make_layer(self, inp, layer_number, training):
        out_size = self.layers_dims[layer_number + 1]
        batch_norm = self.batch_norm[layer_number]
        dropout = self.dropout_probs[layer_number]
        activation = self.activations[layer_number]
        _to_format = [out_size, batch_norm, dropout, activation]
        layer_name = "layer_{}".format(layer_number + 1) if self._net_constructed_once else None
        logger.debug("Creating new layer:")
        string = "Output size = {}\tBatch norm = {}\tDropout prob = {}\tActivation = {}"
        logger.debug(string.format(*_to_format))
        W = self.weights[layer_number]
        B = self.biases[layer_number]
        out = tf.matmul(inp, W) if B is None else tf.matmul(inp, W) + B
        out = tf.layers.batch_normalization(out, training=training, reuse=layer_name)
        out = out if activation is None else activation(out)
        out = tf.layers.dropout(out, rate=dropout, training=training) if dropout != 0 else out
        return out


## Computes the loss tensor according to the arguments passed to the software
# @param args arguments passed to the command line
# @param net the network object
# @see MLP example of a network object
def get_loss_sum(args, out, out_true):
    logger.debug("Defining loss")
    loss_type = args.loss_type
    if loss_type == "mse":
        loss = tf.reduce_sum(tf.reduce_mean((out - out_true) ** 2, axis=1))
    if loss_type == "rmse":
        loss = tf.reduce_sum(tf.reduce_mean(tf.abs(out - out_true), axis=1))
    if loss_type == "softmax_ce":
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=out_true))
    return loss


## Compute a tensor containing the amount of example correctly classified in a batch
# @param net the network object
# @see MLP example of a network object
def get_ncorrect(out, out_true):
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(out_true, 1))
    ncorrect = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    return ncorrect


## Defines an training operation according to the arguments passed to the software
# @param args arguments passed to the command line
# @param loss loss tensor
# @see get_loss_sum function to generate a loss tensor
def get_train_op(args, loss):
    logger.debug("Defining optimizer")
    optimizer_type = args.optimizer_type
    lr = args.learning_rate
    if optimizer_type == "adam":
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
    if optimizer_type == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(lr).minimize(loss)
    if optimizer_type == "gradient_descent":
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    return optimizer


## General function defining a complete tensorflow graph
# @param args arguments passed to the command line
# @see get_loss_sum function to generate a loss tensor
# @see get_ncorrect function to generate a tensor containing number of correct classification in a batch
# @see get_optimizer function to generate an optimizer object
# @see MLP example of a network object
def get_graph(args, data_tensors):
    logger.info("Defining graph")
    graph = {}
    net = MLP(args)
    graph["test_out"] = net(data_tensors["test_features"])
    graph["test_batch_size"] = tf.shape(graph["test_out"])[0]
    graph["test_loss_sum"] = get_loss_sum(args, graph["test_out"], data_tensors["test_labels"])
    graph["test_ncorrect"] = get_ncorrect(graph["test_out"], data_tensors["test_labels"])
    if args.test_or_train == "train":
        graph["train_out"] = net(data_tensors["train_features"])
        graph["train_batch_size"] = tf.shape(graph["train_out"])[0]
        graph["train_loss_sum"] = get_loss_sum(args, graph["train_out"], data_tensors["train_labels"])
        graph["train_ncorrect"] = get_ncorrect(graph["train_out"], data_tensors["train_labels"])
        graph["train_op"] = get_train_op(args, graph["train_loss_sum"])
    logger.info("Graph defined")
    return graph
