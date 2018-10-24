## @package network
#  Package responsible for the neural network definition.
#
#  This package provides functions to define a multi-layer perceptron according to the arguments passed.
#  It also provides separate functions to define the loss and the training algorithm.
import tensorflow as tf
import numpy as np
import logging as log


logger = log.getLogger("classifier")


DTYPE = tf.float32


class MLP:
    def __init__(self, args):
        logger.debug("Defining multilayer perceptron")
        self.layers_dims = np.array(args.layers_dims)
        self.batch_norm = np.array(args.batch_norms)
        self.activations = np.array(args.activations)
        self.dropout_probs = np.array(args.dropout_probs)
        self.layers = [tf.placeholder(shape=(None, self.layers_dims[0]), dtype=DTYPE)]
        self.weights = []
        self.biases = []
        self.training = tf.placeholder(shape=(), dtype=tf.bool)
        for out_size, batch_norm, dropout, activation in zip(self.layers_dims[1:], self.batch_norm, self.dropout_probs, self.activations):
            self.add_layer(self.layers[-1], out_size, batch_norm, dropout, activation)
        self.inp = self.layers[0]
        self.out = self.layers[-1]
        self.out_true = tf.placeholder(shape=self.out.shape, dtype=self.out.dtype)

    def add_layer(self, inp, out_size, batch_norm, dropout, activation):
        logger.debug("Creating new layer:")
        logger.debug("Output size = {}\tBatch norm = {}\tDropout prob = {}\tActivation = {}".format(out_size, batch_norm, dropout, activation))
        in_size = int(self.layers[-1].shape[-1])
        W = tf.Variable(tf.truncated_normal(stddev=0.01, shape=(in_size, out_size)))
        B = None if batch_norm else tf.Variable(tf.zeros(shape=(1, out_size)))
        out = tf.matmul(inp, W) if B is None else tf.matmul(inp, W) + B
        out = tf.layers.batch_normalization(out, training=self.training)
        out = out if activation is None else activation(out)
        out = tf.layers.dropout(out, rate=dropout, training=self.training) if dropout != 0 else out
        self.layers.append(out)
        self.weights.append(W)
        self.biases.append(B)


def get_loss_sum(args, net):
    logger.debug("Defining loss")
    loss_type = args.loss_type
    if loss_type == "mse":
        loss = tf.reduce_sum(tf.reduce_mean((net.out - net.out_true) ** 2, axis=1))
    if loss_type == "rmse":
        loss = tf.reduce_sum(tf.reduce_mean(tf.abs(net.out - net.out_true), axis=1))
    if loss_type == "softmax_ce":
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=net.out, labels=net.out_true))
    return loss


def get_ncorrect(net):
    out = net.out
    out_true = net.out_true
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(out_true, 1))
    ncorrect = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    return ncorrect


def get_optimizer(args, loss):
    logger.debug("Defining optimizer")
    optimizer_type = args.optimizer_type
    lr = args.learning_rate
    if optimizer_type == "adam":
        optimizer = tf.train.AdamOptimizer(lr)
    if optimizer_type == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(lr)
    if optimizer_type == "gradient_descent":
        optimizer = tf.train.GradientDescentOptimizer(lr)
    return optimizer


def get_graph(args):
    logger.info("Defining graph")
    graph = {}
    net = MLP(args)
    graph["network"] = net
    graph["ncorrect"] = get_ncorrect(net)
    if args.test_or_train == "train":
        graph["loss_sum"] = get_loss_sum(args, net)
        graph["optimizer"] = get_optimizer(args, graph["loss_sum"])
        graph["train_op"] = graph["optimizer"].minimize(graph["loss_sum"])
    logger.info("Graph defined")
    return graph
