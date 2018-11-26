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

## Interprets the string to the activation functions --activations
# @param acti_names, list of activations names
# @return acti_funcs, list of actiations functions
def convert_activation(acti_names):
    acti_funcs = []
    for act in acti_names:
        if act == 'relu':
            acti_funcs.append(tf.nn.relu)
        elif act == 'lrelu':
            acti_funcs.append(tf.nn.leaky_relu)
        elif act == 'tanh':
            acti_funcs.append(tf.tanh)
        elif act == 'sigmoid':
            acti_funcs.append(tf.nn.sigmoid)
        elif act == 'None':
            acti_funcs.append(None)
        else:
            raise NameError("Activation function not recognized: {}".format(act))
    return acti_funcs


## Class for a Multilayer perceptron
# defines and store the layers of the network
# implements batch normalization and dropout
class MLP:
    ## Constructor
    #  @param args arguments passed to the command line
    def __init__(self, args):
        logger.debug("Defining multilayer perceptron")
        self.layer_dims = np.array(args.layer_dims)
        self.batch_norms = np.array(args.batch_norms)
        self.activations = convert_activation(args.activations)
        self.dropout_probs = np.array(args.dropout_probs)
        assert(self.layer_dims[-1] == args.num_classes), "Softmax output does not match number of classes"
        assert(self.layer_dims[0] == args.data_len), "Dim of first layer should be the same as the data length"
        assert(len(self.batch_norms) ==
               len(self.activations) ==
               len(self.dropout_probs) ==
               len(self.layer_dims) - 1), "Passed in dims of batch norms or activations or drop do not match"
        self.n_layers = len(self.batch_norms)
        self._net_constructed_once = False
        self._define_variables()

    def _define_variables(self):
        self.weights = []
        self.biases = []
        initializer = tf.contrib.layers.xavier_initializer()
        for in_size, out_size, batch_norm in zip(self.layer_dims[0:-1], self.layer_dims[1:], self.batch_norms):
            # random = tf.truncated_normal(stddev=0.01, shape=(in_size, out_size))
            W = tf.Variable(initializer((in_size, out_size)))
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
        out_size = self.layer_dims[layer_number + 1]
        batch_norm = self.batch_norms[layer_number]
        dropout = self.dropout_probs[layer_number]
        activation = self.activations[layer_number]
        _to_format = [out_size, batch_norm, dropout, activation, training]
        layer_name = "layer_{}".format(layer_number + 1)
        logger.debug("Creating new layer:")
        string = "Output size = {}\tBatch norm = {}\tDropout prob = {}\tActivation = {} (training = {})"
        logger.debug(string.format(*_to_format))
        W = self.weights[layer_number]
        B = self.biases[layer_number]
        with tf.variable_scope(layer_name):
            out = tf.matmul(tf.squeeze(inp), W) if B is None else tf.matmul(tf.squeeze(inp), W) + B
            out = tf.layers.batch_normalization(
                out,
                training=training,
                reuse=self._net_constructed_once,
                name=layer_name) if batch_norm else out
            out = out if activation is None else activation(out)
            out = tf.layers.dropout(out, rate=dropout, training=training) if dropout != 0 else out
            return out


class CNN:
    ## Constructor
    # construct a CNN: cnn 8*3*1-pool2*1-cnn 16*3*1-pool2*1-cnn 32 3*1-pool2*1-fnn--softmax(class)
    #  @param args arguments passed to the command line
    def __init__(self, args):
        logger.debug("Defining CNN")
        self.data_len = args.data_len
        self.out_channels = np.array(args.out_channels)
        self.fc_dims = np.array(args.fc)
        self.kernel_size = args.kernel_size
        self.pool_size = args.pool_size
        self.bn_cnn = np.array(args.batch_norms)[0:len(self.out_channels)]
        self.bn_fnn= np.array(args.batch_norms)[len(self.out_channels):]
        self.activations_cnn = convert_activation(args.activations)[0:len(self.out_channels)]
        self.activations_fnn = convert_activation(args.activations)[len(self.out_channels):]
        self.drop_cnn = np.array(args.dropout_probs)[0:len(self.out_channels)]
        self.drop_fnn = np.array(args.dropout_probs)[len(self.out_channels):]

        assert (len(args.batch_norms) ==            # check total including FNN part
               len(args.activations) ==
               len(args.dropout_probs))

    def __call__(self, features, training=False):
        out = tf.reshape(features, [-1, self.data_len, 1])
        self._net_constructed_once = True
        out = self.construct_cnn_layers(out, training)
        out = self.construct_fnn_layers(out, training)
        return out

    def construct_cnn_layers(self, inp, training):
        """
        Construct the whole cnn layers
        :param inp:
        :param training:
        :return:
        """
        layer_number = 1
        out = inp
        for (out_ch, bn, activation, drop) in zip(self.out_channels, self.bn_cnn,
                                                        self.activations_cnn, self.drop_cnn):
            out = self._make_cnn_layer(out, out_ch, bn, activation, drop, layer_number, training)
            layer_number += 1

        return out

    def construct_fnn_layers(self, inp, training):
        """
        Construct the whole fully-connected layers
        :param inp: input tensors
        :param training: bool
        :return: output tensors of the layers
        """
        layer_number = 1
        out = tf.layers.flatten(inp)
        for (out_dim, bn, activation, drop) in zip(self.fc_dims, self.bn_cnn,
                                                   self.activations_cnn, self.drop_cnn):
            out = self._make_fnn_layer(out, out_dim, bn, activation, drop, layer_number, training)

        return out

    ## Private function for adding one conv layers to the network
    # @param inp input tensor
    # @param out_ch size of the new layer
    # @param pool, kernel size of the pooling layer
    # @param out_ch size of the new layer
    # @param bn bool stating if batch normalization should be used
    # @param activation activation function
    # @param drop, float droupout probability. Set to 0 to disable
    # @param layer_number, int droupout probability. Set to 0 to disable
    # @param training, bool
    def _make_cnn_layer(self, inp, out_ch, bn, activation, drop, layer_number, training):
        _to_format = [out_ch, bn, drop, activation, training]
        layer_name = "layer_{}".format(layer_number + 1)
        logger.debug("Creating new layer:")
        string = "Output size = {}\tBatch norm = {}\tDropout prob = {}\tActivation = {} (training = {})"
        logger.debug(string.format(*_to_format))
        with tf.variable_scope(layer_name):
            out = tf.layers.conv1d(inp, out_ch, self.kernel_size, 1, padding='SAME')
            out = tf.layers.max_pooling1d(out, self.pool_size, self.pool_size, padding="SAME")
            out = tf.layers.batch_normalization(
                out,
                training=training,
                reuse=self._net_constructed_once,
                name=layer_name) if bn else out
            out = out if activation is None else activation(out)
            out = tf.layers.dropout(out, rate=drop, training=training) if drop != 0 else out
        return out

    ## Private function for adding fully-connected layers to the network
    # @param inp input tensor
    # @param out_size size of the new layer
    # @param batch_norm bool stating if batch normalization should be used
    # @param dropout droupout probability. Set to 0 to disable
    # @param activation activation function
    def _make_fnn_layer(self, inp, out_dim, bn, activation, drop, layer_number, training):
        _to_format = [out_dim, bn, drop, activation, training]
        layer_name = "layer_{}".format(layer_number + 1)
        logger.debug("Creating new layer:")
        string = "Output size = {}\tBatch norm = {}\tDropout prob = {}\tActivation = {} (training = {})"
        logger.debug(string.format(*_to_format))
        with tf.variable_scope(layer_name):
            out = tf.layers.dense(inp, out_dim, activation=activation)
            out = tf.layers.batch_normalization(
                out,
                training=training,
                reuse=self._net_constructed_once,
                name=layer_name) if bn else out
            out = tf.layers.dropout(out, rate=drop, training=training) if drop != 0 else out
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
    wrong_inds = correct_prediction
    # wrong_inds = tf.reshape(tf.where(tf.argmax(out, 1) != tf.argmax(out_true, 1)), [-1,])
    return ncorrect, wrong_inds

## Compute the confusion matrix
# @param net the network object
# @see MLP example of a network object
def get_confusion_matrix(out, out_true, num_classes):
    conf_matrix = tf.confusion_matrix(tf.argmax(out_true, 1), tf.argmax(out, 1), num_classes=num_classes)
    return conf_matrix


## Defines an training operation according to the arguments passed to the software
# @param args arguments passed to the command line
# @param loss loss tensor
# @see get_loss_sum function to generate a loss tensor
def get_train_op(args, loss):
    logger.debug("Defining optimizer")
    optimizer_type = args.optimizer_type
    lr = args.learning_rate
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
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
    graph = data_tensors
    if args.model_name == "MLP":
        net = MLP(args)
    elif args.model_name == "CNN":
        net = CNN(args)

    graph["test_out"] = net(data_tensors["test_features"])
    graph["test_batch_size"] = tf.shape(graph["test_out"])[0]
    graph["test_loss_sum"] = get_loss_sum(args, graph["test_out"], data_tensors["test_labels"])
    graph["test_ncorrect"], graph["test_wrong_inds"] = get_ncorrect(graph["test_out"], data_tensors["test_labels"])
    graph["test_confusion"] = get_confusion_matrix(graph["test_out"], data_tensors["test_labels"], args.num_classes)
    if args.test_or_train == "train":
        graph["train_out"] = net(data_tensors["train_features"], training=True)
        graph["train_labels"] = data_tensors["train_labels"]
        graph["train_batch_size"] = tf.shape(graph["train_out"])[0]
        graph["train_loss_sum"] = get_loss_sum(args, graph["train_out"], data_tensors["train_labels"])
        graph["train_ncorrect"], graph["train_wrong_inds"] = get_ncorrect(graph["train_out"], data_tensors["train_labels"])
        graph["train_confusion"] = get_confusion_matrix(graph["train_out"], data_tensors["train_labels"], args.num_classes)
        graph["train_op"] = get_train_op(args, graph["train_loss_sum"])
    logger.info("Graph defined")
    return graph
