## @package graph
#  Package responsible for the tensorflow graph definition.
#
#  This package provides functions to define a multi-layer perceptron according to the arguments passed.
#  It also provides separate functions to define the loss and the training algorithm.
import tensorflow as tf
import numpy as np
import logging as log


logger = log.getLogger("classifier")

regularizer = tf.keras.regularizers.l2(l=0.01)
# python3
# initializer = tf.keras.initializers.he_normal(seed=589)
initializer = tf.compat.v1.keras.initializers.he_normal()



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
        elif act == 'softmax':
            acti_funcs.append(tf.nn.softmax)
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
        print("-------Building {} network-----------".format(args.model_name))
        self.layer_dims = args.layer_dims
        self.drop_fc = args.drop_fc
        self.batch_norm = args.batch_norm
        self.num_classes = args.num_classes

    def __call__(self, features, training=False):
        out = {}
        net = features
        for ind, dim in enumerate(self.layer_dims):
            net = self._make_layer(net, ind, self.drop_fc[ind], training)
        activity = net

        with tf.compat.v1.variable_scope("logits", reuse=tf.compat.v1.AUTO_REUSE):
            net = tf.compat.v1.layers.dense(net, self.num_classes,
                                            kernel_initializer=initializer,
                                            activation=None)
            net = tf.compat.v1.layers.batch_normalization(net, training=training) if self.batch_norm else net
            logits = tf.nn.softmax(net)
        
        ##### track all variables
        all_trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        out["total_trainables"] = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in all_trainable_vars])
        print("MLP total_trainables {} during training={}".format(out["total_trainables"], training))
        
        out["logits"] = logits
        out["activity"] = activity
        return out

    def _make_layer(self, inp, layer_number, dropout, training):
        """
        Make dense layer
        :param inp:
        :param layer_number:
        :param dropout:
        :param training:
        :return:
        """
        out_size = self.layer_dims[layer_number]

        _to_format = [out_size, dropout, training]
        layer_name = "layer_{}".format(layer_number)
        logger.debug("Creating new layer:")
        string = "Output size = {}\tDropout prob = {}\t(training = {})"
        logger.debug(string.format(*_to_format))

        print("-------Building network-----------")
        with tf.compat.v1.variable_scope(layer_name, reuse=tf.compat.v1.AUTO_REUSE):
            net = tf.compat.v1.layers.dense(inp, out_size,
                                            kernel_initializer=initializer,
                                            activation=None)
            net = tf.compat.v1.layers.batch_normalization(net, training=training) if self.batch_norm else net
            net = tf.nn.relu(net)
            net = tf.compat.v1.layers.dropout(net, rate=dropout, training=training) if dropout != 0 else net
            print("layer: {}, in_size:{}, out_size:{}".format(layer_name, inp.get_shape().as_list(),
                                                              net.get_shape().as_list()))
            return net


class CNN:
    ## Constructor
    # construct a CNN: cnn 8*3*1-pool2*1-cnn 16*3*1-pool2*1-cnn 32 3*1-pool2*1-fnn--softmax(class)
    #  @param args arguments passed to the command line
    def __init__(self, args):
        print("-------Building {} network-----------".format(args.model_name))
        self.height = args.height
        self.width = args.width
        self.out_channels = np.array(args.out_channels)
        self.fc_dims = np.array(args.fc)
        self.kernel_size = args.kernel_size
        self.pool_size = args.pool_size
        self.bn_cnn = np.array(args.batch_norms)[0:len(self.out_channels)]
        self.bn_fnn = np.array(args.batch_norms)[len(self.out_channels):]
        self.activations_cnn = convert_activation(args.activations)[0:len(self.out_channels)]
        self.activations_fnn = convert_activation(args.activations)[len(self.out_channels):]
        self.drop_cnn = np.array(args.dropout_probs)[0:len(self.out_channels)]
        self.drop_fnn = np.array(args.dropout_probs)[len(self.out_channels):]

        assert (self.fc_dims[-1] == args.num_classes), "Softmax output does not match number of classes"
        assert (len(self.bn_cnn) == len(self.out_channels) == len(self.drop_cnn) == len(self.activations_cnn))
        assert (len(self.bn_fnn) == len(self.fc_dims) == len(self.drop_fnn) == len(self.activations_fnn))

    def __call__(self, features, training=False):
        out = {}
        net = tf.reshape(features, [-1, self.height, self.width, 1])
        self._net_constructed_once = True
        net = self.construct_cnn_layers(net, training)
        net = self.construct_fnn_layers(net, training)
        
        ##### track all variables
        all_trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        out["total_trainables"] = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in all_trainable_vars])
        print("CNN total_trainables {} during training={}".format(out["total_trainables"], training))
        out["logits"] = net
        return out

    def construct_cnn_layers(self, inp, training):
        """
        Construct the whole cnn layers
        :param inp:
        :param training:
        :return:
        """
        layer_number = 1
        net = inp
        for (out_ch, bn, activation, drop) in zip(self.out_channels, self.bn_cnn, self.activations_cnn, self.drop_cnn):
            net = self._make_cnn_layer(net, out_ch, bn, activation, drop, layer_number, training)
            layer_number += 1

        return net

    def construct_fnn_layers(self, inp, training):
        """
        Construct the whole fully-connected layers
        :param inp: input tensors
        :param training: bool
        :return: output tensors of the layers
        """
        layer_number = 1
        net = tf.compat.v1.layers.flatten(inp)
        for (out_dim, bn, activation, drop) in zip(self.fc_dims, self.bn_cnn, self.activations_cnn, self.drop_cnn):
            net = self._make_fnn_layer(net, out_dim, bn, activation, drop, layer_number, training)
            layer_number += 1

        return net

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
        layer_name = "cnn_layer_{}".format(layer_number)
        logger.debug("Creating new layer:")
        string = "Output size = {}\tBatch norm = {}\tDropout prob = {}\tActivation = {} (training = {})"
        logger.debug(string.format(*_to_format))
        with tf.compat.v1.variable_scope(layer_name, reuse=tf.compat.v1.AUTO_REUSE):
            print("layer {} in_size {} out_size {}".format(layer_name, inp.get_shape().as_list(), out_ch))
            kernel_size = inp.get_shape().as_list()[1]
            net = tf.compat.v1.layers.conv2d(inp, out_ch,
                                             kernel_size, 1,
                                             padding='SAME',
                                             kernel_initializer=initializer)
            net = tf.compat.v1.layers.max_pooling2d(net, pool_size=[self.pool_size, 1],
                                                    strides=[self.stride, 1],
                                                    padding='same')
            net = tf.compat.v1.layers.batch_normalization(
                net, training=training) if bn else net
            net = net if activation is None else activation(net)
            net = tf.compat.v1.layers.dropout(net, rate=drop, training=training) if drop != 0 else net
        print("layer {} out_size {}".format(layer_name, net.get_shape().as_list()))

        return net

    ## Private function for adding fully-connected layers to the network
    # @param inp input tensor
    # @param out_size size of the new layer
    # @param batch_norm bool stating if batch normalization should be used
    # @param dropout droupout probability. Set to 0 to disable
    # @param activation activation function
    def _make_fnn_layer(self, inp, out_dim, bn, activation, drop, layer_number, training):
        _to_format = [out_dim, bn, drop, activation, training]
        layer_name = "fc_layer_{}".format(layer_number)
        logger.debug("Creating new layer:")
        string = "Output size = {}\tBatch norm = {}\tDropout prob = {}\tActivation = {} (training = {})"
        logger.debug(string.format(*_to_format))
        with tf.compat.v1.variable_scope(layer_name, reuse=tf.compat.v1.AUTO_REUSE):
            print("layer {} in_size {} out_size {}".format(layer_name, inp.get_shape().as_list(), out_dim))
            net = tf.compat.v1.layers.dense(inp, out_dim,
                                            kernel_initializer=initializer,
                                            activation=activation)
            net = tf.compat.v1.layers.batch_normalization(
                net, training=training) if bn else net
            net = tf.compat.v1.layers.dropout(net, rate=drop, training=training) if drop != 0 else net
        print("layer {} out_size {}".format(layer_name, net.get_shape().as_list()))
        return net


class CNN_CAM:
    ## Constructor
    # construct a CNN: cnn 8*3*1-pool2*1-cnn 16*3*1-pool2*1-cnn 32 3*1-pool2*1-fnn--softmax(class)
    #  @param args arguments passed to the command line
    def __init__(self, args):
        print("-------Building {} network-----------".format(args.model_name))
        self.height = args.height
        self.width = args.width
        self.out_channels = np.array(args.out_channels)
        self.kernel_size = args.kernel_size
        self.pool_size = args.pool_size
        self.strides = args.strides
        self.bn_cnn = np.array(args.batch_norms)[0:len(self.out_channels)]
        self.activations_cnn = convert_activation(args.activations)[0:len(self.out_channels)]
        self.drop_cnn = np.array(args.dropout_probs)[0:len(self.out_channels)]
        self.num_classes = args.num_classes
        self.num_layers = args.num_layers

        assert (len(self.bn_cnn) == len(self.out_channels) == len(self.drop_cnn) == len(self.activations_cnn))

    def __call__(self, features, training=False):
        out = {}
        inp = tf.reshape(features, [-1, self.height, self.width, 1])
        self._net_constructed_once = True
        out["conv"] = self.construct_cnn_layers(inp, training)
        # GAP layer - global average pooling
        with tf.compat.v1.variable_scope('GAP', reuse=tf.compat.v1.AUTO_REUSE) as scope:
            net_gap = tf.reduce_mean(out["conv"],
                                     (1))  # get the mean of axis 1 and 2 resulting in shape [batch_size, filters]
            print("gap shape", net_gap.shape.as_list())

            gap_w = tf.compat.v1.get_variable('W_gap', shape=[net_gap.get_shape().as_list()[-1], self.num_classes],
                                              initializer=tf.random_normal_initializer(0., 0.01))
            logits = tf.nn.softmax(tf.matmul(net_gap, gap_w))

        ##### track all variables
        all_trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        out["total_trainables"] = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in all_trainable_vars])
        print("CNN_CAM total_trainables {} during training={}".format(out["total_trainables"], training))
        
        out["logits"] = logits
        out["gap_w"] = gap_w
        return out

    def construct_cnn_layers(self, inp, training):
        """
        Construct the whole cnn layers
        :param inp:
        :param training:
        :return:
        """
        layer_number = 1
        net = inp
        for (out_ch, bn, activation, drop, num_l) in zip(self.out_channels, self.bn_cnn, self.activations_cnn,
                                                         self.drop_cnn, self.num_layers):
            net = self._make_cnn_layer(net, out_ch, bn, activation, drop, layer_number, num_l, training)
            layer_number += 1

        return net

    def construct_fnn_layers(self, inp, training):
        """
        Construct the whole fully-connected layers
        :param inp: input tensors
        :param training: bool
        :return: output tensors of the layers
        """
        layer_number = 1
        net = tf.compat.v1.layers.flatten(inp)
        for (out_dim, bn, activation, drop) in zip(self.fc_dims, self.bn_cnn, self.activations_cnn, self.drop_cnn):
            net = self._make_fnn_layer(net, out_dim, bn, activation, drop, layer_number, training)
            layer_number += 1

        return net

    def _make_cnn_layer(self, inp, out_ch, bn, activation, drop, layer_number, num_layers, training):
        """
        To creat CNN block
        :param inp:
        :param out_ch: int, num of filters to use
        :param bn: bool
        :param activation: tf function
        :param drop: float, drop rate
        :param layer_number: int
        :param num_layers: int, how many layers in one CNN block, like VGG
        :param training:
        :return:
        """
        _to_format = [out_ch, bn, drop, activation, training]
        layer_name = "cnn_layer_{}".format(layer_number)
        logger.debug("Creating new layer:")
        string = "Output size = {}\tBatch norm = {}\tDropout prob = {}\tActivation = {} (training = {})"
        logger.debug(string.format(*_to_format))
        net = inp
        with tf.compat.v1.variable_scope(layer_name, reuse=tf.compat.v1.AUTO_REUSE):
            print("layer {} in_size {} out_size {}".format(layer_name, inp.get_shape().as_list(), out_ch))
            if self.kernel_size >= 100:
                kernel_size = inp.get_shape().as_list()[
                                  1] // 2  # later layers, the filter size should be adjusted by the input
            else:
                kernel_size = self.kernel_size
            net = tf.compat.v1.layers.conv2d(inputs=net,
                                             filters=out_ch,
                                             kernel_size=[kernel_size, 1],
                                             strides=[1, 1],
                                             padding='SAME',
                                             kernel_initializer=initializer,
                                             # kernel_regularizer = regularizer,
                                             activation=None)
            # if np.mod(layer_number, 2) == 1:  # only pool after odd number layer
            net = tf.compat.v1.layers.max_pooling2d(net, pool_size=[self.pool_size, 1],
                                                    strides=[self.stride, 1],
                                                    padding='same')
            net = tf.compat.v1.layers.batch_normalization(
                net, training=training) if bn else net
            net = net if activation is None else activation(net)
            net = tf.compat.v1.layers.dropout(net, rate=drop, training=training) if drop != 0 else net
        print("layer {} out_size {}".format(layer_name, net.get_shape().as_list()))

        return net

    ## Private function for adding fully-connected layers to the network
    # @param inp input tensor
    # @param out_size size of the new layer
    # @param batch_norm bool stating if batch normalization should be used
    # @param dropout droupout probability. Set to 0 to disable
    # @param activation activation function
    def _make_fnn_layer(self, inp, out_dim, bn, activation, drop, layer_number, training):
        _to_format = [out_dim, bn, drop, activation, training]
        layer_name = "fc_layer_{}".format(layer_number)
        logger.debug("Creating new layer:")
        string = "Output size = {}\tBatch norm = {}\tDropout prob = {}\tActivation = {} (training = {})"
        logger.debug(string.format(*_to_format))
        with tf.compat.v1.variable_scope(layer_name, reuse=tf.compat.v1.AUTO_REUSE):
            print("layer {} in_size {} out_size {}".format(layer_name, inp.get_shape().as_list(), out_dim))
            net = tf.compat.v1.layers.dense(inp, out_dim,
                                            kernel_initializer=initializer,
                                            activation=activation)
            net = tf.compat.v1.layers.batch_normalization(
                net, training=training) if bn else net
            net = tf.compat.v1.layers.dropout(net, rate=drop, training=training) if drop != 0 else net
        print("layer {} out_size {}".format(layer_name, net.get_shape().as_list()))
        return net


class Res_FNN:
    ## Construct residual fully-connected networks
    def __init__(self, args):
        logger.info("constructing Res_FNN")

    def __call__(self, features, training=False):
        out = {}
        net = features
        net = self.make_res_fnn_block(net, training=training)

        out["conv"] = self.construct_res_blocks_ecg(net, training=training)
        # GAP layer - global average pooling
        with tf.compat.v1.variable_scope('GAP', reuse=tf.compat.v1.AUTO_REUSE) as scope:
            net_gap = tf.squeeze(tf.reduce_mean(out["conv"], (1)),
                                 axis=1)  # get the mean of axis 1 and 2 resulting in shape [batch_size, filters]

            print("gap shape", net_gap.get_shape().as_list())

            gap_w = tf.compat.v1.get_variable('W_gap', shape=[net_gap.get_shape().as_list()[-1], self.num_classes],
                                              initializer=tf.random_normal_initializer(0., 0.01))
            logits = tf.nn.softmax(tf.matmul(net_gap, gap_w))

        ##### track all variables
        all_trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        out["total_trainables"] = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in all_trainable_vars])
        print("Res_ECG_CAM total_trainables {} during training={}".format(out["total_trainables"], training))

        out["logits"] = logits
        out["gap_w"] = gap_w
        return out

    def make_res_fnn_block(self, inp, unit, layer_id=0, training=True):
        """
        Construct the whole cnn layers
        :param inp:
        :param training:
        :return:
        """
        net = inp
        with tf.compat.v1.variable_scope("res_fnn_block" + str(layer_id), reuse=tf.compat.v1.AUTO_REUSE):
            for j in range(self.num_layers_in_res):  # there are two conv layers in one block

                net = tf.compat.v1.layers.batch_normalization(net, training=training)
                net = tf.nn.relu(net)
                if not (layer_id == 0 and j == 0):
                    drop = self.drop_fnn if j > 0 else 0
                    net = tf.compat.v1.layers.dropout(net, drop, training=training)

                net = tf.compat.v1.layers.dense(net, unit, kernel_initializer=initializer, activation=None)

            shortcut = inp
            output = tf.nn.relu(shortcut + net)
            print("ResiBlock{}-output pooling shape".format(layer_id), net.shape.as_list())
            return output
        return net



class Res_ECG_CAM:
    ## Constructor
    # construct a CNN: cnn 8*3*1-pool2*1-cnn 16*3*1-pool2*1-cnn 32 3*1-pool2*1-fnn--softmax(class)
    #  @param args arguments passed to the command line
    def __init__(self, args):
        print("-------Building {} network-----------".format(args.model_name))
        self.height = args.height
        self.width = args.width
        self.channel_start = args.out_channels  # Starting num of channels
        self.num_layers_in_res = args.num_layers_in_res  #
        self.num_res_blocks = args.num_res_blocks
        self.kernel_size = args.filter_size  # repeat for all the cnn
        self.pool_size = args.pool_size  # repeat for all the cnn
        self.stride = args.stride  # repeat for all the cnn
        self.drop_cnn = args.drop_cnn  # repeat for all the cnn
        self.bn = args.bn
        self.num_classes = args.num_classes
        self.increase_interval = min(self.num_res_blocks // 3, 4)

    def __call__(self, features, training=False):
        out = {}
        inp = tf.reshape(features, [-1, self.height, self.width, 1])
        self._net_constructed_once = True

        net = self._make_cnn_layer(inp, self.channel_start,
                                   self.bn, self.drop_cnn, 1, 1,
                                   training=training)
        net = self.build_res_block_ecg_1st(net, training=training)

        out["conv"] = self.construct_res_blocks_ecg(net, training=training)
        # GAP layer - global average pooling
        with tf.compat.v1.variable_scope('GAP', reuse=tf.compat.v1.AUTO_REUSE) as scope:
            net_gap = tf.squeeze(tf.reduce_mean(out["conv"], (1)),
                                 axis=1)  # get the mean of axis 1 and 2 resulting in shape [batch_size, filters]

            print("gap shape", net_gap.get_shape().as_list())

            gap_w = tf.compat.v1.get_variable('W_gap', shape=[net_gap.get_shape().as_list()[-1], self.num_classes],
                                              initializer=tf.random_normal_initializer(0., 0.01))
            logits = tf.nn.softmax(tf.matmul(net_gap, gap_w))
            
        ##### track all variables
        all_trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        out["total_trainables"] = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in all_trainable_vars])
        print("Res_ECG_CAM total_trainables {} during training={}".format(out["total_trainables"], training))
        
        out["logits"] = logits
        out["gap_w"] = gap_w
        return out

    def construct_res_blocks_ecg(self, inp, training=True):
        """
        Construct the whole cnn layers
        :param inp:
        :param training:
        :return:
        """
        net = inp
        channel = self.channel_start
        k = 0
        strides = [2 if (i + 1) % self.increase_interval == 0 else 1 for i in
                   range(self.num_res_blocks)]  # downsizing in every 4 blocks
        block_ids = np.arange(self.num_res_blocks)

        for bl_id, s in zip(block_ids, strides):
            if (bl_id + 1) % self.increase_interval == 0 and bl_id > 0:
                k += 1
                channel = self.channel_start * np.power(2, k)

            net = self.build_res_blocks_ecg(net, channel, s,
                                            layer_id=bl_id, training=training)
        return net

    def _make_cnn_layer(self, inp, out_ch, bn, drop, layer_number, num_layers, training=True):
        """
        To creat CNN block
        :param inp:
        :param out_ch: int, num of filters to use
        :param bn: bool
        :param activation: tf function
        :param drop: float, drop rate
        :param layer_number: int
        :param num_layers: int, how many layers in one CNN block, like VGG
        :param training:
        :return:
        """
        _to_format = [out_ch, bn, drop, training]
        layer_name = "cnn_layer_{}".format(layer_number)
        logger.debug("Creating new layer:")

        net = inp
        with tf.compat.v1.variable_scope(layer_name, reuse=tf.compat.v1.AUTO_REUSE):
            print("layer {} in_size {} out_size {}".format(layer_name, inp.get_shape().as_list(), out_ch))
            kernel_size = min(self.kernel_size, inp.get_shape().as_list()[1])
            net = tf.compat.v1.layers.conv2d(inputs=net,
                                             filters=out_ch,
                                             kernel_size=[kernel_size, 1],
                                             strides=[1, 1],
                                             padding='SAME',
                                             kernel_initializer=initializer,
                                             # kernel_regularizer = regularizer,
                                             activation=None)

            net = tf.compat.v1.layers.batch_normalization(
                net, training=training) if bn else net
            net = tf.nn.relu(net)
            # out = tf.compat.v1.layers.dropout(out, rate=drop, training=training) if drop != 0 else out

        return net

    def build_res_block_ecg_1st(self, inp, training=True):
        """
        https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
        :param out_channels: int, the filters to use in this block
        :param filter_size: [height, width], the kernel size
        :param num_layers: int, how many cov layers in one resi block. inp--> (conv -->...--> conv) -->+inp-->
        :param layer_id: int, the layer id
        :return: Conv bn relu drop conv
        """
        net = inp

        with tf.compat.v1.variable_scope("res_block_start", reuse=tf.compat.v1.AUTO_REUSE):
            net = tf.compat.v1.layers.conv2d(
                inputs=net,
                filters=self.channel_start,
                kernel_size=[self.kernel_size, 1],
                strides=[self.stride, 1],  # reduce the height, because shortcut also reduce the height
                padding='SAME',
                kernel_initializer=initializer,
                # kernel_regularizer=regularizer,
                activation=None
            )
            net = tf.compat.v1.layers.batch_normalization(net, training=training)
            net = tf.nn.relu(net)
            net = tf.compat.v1.layers.dropout(net, self.drop_cnn, training=training)
            net = tf.compat.v1.layers.conv2d(
                inputs=net,
                filters=self.channel_start,
                kernel_size=[self.kernel_size, 1],
                padding='SAME',
                kernel_initializer=initializer,
                # kernel_regularizer=regularizer,
                activation=None
            )
            shortcut = tf.compat.v1.layers.max_pooling2d(inp, pool_size=[self.pool_size, 1],
                                                         strides=[self.stride, 1],
                                                         padding='same')
            output = tf.nn.relu(shortcut + net)
            print("ResiBlock_start-output pooling shape", net.shape.as_list())
            return output

    def build_res_blocks_ecg(self, x, out_channel, stride, layer_id=0, training=True):
        """
        https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
        :param out_channels: int, the filters to use in this block
        :param filter_size: [height, width], the kernel size
        :param num_layers: int, how many cov layers in one resi block. inp--> (conv -->...--> conv) -->+inp-->
        :param layer_id: int, the layer id
        :return: bn relu  conv bn relu drop conv
        """
        net = x
        if (
                layer_id + 1) % self.increase_interval == 0 and layer_id > 0:  # only every 4 blocks increase the number of channels and decrease the height
            zeros_x = tf.zeros_like(x)
            concat_long_ch = tf.concat([x, zeros_x], axis=3)
            x = concat_long_ch

        with tf.compat.v1.variable_scope("res_block" + str(layer_id), reuse=tf.compat.v1.AUTO_REUSE):
            for j in range(self.num_layers_in_res):  # there are two conv layers in one block
                print(training)
                net = tf.compat.v1.layers.batch_normalization(net, training=training)
                net = tf.nn.relu(net)
                if not (layer_id == 0 and j == 0):
                    drop = self.drop_cnn if j > 0 else 0
                    net = tf.compat.v1.layers.dropout(net, drop, training=training)

                net = tf.compat.v1.layers.conv2d(
                    inputs=net,
                    filters=out_channel,
                    kernel_size=[self.kernel_size, 1],
                    padding='SAME',
                    strides=[stride, 1] if j == 0 else [1, 1],
                    kernel_initializer=initializer,
                    # kernel_regularizer=regularizer,
                    activation=None
                )

            shortcut = tf.compat.v1.layers.max_pooling2d(x, pool_size=[self.pool_size, 1], strides=[stride, 1],
                                                         padding='same')
            output = tf.nn.relu(shortcut + net)
            print("ResiBlock{}-output pooling shape".format(layer_id), net.shape.as_list())
            return output


class Inception:
    """
    https://github.com/Natsu6767/Inception-Module-Tensorflow
    https://github.com/Natsu6767/Inception-Module-Tensorflow/blob/master/Inception%20Train%20%26%20Test.ipynb
    Make an inception model
    total_trainables 234338
    """

    def __init__(self, args):
        "https://mohitjain.me/2018/06/09/googlenet/"
        print("-------Building {} network-----------".format(args.model_name))
        self.height = args.height
        self.width = args.width
        self.num_classes = args.num_classes
        self.channel_start = args.out_channels
        self.bn = args.bn
        self.activations = args.activations
        self.drop_fnn = args.drop_fnn
        self.drop_cnn = args.drop_cnn
        self.incep_fs = args.incep_filter_size
        self.filter_size = args.filter_size
        # self.num_incep_blocks = args.num_incep_blocks
        self.conv_1_size = args.conv_1_size
        self.conv_3_size = args.conv_3_size
        self.conv_3_reduced_size = args.conv_3_reduced_size
        self.conv_5_reduced_size = args.conv_5_reduced_size
        self.conv_5_size = args.conv_5_size
        self.pool_size = args.pool_size
        self.stride = args.stride
        self.ks_small = args.ks_small
        self.ks_big = args.ks_big
        self.ks_bbig = args.ks_bbig
        self.num_moduleA = args.num_moduleA
        self.num_moduleB = args.num_moduleB
        self.factor = args.reduce_factor
        

    def _make_cnn_layer(self, inp, out_ch, bn, drop, layer_number, training=True):
        """
        To creat CNN block
        :param inp:
        :param out_ch: int, num of filters to use
        :param bn: bool
        :param activation: tf function
        :param drop: float, drop rate
        :param layer_number: int
        :param training:
        :return:
        """
        _to_format = [out_ch, bn, drop, training]
        layer_name = "cnn_layer_{}".format(layer_number)
        logger.debug("Creating new layer:")
        string = "Output size = {}\tBatch norm = {}\tDropout prob = {}\t (training = {})"
        logger.debug(string.format(*_to_format))
        net = inp
        with tf.compat.v1.variable_scope(layer_name, reuse=tf.compat.v1.AUTO_REUSE):
            print("layer {} in_size {} out_size {}".format(layer_name, inp.get_shape().as_list(), out_ch))
            kernel_size = min(self.kernel_size, inp.get_shape().as_list()[1])
            net = tf.compat.v1.layers.conv2d(inputs=net,
                                             filters=out_ch,
                                             kernel_size=[kernel_size, 1],
                                             strides=[1, 1],
                                             padding='SAME',
                                             kernel_initializer=initializer,
                                             kernel_regularizer=regularizer,
                                             activation=None)

            net = tf.compat.v1.layers.batch_normalization(
                net, training=training) if bn else net
            net = tf.nn.relu(net)
            # out = tf.compat.v1.layers.dropout(out, rate=drop, training=training) if drop != 0 else out
        print("layer {} out_size {}".format(layer_name, net.get_shape().as_list()))

        return net

    def conv_layer(self, x, filter_height, filter_width,
                   num_filters, name, stride=1, padding='SAME', training=False):
        """Create a convolution layer."""
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
            net = tf.compat.v1.layers.conv2d(
                inputs=x,
                filters=num_filters,
                kernel_size=[filter_height, 1],
                strides=[stride, 1],  # reduce the height, because shortcut also reduce the height
                padding=padding,
                kernel_initializer=initializer,
                # kernel_regularizer=regularizer,
                activation=None
            )
            net = tf.compat.v1.layers.batch_normalization(net, training=training)
            net = tf.nn.relu(net)

            return net

    def inception_layer(self, x,
                        conv_1_size=64,
                        conv_3_reduce_size=64, conv_3_size=64,
                        conv_5_reduce_size=128, conv_5_size=128,
                        pool_proj_size=2,
                        name='inception'):
        """
        Create an Inception Layer
        https://mohitjain.me/2018/06/09/googlenet/
        :param x: input
        :param conv_1_size: number of filters in 1x1 branch
        :param conv_3_reduce_size: No. of filters in 1x1 conv layer before 3x3 conv to reduce computation,
        :param conv_3_size: No. of filters in 3x3 conv
        :param conv_5_reduce_size: No. of filters in 1x1 conv layer before 3x3 conv to reduce computation
        :param conv_5_size: No. of filters in 5x5 conv
        :param pool_proj_size: No. of filters following 3x3 max pooling
        :param name:
        :return:
        """

        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
            conv_1 = self.conv_layer(x, filter_height=1, filter_width=1,
                                     num_filters=conv_1_size, name='{}_1x1'.format(name))

            conv_3_reduce = self.conv_layer(x, filter_height=1, filter_width=1,
                                            num_filters=conv_3_reduce_size, name='{}_3x3_reduce'.format(name))

            conv_3 = self.conv_layer(conv_3_reduce, filter_height=self.ks_small, filter_width=1,
                                     num_filters=conv_3_size, name='{}_3x3'.format(name))

            conv_5_reduce = self.conv_layer(x, filter_height=1, filter_width=1,
                                            num_filters=conv_5_reduce_size, name='{}_5x5_reduce'.format(name))

            conv_5 = self.conv_layer(conv_5_reduce, filter_height=self.ks_big, filter_width=1,
                                     num_filters=conv_5_size, name='{}_5x5'.format(name))

            pool = tf.compat.v1.layers.max_pooling2d(x, pool_size=[self.pool_size, 1], strides=[1, 1], padding="SAME")
            pool_proj = self.conv_layer(pool, filter_height=1, filter_width=1, num_filters=pool_proj_size,
                                        name='{}_pool_proj'.format(name))
            
            return tf.concat([conv_1, conv_3, conv_5, pool_proj], axis=3, name='{}_concat'.format(name))
    
    
    def incepModuleB(self, x, ks_small, ks_bbig, factor=1, channel_axis=3, name="mixed5"):
        """
        Build inception module B, omit 1xN conv
        :param x:
        :param channel_axis:
        :param name:
        :return:
        """
        # branch1x1 = conv2d_bn(x, 192//factor, 1, 1)
        branch1x1 = self.conv_layer(x, filter_height=1, filter_width=1,
                                    num_filters=192//factor, stride=1,
                                    name=name+"branch1x1", padding='same')
        
        # branch7x7 = conv2d_bn(x, 128//factor, 1, 1)
        branch7x7 = self.conv_layer(x, filter_height=1, filter_width=1, name=name+"771l",
                                            num_filters=128//factor, stride=1,
                                    padding='same')
        # branch7x7 = conv2d_bn(branch7x7, 192//factor, ks_bbig, 1)
        branch7x7 = self.conv_layer(branch7x7, filter_height=ks_bbig, filter_width=1,
                                                    num_filters=192//factor, stride=1,
                                                    name=name+"7771", padding='same')
        
        # branch7x7dbl = conv2d_bn(x, 128//factor, 1, 1)
        branch7x7dbl = self.conv_layer(branch7x7, filter_height=1, filter_width=1,
                                       num_filters=128//factor, name=name+"77db1l",
                                       stride=1, padding='same')
        # branch7x7dbl = conv2d_bn(branch7x7dbl, 128//factor, ks_bbig, 1)
        branch7x7dbl = self.conv_layer(branch7x7dbl, filter_height=ks_bbig, filter_width=1,
                                               num_filters=128//factor, name=name+"77db71_0",
                                               stride=1, padding='same')
        # branch7x7dbl = conv2d_bn(branch7x7dbl, 128//factor, ks_bbig, 1)
        branch7x7dbl = self.conv_layer(branch7x7dbl, filter_height=ks_bbig, filter_width=1,
                                                       num_filters=128//factor, name=name+"77db71_1",
                                                       stride=1, padding='same')

        # branch_pool = tf.keras.layers.AveragePooling2D(
        #     (ks_small, 1), strides=(1, 1), padding='same')(x)
        branch_pool = tf.compat.v1.layers.max_pooling2d(x, pool_size=[self.ks_small, 1], strides=[1, 1], name=name+"pool", padding="SAME")
        
        # branch_pool = conv2d_bn(branch_pool, 192//factor, 1, 1)
        branch_pool = self.conv_layer(branch_pool, filter_height=1, filter_width=1, name=name+"pool_11", num_filters=192//factor, stride=1, padding='same')
        # x = tf.keras.layers.concatenate(
        #     [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        #     axis=channel_axis,
        #     name=name)
        x = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='{}_concat'.format(name))
        return x
    
    
    def incepModuleA(self, x, ks_small, ks_big, factor=1, channel_axis=3, name="mixed0"):
        """
        build the inception module A
        https://sh-tsang.medium.com/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c
        :param channel_axis:
        :param x:
        :param ks_small
        :param ks_big:
        :param factor: the factor to reduce the number of filters
        :return:
        """
        # branch1x1 = conv2d_bn(x, 64//factor, 1, 1)
        branch1x1 = self.conv_layer(x, filter_height=1, filter_width=1,
                                    num_filters=64//factor, name=name+"1x1",
                                    stride=1, padding='same')
        
        # branch5x5 = conv2d_bn(x, 48//factor, 1, 1)
        branch5x5 = self.conv_layer(x, filter_height=1, filter_width=1,
                                            num_filters=48//factor, name=name+"5511",
                                            stride=1, padding='same')
        # branch5x5 = conv2d_bn(branch5x5, 64//factor, ks_big, 1)
        branch5x5 = self.conv_layer(branch5x5, filter_height=ks_big, filter_width=1,
                                            num_filters=64//factor, name=name+"5551",
                                            stride=1, padding='same')
        
        # branch3x3dbl = conv2d_bn(x, 64//factor, 1, 1)
        branch3x3dbl = self.conv_layer(x, filter_height=1, filter_width=1,
                                            num_filters=64//factor, name=name+"33db11",
                                            stride=1, padding='same')
        # branch3x3dbl = conv2d_bn(branch3x3dbl, 96//factor, ks_small, 1)
        branch3x3dbl = self.conv_layer(branch3x3dbl, filter_height=ks_small, filter_width=1,
                                            num_filters=96//factor, name=name+"33db31",
                                            stride=1, padding='same')
        # branch3x3dbl = conv2d_bn(branch3x3dbl, 96//factor, ks_small, 1)
        branch3x3dbl = self.conv_layer(branch3x3dbl, filter_height=ks_small, filter_width=1,
                                            num_filters=96//factor, name=name+"33db31_2",
                                            stride=1, padding='same')
        # branch_pool = tf.keras.layers.AveragePooling2D((ks_small, 1), strides=(1, 1),
        #                                                padding='same')(x)
        branch_pool = tf.compat.v1.layers.max_pooling2d(x, pool_size=[self.ks_small, 1],
                                                        name=name+"pool",
                                                          strides=[1, 1], padding="SAME")
        # branch_pool = conv2d_bn(branch_pool, 64//factor, 1, 1)
        branch_pool = self.conv_layer(branch_pool, filter_height=1, filter_width=1,
                                            num_filters=64//factor, name=name+"pool_11",
                                            stride=1, padding='same')

        # x = tf.keras.layers.concatenate(
        #     [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        #     axis=channel_axis,
        #     name=name)
        #  [?,67,1,16], [?,57,1,16], [?,59,1,24], [?,67,1,16],
        x = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='{}_concat'.format(name))
        return x
    
    
       
    def __call__(self, features, training=False):
        ret = {}
        conv = tf.reshape(features, [-1, self.height, self.width, 1])

        # Build the whole graph
        # x = conv2d_bn(img_input, 32, self.ks_small, 1, strides=(2, 1), padding='same')
        conv = self.conv_layer(conv, filter_height=self.ks_small, filter_width=1,
                               num_filters=32, stride=1, name='conv1', padding="same")
        print("layer {} out_size {}".format(conv.name, conv.get_shape().as_list()))
        # x = conv2d_bn(x, 32, ks_small, 1, padding='same')
        conv = self.conv_layer(conv, filter_height=self.ks_small, filter_width=1,
                               num_filters=32, stride=1, name='conv2', padding='same')
        print("layer {} out_size {}".format(conv.name, conv.get_shape().as_list()))
        # x = conv2d_bn(x, 64, ks_small, 1)
        conv = self.conv_layer(conv, filter_height=self.ks_small, filter_width=1,
                               num_filters=64, stride=1, name='conv3', padding='same')
        print("layer {} out_size {}".format(conv.name, conv.get_shape().as_list()))
        # x = tf.keras.layers.MaxPooling2D((3, 1), strides=(2, 1))(x)
        conv = tf.compat.v1.layers.max_pooling2d(conv, pool_size=[self.ks_small, 1],
                                                          strides=[self.stride, 1], padding="SAME",
                                                          name="pool1")
        print("layer {} out_size {}".format(conv.name, conv.get_shape().as_list()))
      
        # x = conv2d_bn(x, 80, 1, 1, padding='same')
        conv = self.conv_layer(conv, filter_height=1, filter_width=1, num_filters=80,
                               stride=1, name='conv4', padding='same')
        print("layer {} out_size {}".format(conv.name, conv.get_shape().as_list()))
        # x = conv2d_bn(x, 192, ks_small, 1, padding='same')
        conv = self.conv_layer(conv, filter_height=self.ks_small, filter_width=1,
                               num_filters=192, stride=1, name='conv5', padding='same')
        print("layer {} out_size {}".format(conv.name, conv.get_shape().as_list()))
        # x = tf.keras.layers.MaxPooling2D((self.ks_small, 1), strides=(2, 1))(x)
        conv = tf.compat.v1.layers.max_pooling2d(conv, pool_size=[self.ks_small, 1],
                                                          strides=[self.stride, 1], padding="SAME",
                                                          name="pool2")
        print("layer {} out_size {}".format(conv.name, conv.get_shape().as_list()))
        ## values = original / 4
        factor = 4
        module_count = 0
        for ii in range(self.num_moduleA):
            conv = self.incepModuleA(conv, self.ks_small, self.ks_big, factor=factor,
                                     channel_axis=3, name="mixed{}".format(module_count))
            module_count += 1
    
    
        # mixed 3: 17 x 17 x 768
        # branch3x3 = conv2d_bn(x, 384//factor, ks_small, 1, strides=(2, 1), padding='same')
        branch3x3 = self.conv_layer(conv, filter_height=self.ks_small, filter_width=1,
                                       num_filters=384//factor, stride=2, name='conv6', padding='same')
      
        # branch3x3dbl = conv2d_bn(x, 64//factor, 1, 1)
        branch3x3dbl = self.conv_layer(conv, filter_height=1, filter_width=1,
                                       num_filters=64//factor, stride=1, name='conv7', padding='same')
        # branch3x3dbl = conv2d_bn(branch3x3dbl, 96//factor, ks_small, 1)
        branch3x3dbl = self.conv_layer(branch3x3dbl, filter_height=self.ks_small, filter_width=1,
                                       num_filters=96//factor, stride=1, name='conv8', padding='same')
        # branch3x3dbl = conv2d_bn(
        #     branch3x3dbl, 96//factor, ks_small, 1, strides=(2, 1), padding='same')
        branch3x3dbl = self.conv_layer(branch3x3dbl, filter_height=self.ks_small, filter_width=1,
                                       num_filters=96//factor, stride=2, name='conv9', padding='same')
      
        # branch_pool = tf.keras.layers.MaxPooling2D((self.ks_small, 1), strides=(2, 1))(x)
        branch_pool = tf.compat.v1.layers.max_pooling2d(conv, pool_size=[self.ks_small, 1],
                                                                  strides=[2, 1], padding="SAME",
                                                                  name="pool3")
        conv = tf.keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool],
                               axis=3, name='mixed{}'.format(module_count))
        module_count += 1
      
        # mixed 4: 17 x 17 x 768
        for jj in range(module_count, module_count + self.num_moduleB):
            conv = self.incepModuleB(conv, self.ks_small, self.ks_bbig,
                                     factor=factor, channel_axis=3, name="mixed{}".format(jj))
            module_count += 1
    
        # mixed 8: 8 x 8 x 1280
        # branch3x3 = conv2d_bn(x, 192//factor, 1, 1)
        branch3x3 = self.conv_layer(conv, filter_height=1, filter_width=1,
                                    num_filters=192//factor, stride=1,
                                    name='conv10', padding='same')
        
        # branch3x3 = conv2d_bn(branch3x3, 320//factor, ks_small, 1, strides=(2, 1), padding='same')
        branch3x3 = self.conv_layer(branch3x3, filter_height=self.ks_small, filter_width=1,
                                            num_filters=320//factor, stride=2,
                                            name='conv11', padding='same')
      
        # branch7x7x3 = conv2d_bn(x, 192//factor, 1, 1)
        branch7x7x3 = self.conv_layer(conv, filter_height=1, filter_width=1,
                                            num_filters=192//factor, stride=1,
                                            name='conv12', padding='same')
        # branch7x7x3 = conv2d_bn(branch7x7x3, 192//factor, ks_bbig, 1)
        branch7x7x3 = self.conv_layer(branch7x7x3, filter_height=self.ks_bbig, filter_width=1,
                                            num_filters=192//factor, stride=1,
                                            name='conv13', padding='same')
        # branch7x7x3 = conv2d_bn(
        #     branch7x7x3, 192//factor, ks_small, 1, strides=(2, 1), padding='same')
        branch7x7x3 = self.conv_layer(branch7x7x3, filter_height=self.ks_small, filter_width=1,
                                            num_filters=192//factor, stride=2,
                                            name='conv14', padding='same')
      
        # branch_pool = tf.keras.layers.MaxPooling2D((ks_small, 1), strides=(2,1))(x)
        branch_pool = tf.compat.v1.layers.max_pooling2d(conv,
                                                        pool_size=[self.ks_small, 1],
                                                        strides=[2, 1], padding="same",
                                                        name="pool4")
        # conv = tf.keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool],
        #                       axis=3, name='mixed{}'.format(module_count))
        conv = tf.keras.layers.concatenate([branch3x3, branch7x7x3, branch_pool],
                               axis=3,
                               name='mixed{}'.format(module_count))

        # inception2a = self.inception_layer(conv, conv_1_size=64 // factor,
        #                                    conv_3_reduce_size=64 // factor, conv_3_size=96 // factor,
        #                                    conv_5_reduce_size=48 // factor, conv_5_size=64 // factor,
        #                                    pool_proj_size=64 // factor, name="inception1a")
        #
        # print("layer inception1a out_size {}".format(inception2a.get_shape().as_list()))
        #
        # inception2b = self.inception_layer(inception2a, conv_1_size=128 // factor,
        #                                    conv_3_reduce_size=128 // factor, conv_3_size=192 // factor,
        #                                    conv_5_reduce_size=32 // factor, conv_5_size=96 // factor,
        #                                    pool_proj_size=64 // factor, name="inception1b")
        # print("layer inception1b out_size {}".format(inception2b.get_shape().as_list()))
        #
        # pool2 = tf.compat.v1.layers.max_pooling2d(inception2b, pool_size=[self.pool_size, 1],
        #                                           strides=[self.stride, 1], padding="SAME",
        #                                           name="pool2")
        # print("layer pool2 out_size {}".format(pool2.get_shape().as_list()))
        #
        # inception3a = self.inception_layer(pool2, conv_1_size=192 // factor,
        #                                    conv_3_reduce_size=96 // factor, conv_3_size=208 // factor,
        #                                    conv_5_reduce_size=16 // factor, conv_5_size=48 // factor,
        #                                    pool_proj_size=64, name="inception3a")
        # print("layer inception3a out_size {}".format(inception3a.get_shape().as_list()))
        #
        # inception3b = self.inception_layer(inception3a, conv_1_size=160 // factor,
        #                                    conv_3_reduce_size=112 // factor, conv_3_size=224 // factor,
        #                                    conv_5_reduce_size=24 // factor, conv_5_size=64 // factor,
        #                                    pool_proj_size=64 // factor, name="inception3b")
        # print("layer inception3b out_size {}".format(inception3b.get_shape().as_list()))

        with tf.compat.v1.variable_scope("GAP", reuse=tf.compat.v1.AUTO_REUSE):
            gap = tf.reduce_mean(conv, (1))
            print("layer gap out_size {}".format(gap.get_shape().as_list()))
            gap_dropout = tf.compat.v1.layers.dropout(gap, rate=self.drop_cnn,
                                                      training=training) if self.drop_cnn != 0 else gap
            flatten = tf.compat.v1.layers.flatten(gap_dropout)

        with tf.compat.v1.variable_scope("logits", reuse=tf.compat.v1.AUTO_REUSE):
            logits = tf.compat.v1.layers.dense(flatten, self.num_classes,
                                               kernel_initializer=initializer,
                                               activation=self.activations[-1])
        
        ##### track all variables
        all_trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        ret["total_trainables"] = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in all_trainable_vars])
        print("Inception total_trainables {} during training={}".format(ret["total_trainables"], training))
        ret["logits"] = logits
        return ret


class RNN(object):
    """
    total_trainables 234338
    """
    def __init__(self, args):
        print("-------Building {} network-----------".format(args.model_name))
        self.rnn_dims = args.rnn_dims
        self.drop_rnn = args.drop_rnn  # to drop for the linear transformation of the recurrent state.
        self.drop_rnn_ln = args.drop_rnn_ln  # to drop for the linear transformation of the inputs.
        self.drop_fc = args.drop_fc
        self.fc_dim = args.fc_dim
        self.height = args.height
        self.width = args.width
        self.num_classes = args.num_classes
        self.initializer = tf.compat.v1.keras.initializers.glorot_uniform()

    def build_rnn(self, inp, units, layer_id=0):
        with tf.compat.v1.variable_scope("RNN_{}".format(layer_id), reuse=tf.compat.v1.AUTO_REUSE):
            inputs = tf.unstack(tf.expand_dims(inp, axis=2), axis=1)
            self.rnn_cell = tf.keras.layers.LSTMCell(units, recurrent_dropout=self.drop_rnn, dropout=self.drop_rnn_ln,
                                                     kernel_regularizer=self.initializer)
            rnn_outputs, rnn_state = tf.compat.v1.nn.static_rnn(self.rnn_cell, inputs, dtype=tf.float32)
            # enc_ouputs = tf.stack(rnn_outputs, axis=0)
            print("RNN_{}: rnn_outputs shape: {}".format(layer_id, rnn_outputs[-1].get_shape().as_list()))

        return rnn_outputs[-1]

    def __call__(self, features, training=False):
        ret = {}
        net = tf.reshape(features, [-1, self.height])  # make sure each sample is one time serie dat

        with tf.compat.v1.variable_scope("FCb4RNN", reuse=tf.compat.v1.AUTO_REUSE):
            for ii, unit in enumerate(self.fc_dim):
                print("-------Building network-----------")
                net = tf.compat.v1.layers.dense(net, unit,
                                                kernel_initializer=self.initializer,
                                                activation=None)
                net = tf.compat.v1.layers.batch_normalization(net, training=training)
                net = tf.nn.relu(net)
                net = tf.compat.v1.layers.dropout(net, rate=self.drop_fc, training=training)
                print("FCb4RNN_{}: output shape: {}".format(ii, net.get_shape().as_list()))


            for ii, unit in enumerate(self.rnn_dims):
                print("-------Building network-----------")
                with tf.compat.v1.variable_scope("RNN{}".format(ii), reuse=tf.compat.v1.AUTO_REUSE):
                    net = tf.unstack(tf.expand_dims(net, axis=2), axis=1)
                    rnn_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(unit)
                    # self.rnn_cell = tf.keras.layers.LSTMCell(unit, recurrent_dropout=self.drop_rnn,
                    #                                          dropout=self.drop_rnn_ln,
                    #                                          kernel_regularizer=self.initializer)
                    rnn_outputs, rnn_state = tf.compat.v1.nn.static_rnn(rnn_cell, net, dtype=tf.float32)
                    # enc_ouputs = tf.stack(rnn_outputs, axis=0)
                    net = rnn_outputs[-1]
                    net = tf.compat.v1.layers.dropout(net, rate=self.drop_rnn, training=training)
                    print("RNN_{}: rnn_outputs shape: {}".format(ii, net.get_shape().as_list()))
            #
        # for layer_id, rnn_dim in enumerate(self.rnn_dims):
        #     net = self.build_rnn(net, rnn_dim, layer_id=layer_id)

        ret["rnn_state"] = net

        with tf.compat.v1.variable_scope("FC", reuse=tf.compat.v1.AUTO_REUSE):
            logits = tf.compat.v1.layers.dense(net, self.num_classes,
                                               kernel_initializer=self.initializer,
                                               activation=tf.nn.softmax)

        ##### track all variables
        all_trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        ret["total_trainables"] = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in all_trainable_vars])
        print("RNN total_trainables {} during training={}".format(ret["total_trainables"], training))
        
        ret["logits"] = logits
        return ret


# --------------------------utile functions-----------------------------------
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
        loss = tf.reduce_sum(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=out_true))
    return loss


## Compute a tensor containing the amount of example correctly classified in a batch
# @param net the network object
# @see MLP example of a network object
def get_ncorrect(out, out_true):
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(out_true, 1))
    ncorrect = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    right_inds = correct_prediction
    # wrong_inds = tf.reshape(tf.where(tf.argmax(out, 1) != tf.argmax(out_true, 1)), [-1,])
    return ncorrect, right_inds


## Compute the confusion matrix
# @param net the network object
# @see MLP example of a network object
def get_confusion_matrix(predictions, labels, num_classes):
    conf_matrix = tf.compat.v1.confusion_matrix(tf.argmax(labels, 1), tf.argmax(predictions, 1),
                                                num_classes=num_classes)
    return conf_matrix


def get_roc_curve(predictions, labels, num_classes):
    """
    Get the ROC AUC curve
    :param out:
    :param out_true:
    :param num_classes:
    :return:
    """
    if num_classes == 2:
        return tf.compat.v1.metrics.auc(tf.argmax(labels, 1), tf.argmax(predictions, 1), curve='ROC')


## Defines an training operation according to the arguments passed to the software
# @param args arguments passed to the command line
# @param loss loss tensor
# @see get_loss_sum function to generate a loss tensor
def get_train_op(args, loss, learning_rate_op):
    logger.debug("Defining optimizer")
    optimizer_type = args.optimizer_type
    # lr = args.learning_rate
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if optimizer_type == "adam":
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate_op).minimize(loss)
            print("learning rate", learning_rate_op)
        if optimizer_type == "rmsprop":
            optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate_op).minimize(loss)
        if optimizer_type == "gradient_descent":
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate_op).minimize(loss)
    return optimizer


## General function defining a complete tensorflow graph
# @param args arguments passed to the command line
# @see get_loss_sum function to generate a loss tensor
# @see get_ncorrect function to generate a tensor containing number of correct classification in a batch
# @see get_optimizer function to generate an optimizer object
# @see MLP example of a network object
def get_graph(args, data_tensors):
    print("Defining graph")
    graph = data_tensors
    if args.model_name == "MLP":
        net = MLP(args)
    elif args.model_name == "CNN":
        net = CNN(args)
    elif args.model_name == "CNN_CAM":
        net = CNN_CAM(args)
    elif args.model_name == "Res_ECG_CAM":
        net = Res_ECG_CAM(args)
    elif args.model_name == "Inception":
        net = Inception(args)
    elif args.model_name == "RNN":
        net = RNN(args)

    lr_placeholder = tf.compat.v1.placeholder(tf.float32, [], name='learning_rate')
    net_out = net(data_tensors["test_features"])
    graph["test_logits"] = net_out["logits"]
    if "conv" in net_out.keys():
        graph["test_conv"] = net_out["conv"]
        graph["test_gap_w"] = net_out["gap_w"]
    graph["test_batch_size"] = tf.shape(graph["test_logits"])[0]
    graph["test_num_batches"] = graph["test_num_samples"] // args.test_bs
    graph["test_loss"] = get_loss_sum(args, graph["test_logits"], data_tensors["test_labels"])
    graph["test_num_correct"], graph["test_wrong_inds"] = get_ncorrect(graph["test_logits"],
                                                                       data_tensors["test_labels"])
    graph["test_confusion"] = get_confusion_matrix(graph["test_logits"], data_tensors["test_labels"], args.num_classes)
    graph["test_auc"] = get_roc_curve(graph["test_logits"], data_tensors["test_labels"], args.num_classes)
    if args.train_or_test == "train":
        net_out = net(data_tensors["train_features"], training=True)
        graph["train_logits"] = net_out["logits"]
        graph["train_labels"] = data_tensors["train_labels"]
        graph["train_batch_size"] = tf.shape(graph["train_logits"])[0]
        graph["train_loss"] = get_loss_sum(args, graph["train_logits"], data_tensors["train_labels"])
        graph["train_num_correct"], graph["train_wrong_inds"] = get_ncorrect(graph["train_logits"],
                                                                             data_tensors["train_labels"])
        graph["train_confusion"] = get_confusion_matrix(graph["train_logits"], data_tensors["train_labels"],
                                                        args.num_classes)
        graph["train_learning_rate_op"] = tf.compat.v1.placeholder(tf.float32, [], name='learning_rate')
        # graph["train_op"] = tf.compat.v1.train.AdamOptimizer(learning_rate=graph["learning_rate_op"]).minimize(graph["train_loss"])
        graph["train_op"] = get_train_op(args, graph["train_loss"], graph["train_learning_rate_op"])

    print("Graph defined")
    return graph
