'''
https://github.com/nickbiso/Skopt/blob/master/Scikit%2BOptimize-MNIST.ipynb
define Hparams space
create model with the Hparams
save names with the values of the Hparams
save the best model with the Hparams
Good one (for simple function): https://blog.floydhub.com/guide-to-hyperparameters-search-for-deep-learning-models/
good one: https://scikit-optimize.github.io/stable/
'''
import os
import numpy as np
import datetime

import scipy.io
import matplotlib.pyplot as plt
# from shutil import copyfile
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Reshape, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, Dense, Flatten, LSTM
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
# from skopt.plots import plot_objective, plot_evaluations
# from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args
from sklearn.model_selection import train_test_split

np.random.seed(2594)

def get_hyparams_of_models(model_name="MLP"):
    """
    Given the name of a model to get the hyper-space of parameters
    :param model_name:
    :return:
    """
    dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
                             name='learning_rate')
    dim_patience = Integer(low=1, high=25, name='dim_patience')
    if model_name == "MLP":
        dim_fnn1 = Integer(low=16, high=512, name='dim_fnn1')
        dim_fnn2 = Integer(low=16, high=512, name='dim_fnn2')
        dim_fnn3 = Integer(low=16, high=512, name='dim_fnn3')
        dim_fnn4 = Integer(low=16, high=512, name='dim_fnn4')

        parameter_space = [
            dim_learning_rate,
            dim_fnn1,
            dim_fnn2,
            dim_fnn3,
            dim_fnn4,
            dim_patience
        ]
        default_parameters = [1e-3, 32, 32, 32, 32, 4]
    elif model_name == "RNN" or model_name == "rnn":
        dim_fnn1 = Integer(low=16, high=512, name='dim_fnn1')
        dim_fnn2 = Integer(low=16, high=512, name='dim_fnn2')
        dim_rnn1 = Integer(low=16, high=128, name='dim_rnn1')
        dim_rnn2 = Integer(low=16, high=128, name='dim_rnn2')

        parameter_space = [
            dim_learning_rate,
            dim_fnn1,
            dim_fnn2,
            dim_rnn1,
            dim_rnn2,
            dim_patience
        ]
        default_parameters = [1e-3, 256, 128, 64, 32, 6]
    
    elif model_name == "inception" or model_name == "Inception":
        num_moduleA = Integer(low=1, high=3, name='num_moduleA')
        num_moduleB = Integer(low=1, high=4, name='num_moduleB')
        ks_small = Integer(low=3, high=7, name='ks_small')
        ks_big = Integer(low=7, high=11, name='ks_big')
        ks_bbig = Integer(low=11, high=16, name='ks_bbig')
        reduceX = Integer(low=1, high=6, name='reduceX')

        parameter_space = [
            dim_learning_rate,
            num_moduleA,
            num_moduleB,
            ks_small,
            ks_big,
            ks_bbig,
            reduceX,
            dim_patience
        ]
        default_parameters = [1e-3, 2, 2, 3, 7, 11, 2, 6]

    return default_parameters, parameter_space

# hyparameters_space = {"rnn": [dim_learning_rate, dim_fnn1, dim_fnn2, dim_rnn1, dim_rnn2, dim_patience],
#            "CAM": [dim_learning_rate, num_cnn1, num_cnn2, num_cnn3, dim_activation, dim_kernel_size, dim_patience]
#                       }
# default_par_arrays= {
#     "CAM": [1e-3, 8, 16, 32, 'relu', 100, 4],
#     "rnn": [1e-3, 128, 64, 32, 32, 4]
#                      }
results_root = "../results/Hyperparameter_Optimize"
MODEL_NAME = "inception"

default_params, params_space = get_hyparams_of_models(model_name=MODEL_NAME)


def log_dir_name(**kwargs):
    configs = "{}-".format(MODEL_NAME)
    for k in kwargs:
        configs += "{}_{}-".format(k, kwargs[k])
        
    # #  The dir-name for the TensorBoard log-dir.
    # s = "CAM-lr_{0:.0e}-cnn1_{1}-cnn2_{2}-cnn3_{3}-act_{4}-kernel_{5}"
    #
    # # Insert all the hyper-parameters in the dir-name.
    # log_dir = s.format(learning_rate,
    #                    num_cnn1, num_cnn2, num_cnn3,
    #                    activation,
    #                    kernel_size)

    return configs


def load_and_test(model_dir):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """

    # Print the hyper-parameters.
    # print('learning rate: {0:.1e}'.format(learning_rate))
    # print('num_dense_layers:', num_dense_layers)
    # print('num_dense_nodes:', num_dense_nodes)
    # print('activation:', activation)
    # print('kernel_size:', kernel_size)
    # print('num_filters_cnn1:', num_filters_cnn1)
    # print('num_filters_cnn2:', num_filters_cnn2)
    # print('num_filters_cnn3:', num_filters_cnn3)
    # print('drop_rate:', drop_rate)
    # print()

    results_root = 'Hyperparameter_Optimize_test'
    loaded_model = load_model(model_dir)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = loaded_model.evaluate(x=test_data["features"], y=test_data["labels"], verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(score))
    print()
    
    
def global_average_pooling(x):
    return K.mean(x, axis = (2, 3))


def global_average_pooling_shape(input_shape):
    return input_shape[0:2]


def get_data():
    """
    Load self_saved data. A dict, data["features"], data["labels"]. See the save function in split_data_for_val()
    # First time preprocess data functions are needed: split_data_for_val(args),split_data_for_lout_val(args)
    :param args: Param object with path to the data
    :return:
    """
    data_dir = "../data/20190325/20190325-3class_lout40_train_test_data5.mat"
    mat = scipy.io.loadmat(data_dir)["DATA"]
    spectra = mat[:, 2:]
    labels = mat[:, 1]
    ids = mat[:, 0]
    train_data = {}
    test_data = {}
    num_classes = 2
    ## following code is to get only label 0 and 1 data from the file. TODO: to make this more easy and clear
    if num_classes - 1 < np.max(labels):
        need_inds = np.empty((0))
        for class_id in range(num_classes):
            need_inds = np.append(need_inds, np.where(labels == class_id)[0])
        need_inds = need_inds.astype(np.int32)
        spectra = spectra[need_inds]
        labels = labels[need_inds]
        ids = ids[need_inds]

    test_ratio = 20
    temp_rand = np.arange(len(labels))
    np.random.shuffle(temp_rand)
    spectra_rand = spectra[temp_rand]
    labels_rand = labels[temp_rand]
    ids_rand = ids[temp_rand]
    assert num_classes != np.max(labels), "The number of class doesn't match the data!"
    num_train = int(((100 - test_ratio) * spectra_rand.shape[0]) // 100)

    test_data["spectra"] = spectra_rand[num_train:].astype(np.float32)
    lb_int = np.squeeze(labels_rand[num_train:]).astype(np.int32)
    test_data["labels"] = np.eye(num_classes)[lb_int]
    test_data["ids"] = np.squeeze(ids_rand[num_train:]).astype(np.int32)
    
    ## oversample the minority samples ONLY in training data
    train_data["spectra"] = spectra_rand[0:num_train].astype(np.float32)
    train_data["labels"] = np.squeeze(labels_rand[0:num_train]).astype(np.int32)
    train_data = oversample_train(train_data, num_classes)
    
    # train_data["labels"] = np.eye(num_classes)[np.squeeze(train_data["labels"][0:num_train]).astype(np.int32)]
    
    return train_data, test_data


def oversample_train(train_data, num_classes):
    """
    Oversample the minority samples
    :param train_data:"spectra", 2d array, "labels", 1d array
    :return:
    """
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=34)
    X_resampled, y_resampled = ros.fit_resample(train_data["spectra"], train_data["labels"])
    train_data["spectra"] = X_resampled
    train_data["labels"] = np.eye(num_classes)[np.squeeze(y_resampled).astype(np.int32)]
    
    return train_data


def create_model(learning_rate, num_dense_layers,
                 num_dense_nodes, activation, kernel_size):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """
    
    # Start construction of a Keras Sequential model.
    model = Sequential()

    # Add an input layer which is similar to a feed_dict in TensorFlow.
    # Note that the input-shape must be a tuple containing the image-size.
    model.add(InputLayer(input_shape=(img_size_flat,)))

    # The input from MNIST is a flattened array with 784 elements,
    # but the convolutional layers expect images with shape (28, 28, 1)
    model.add(Reshape(img_shape_full))

    # First convolutional layer.
    # There are many hyper-parameters in this layer, but we only
    # want to optimize the activation-function in this example.
    model.add(Conv2D(kernel_size=kernel_size, strides=(2, 1), filters=16, padding='same', activation=activation, name='layer_conv1'))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))

    # Second convolutional layer.
    # Again, we only want to optimize the activation-function here.
    model.add(Conv2D(kernel_size=kernel_size, strides=(2, 1), filters=36, padding='same',  activation=activation, name='layer_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))

    # Flatten the 4-rank output of the convolutional layers
    # to 2-rank that can be input to a fully-connected / dense layer.
    model.add(Flatten())

    # Add fully-connected / dense layers.
    # The number of layers is a hyper-parameter we want to optimize.
    for i in range(num_dense_layers):
        # Name of the layer. This is not really necessary
        # because Keras should give them unique names.
        name = 'layer_dense_{0}'.format(i+1)

        # Add the dense / fully-connected layer to the model.
        # This has two hyper-parameters we want to optimize:
        # The number of nodes and the activation function.
        model.add(Dense(num_dense_nodes,
                        activation=activation,
                        name=name))

    # Last fully-connected / dense layer with softmax-activation
    # for use in classification.
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    
    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = Adam(lr=learning_rate)
    
    # In Keras we need to compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    
    return model

def create_cam_model(num_cnn1, num_cnn2, num_cnn3, activation, kernel_size):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """
    
    # Start construction of a Keras Sequential model.
    model = Sequential()
    
    # Add an input layer which is similar to a feed_dict in TensorFlow.
    # Note that the input-shape must be a tuple containing the image-size.
    model.add(InputLayer(input_shape=(img_size_flat,)))
    
    # The input from MNIST is a flattened array with 784 elements,
    # but the convolutional layers expect images with shape (28, 28, 1)
    model.add(Reshape(img_shape_full))
    
    # 1st convolutional layer.
    # There are many hyper-parameters in this layer, but we only
    # want to optimize the activation-function in this example.
    model.add(Conv2D(kernel_size=(kernel_size, 1), strides=(1, 1), filters=num_cnn1, padding='same', activation="relu",
                     name='layer_conv1'))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
    
    # 2nd convolutional layer.
    # Again, we only want to optimize the activation-function here.
    model.add(Conv2D(kernel_size=(kernel_size, 1), strides=(1, 1), filters=num_cnn2, padding='same', activation="relu",
                     name='layer_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))

    # 3rd convolutional layer.
    # Again, we only want to optimize the activation-function here.
    model.add(Conv2D(kernel_size=(kernel_size, 1), strides=(1, 1), filters=num_cnn3, padding='same', activation="relu",
                     name='layer_conv3'))
    # Global average pooling
    model.add(Lambda(global_average_pooling,
                     output_shape=global_average_pooling_shape))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer='random_uniform'))
    
    model.summary()
    
    return model


def create_rnn_model(fc_b4rcc1, fc_b4rcc2, lstm1, lstm2, inputsize=[288,]):
    model = Sequential()
    # Note that the input-shape must be a tuple containing the image-size.
    model.add(InputLayer(input_shape=inputsize))
    
    model.add(Dense(fc_b4rcc1, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(fc_b4rcc2, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Reshape(( fc_b4rcc2, 1)))
    model.add(LSTM(lstm1))

    model.add(Dense(num_classes, activation='softmax', kernel_initializer='random_uniform'))
    model.summary()
    return model


def create_MLP_model(fnn1, fnn2, fnn3, fnn4, inputsize=[288,]):
    model = Sequential()
    # Note that the input-shape must be a tuple containing the image-size.
    model.add(InputLayer(input_shape=inputsize))

    model.add(Dense(fnn1, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(fnn2, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(fnn3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(fnn4, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax', kernel_initializer='random_uniform'))
    model.summary()
    return model


def conv2d_bn(x, filters, num_row, num_col, padding='same',
            strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    
    if tf.keras.backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = tf.keras.layers.Conv2D(
          filters, (num_row, num_col),
          strides=strides,
          padding=padding,
          use_bias=False,
          name=conv_name)(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = tf.keras.layers.Activation('relu', name=name)(x)
    
    return x
        
def create_inception_model(num_moduleA, num_moduleB, ks_small, ks_big, ks_bbig, factor, inputsize=[288,]):
    """
    architecture of inceptionV3
    https://sh-tsang.medium.com/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c
    :param num_moduleA:
    :param num_moduleB:
    :param ks_small:
    :param ks_big:
    :param ks_bbig:
    :param factor: the factor to reduced the number of filters used
    :return:
    """
    # Build the whole graph
    img_input = tf.keras.layers.Input(shape=inputsize)
    channel_axis = 3
    include_top = True
    pooling = None
    classifier_activation = 'softmax'
  
    x = conv2d_bn(img_input, 32, ks_small, 1, strides=(2, 1), padding='valid')
    x = conv2d_bn(x, 32, ks_small, 1, padding='valid')
    x = conv2d_bn(x, 64, ks_small, 1)
    x = tf.keras.layers.MaxPooling2D((3, 1), strides=(2, 1))(x)
  
    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, ks_small, 1, padding='valid')
    x = tf.keras.layers.MaxPooling2D((ks_small, 1), strides=(2, 1))(x)
  
    # mixed 0: 35 x 35 x 256
    module_count = 0
    for ii in range(num_moduleA):
        x = incepModuleA(x, ks_small, ks_big, factor=factor, channel_axis=3, name="mixed{}".format(module_count))
        module_count += 1


    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384//factor, ks_small, 1, strides=(2, 1), padding='valid')
  
    branch3x3dbl = conv2d_bn(x, 64//factor, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96//factor, ks_small, 1)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96//factor, ks_small, 1, strides=(2, 1), padding='valid')
  
    branch_pool = tf.keras.layers.MaxPooling2D((ks_small, 1), strides=(2, 1))(x)
    x = tf.keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool],
                           axis=channel_axis,
                           name='mixed{}'.format(module_count))
    module_count += 1
  
    # mixed 4: 17 x 17 x 768
    for jj in range(module_count, module_count+num_moduleB):
        x = incepModuleB(x, ks_small, ks_bbig, factor=factor, channel_axis=3, name="mixed{}".format(jj))
        module_count += 1

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192//factor, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320//factor, ks_small, 1, strides=(2, 1), padding='valid')
  
    branch7x7x3 = conv2d_bn(x, 192//factor, 1, 1)
    # branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192//factor, ks_bbig, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192//factor, ks_small, 1, strides=(2, 1), padding='valid')
  
    branch_pool = tf.keras.layers.MaxPooling2D((ks_small, 1), strides=(2,1))(x)
    x = tf.keras.layers.concatenate([branch3x3, branch7x7x3, branch_pool],
                           axis=channel_axis,
                           name='mixed{}'.format(module_count))
    module_count += 1
    
    if include_top:
        # Classification block
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        
        x = tf.keras.layers.Dense(2, activation=classifier_activation,
                       name='predictions')(x)
    else:
        if pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)
  
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    inputs = img_input
    # Create model.
    model = tf.keras.models.Model(inputs, x, name='inception_v3')
    
    model.summary()
    
    return model


def incepModuleB(x, ks_small, ks_bbig, factor=1, channel_axis=3, name="mixed5"):
    """
    Build inception module B, omit 1xN conv
    :param x:
    :param channel_axis:
    :param name:
    :return:
    """
    branch1x1 = conv2d_bn(x, 192//factor, 1, 1)
    
    branch7x7 = conv2d_bn(x, 128//factor, 1, 1)
    # branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192//factor, ks_bbig, 1)
    
    branch7x7dbl = conv2d_bn(x, 128//factor, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128//factor, ks_bbig, 1)
    # branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128//factor, ks_bbig, 1)
    
    # branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch_pool = tf.keras.layers.AveragePooling2D(
        (ks_small, 1), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192//factor, 1, 1)
    x = tf.keras.layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name=name)
    return x


def incepModuleA(x, ks_small, ks_big, factor=1, channel_axis=3, name="mixed0"):
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
    branch1x1 = conv2d_bn(x, 64//factor, 1, 1)
    
    branch5x5 = conv2d_bn(x, 48//factor, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64//factor, ks_big, 1)
    
    branch3x3dbl = conv2d_bn(x, 64//factor, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96//factor, ks_small, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96//factor, ks_small, 1)
    
    branch_pool = tf.keras.layers.AveragePooling2D((ks_small, 1), strides=(1, 1),
                                                   padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64//factor, 1, 1)
    x = tf.keras.layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name=name)
    return x


@use_named_args(dimensions=params_space)
def fitness(**params):
    # Print the hyper-parameters.
    config_space = []
    conf_str = "{}-".format(MODEL_NAME)
    for k in params:
        config_space.append([k, params[k]])
        print("{}: {}".format(k, params[k]))
        conf_str += "{}_{}-".format(k, params[k])
    
    # Create the neural network with these hyper-parameters.
    if MODEL_NAME == "cam":
        model = create_cam_model(
                                 num_cnn1=params["dim_cnn1"],
                                 num_cnn2=params["dim_cnn2"],
                                 num_cnn3=params["dim_cnn3"],
                                 kernel_size=params["kernel_size"])
    elif MODEL_NAME == "rnn" or MODEL_NAME == "RNN":
        model = create_rnn_model(params["dim_fnn1"], params["dim_fnn2"],
                                 params["dim_rnn1"], params["dim_rnn2"], inputsize=[288, ])
    elif MODEL_NAME == "MLP" or MODEL_NAME == "mlp":
        model = create_MLP_model(params["dim_fnn1"], params["dim_fnn2"],
                                 params["dim_fnn3"], params["dim_fnn4"], inputsize=[288, ])
    elif MODEL_NAME == "inception" or MODEL_NAME == "Inception":
        model = create_inception_model(params["num_moduleA"], params["num_moduleA"],
                                       params["ks_small"], params["ks_big"],
                                       params["ks_bbig"], params["reduceX"],
                                       inputsize=[288, 1, 1])


    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = Adam(lr=params["learning_rate"])

    # In Keras we need to compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Dir-name for the TensorBoard log-files.
    # configs = log_dir_name(lr=learning_rate, dim_fnn1=dim_fnn1, dim_fnn2=dim_fnn2, dim_rnn1=dim_rnn1, dim_rnn2=dim_rnn2)
    
    early_stop = EarlyStopping(monitor='val_accuracy', patience=params["dim_patience"], verbose=1)

    # Use Keras to train the model.
    history = model.fit(x=train_data["spectra"],
                       y=train_data["labels"],
                       epochs=50,
                       batch_size=32,
                       validation_data=validation_data,
                       callbacks=[early_stop])
    
    # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    accuracy = history.history['val_accuracy'][-1]
    
    # run after each epoch has ended during training.
    # This saves the log-files for TensorBoard.
    # Note that there are complications when histogram_freq=1.
    # It might give strange errors and it also does not properly
    # support Keras data-generators for the validation-set.
    # callback_log = TensorBoard(
    #     log_dir=log_dir,
    #     histogram_freq=0,
    #     batch_size=32,
    #     write_graph=True,
    #     write_grads=False,
    #     write_images=False)
   
   
    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.3%}".format(accuracy))
    print()

    # Save the model if it improves on the best-found performance.
    # We use the global keyword so we update the variable outside
    # of this function.
    global best_accuracy

    # If the classification accuracy of the saved model is improved ...
    if accuracy > best_accuracy or accuracy > 0.80:
        time_str = '{0:%Y-%m-%dT%H-%M-%S}'.format(datetime.datetime.now())
        results_dir = os.path.join(results_root,  '{}_{}_{:0.4f}'.format(time_str, MODEL_NAME, accuracy))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        np.savetxt(os.path.join(results_dir, '{}_{}_convergence_plot_{:0.4f}.txt'.format(time_str, MODEL_NAME, accuracy)), np.array(config_space), fmt="%s", delimiter=",")
        #
        # target_file_name = [os.path.join(results_dir, 'Hpopt.py')]
        # src_file_name = ['Hpopt.py']  # copy the model

        # for src_name, target_name in zip(src_file_name, target_file_name):
        #     copyfile(src_name, target_name)
        # with open(os.path.join(results_dir, configs + "-acc-{:0.4f}.csv".format(accuracy)), 'w') as fh:
        #     # Pass the file handle in as a lambda function to make it callable
        #     model.summary(print_fn=lambda x: fh.write(x + '\n'))
            
        # Plot the history accuracy
        # plot_convergence()
        plt.figure()
        plt.plot(history.history['accuracy'], label="train"),
        plt.plot(history.history['val_accuracy'], label="val"),
        plt.ylabel('accuracy'),
        plt.xlabel('trials'),
        plt.legend(loc='upper left'),
        plt.savefig(os.path.join(results_dir, 'accuracy_plot.png'), format='png')
        plt.close()
        
        best_accuracy = accuracy
        
        # Save model
        # model.save(os.path.join(results_dir, 'model_{:0.4f}.h5'.format(accuracy)))

    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    
    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.
    return -accuracy


if __name__ == "__main__":
    data_source = "metabolite"
    best_accuracy = 0.0
    
    if data_source == "metabolite":
        # Number of classes
        num_classes = 2
    
        train_data, test_data = get_data()
        if MODEL_NAME in ["inception", "cam"]:
            train_data["spectra"] = train_data["spectra"].reshape(-1, 288, 1, 1)
            test_data["spectra"] = test_data["spectra"].reshape(-1, 288, 1, 1)
        
        validation_data = (test_data["spectra"], test_data["labels"])
    
    elif data_source == "mnist":
        from tensorflow.examples.tutorials.mnist import input_data
        data = input_data.read_data_sets('data/MNIST/', one_hot=True)

        num_classes = 10
    
        # We know that MNIST images are 28 pixels in each dimension.
        img_size = 288
    
        # Images are stored in one-dimensional arrays of this length.
        img_size_flat = 288
    
        # Tuple with height and width of images used to reshape arrays.
        # This is used for plotting the images.
        img_shape = (img_size, 1)
    
        # Tuple with height, width and depth used to reshape arrays.
        # This is used for reshaping in Keras.
        img_shape_full = (img_size, 1, 1)
    
        # Number of colour channels for the images: 1 channel for gray-scale.
        num_channels = 1

    # fitness(x=default_params)
    time_str = '{0:%Y-%m-%dT%H-%M-%S}'.format(datetime.datetime.now())
    search_result = gp_minimize(func=fitness,
                                dimensions=params_space,
                                acq_func='EI',  # Expected Improvement.
                                n_calls=500,
                                x0=default_params)

    plot_convergence(search_result)
    plt.savefig(os.path.join("../results/Hyperparameter_Optimize", '{}_{}_convergence_plot.png'.format(time_str, MODEL_NAME)), format='png')
    plt.savefig(os.path.join("../results/Hyperparameter_Optimize", '{}_{}_convergence_plot.pdf'.format(time_str, MODEL_NAME)), format='pdf')
    plt.close()
    results = sorted(zip(search_result.func_vals, search_result.x_iters))
    print(results)
    np.savetxt(os.path.join("../results/Hyperparameter_Optimize", '{}_{}_convergence_plot.txt'.format(time_str, MODEL_NAME)), results, fmt="%s", delimiter=",")
    #
    