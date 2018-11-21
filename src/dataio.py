## @package dataio
#  This package handles the interface with the hard drive.
#
#  It can in particular read and write matlab or python matrices.
import numpy as np
import logging as log
import sys
import os
import plot
import tensorflow as tf
import scipy.io


logger = log.getLogger("classifier")


# def get_data(args):
#     spectrums = scipy.io.loadmat(args.input_data + '/BIGDATA.mat')["BIGDATA"]
#     labels = scipy.io.loadmat(args.input_data + '/data.mat')['data'][:, 1]
#     return spectrums.astype(np.float32), labels.astype(np.int32)


def get_data(args):
    mat = scipy.io.loadmat(args.input_data)["DATA"]
    spectra = mat[:, 2:]
    labels = mat[:, 1]
    return spectra.astype(np.float32), labels.astype(np.int32)


## Get batches of data in tf.dataset
# @param args the arguments passed to the software
def get_data_tensors(args, spectrums, labels):
    data = {}
    spectrums_phd = tf.placeholder(shape=spectrums.shape, dtype=tf.float32)
    labels_phd = tf.placeholder(shape=labels.shape, dtype=tf.int32)
    spectrums, labels = tf.constant(spectrums), tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((spectrums, labels))
    # train and test split
    num_train = int(((100 - args.test_ratio) * spectrums.get_shape().as_list()[0]) // 100)
    train_ds = dataset.take(num_train).repeat(args.number_of_epochs).shuffle(buffer_size=num_train // 2).batch(args.batch_size)
    test_ds = dataset.skip(num_train).batch(args.batch_size)
    iter_test = test_ds.make_initializable_iterator()
    data["test_initializer"] = iter_test.initializer
    batch_test = iter_test.get_next()
    data["test_labels"] = tf.one_hot(batch_test[1], args.num_classes)
    data["test_features"] = batch_test[0]
    if args.test_or_train == 'train':
        iter_train = train_ds.make_initializable_iterator()
        batch_train = iter_train.get_next()
        data["train_labels"] = tf.one_hot(batch_train[1], args.num_classes)
        data["train_features"] = batch_train[0]
        data["train_initializer"] = iter_train.initializer
    return data


## Make the output dir
# @param args the arguments passed to the software
def make_output_dir(args):
    path = args.output_path
    if os.path.isdir(path):
        logger.critical("Output path already exists. Please use an other path.")
        raise FileExistsError("Output path already exists.")
    else:
        os.mkdir(path)
        if args.test_or_train == 'train':
            os.mkdir(path + "/network")


def save_command_line(args):
    cmd = " ".join(sys.argv[:])
    with open(args.output_path + "/command_line.txt", 'w') as f:
        f.write(cmd)


def save(sess, args, output_data):
    logger.info("Saving output data")
    plot.all_figures(args, output_data)
    logger.info("Output data saved to {}".format("TODO"))
