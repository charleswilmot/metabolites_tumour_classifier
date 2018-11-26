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
import random

logger = log.getLogger("classifier")


def split_data_for_val(args):
    """
    Split the original data into train_test set and validate set
    :param args:
    :return: save two .mat files
    """
    mat = scipy.io.loadmat(args.input_data)["DATA"]
    spectra = mat[:, 2:]
    labels = mat[:, 1]
    val_data = {}
    validate_features = np.empty((0, 288))
    validate_labels = np.empty((0))
    train_test_data = {}
    train_test_features = np.empty((0, 288))
    train_test_labels = np.empty((0))
    for id in range(args.num_classes):
        # pick 10% of each class for validating
        indices = np.where(labels == id)[0]
        random.shuffle(indices)
        val_inds = indices[0: np.int(0.1 * len(indices))]
        train_test_inds = indices[np.int(0.1 * len(indices)): ]
        train_test_features = np.vstack((train_test_features, spectra[train_test_inds, :]))
        validate_features = np.vstack((validate_features, spectra[val_inds, :]))
        train_test_labels = np.append(train_test_labels, labels[train_test_inds])
        validate_labels = np.append(validate_labels, labels[val_inds])

    val_data["features"] = validate_features
    val_data["labels"] = validate_labels
    train_test_data["features"] = train_test_features
    train_test_data["labels"] = train_test_labels
    scipy.io.savemat(os.path.dirname(args.input_data) + '/2class_val_data.mat', val_data)
    scipy.io.savemat(os.path.dirname(args.input_data) + '/2class_train_test_data.mat', train_test_data)


def get_data(args):
    """
    Load self_saved data. A dict, data["features"], data["labels"]. See the save function in split_data_for_val()
    :param args: Param object with path to the data
    :return:
    """
    spectra = scipy.io.loadmat(args.input_data)["features"]
    labels = scipy.io.loadmat(args.input_data)["labels"]

    return spectra.astype(np.float32), np.squeeze(labels).astype(np.int32)


## Get batches of data in tf.dataset
# @param args the arguments passed to the software
def get_data_tensors(args, spectra, labels):

    data = {}
    spectra, labels = tf.constant(spectra), tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((spectra, labels)).shuffle(buffer_size=10000)
    # train and test split
    num_train = int(((100 - args.test_ratio) * spectra.get_shape().as_list()[0]) // 100)
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
        os.makedirs(path)
        if args.test_or_train == 'train':
            os.makedirs(args.model_save_dir)


def save_command_line(args):
    cmd = " ".join(sys.argv[:])
    with open(args.output_path + "/command_line.txt", 'w') as f:
        f.write(cmd)


def save_plots(sess, args, output_data, training=False):
    logger.info("Saving output data")
    plot.all_figures(args, output_data, training=training)
    logger.info("Output data saved to {}".format("TODO"))


def load_model(saver, sess, save_dir):
    ckpt = tf.train.get_checkpoint_state(save_dir)
    if ckpt:
        logger.info('Checkpoint found: {}'.format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        logger.info('  Global step was: {}'.format(global_step))
        logger.info('  Restoring...')
        saver.restore(sess, ckpt.model_checkpoint_path)
        logger.info(' Done.')
        return global_step
    else:
        logger.info(' No checkpoint found.')
        return None


def save_my_model(saver, sess, save_dir, step, name=None):
    """
    Save the model under current step into save_dir
    :param saver: tf.Saver
    :param sess: tf.Session
    :param save_dir: str, directory to save the model
    :param step: int, current training step
    :param name: if specify a name, then save with this name
    :return:
    """
    model_name = 'model.ckpt'
    if not name:
        checkpoint_path = os.path.join(save_dir, model_name)
    else:
        checkpoint_path = os.path.join(save_dir, name + model_name)
    logger.info('Saving checkpoint to {} ...'.format(save_dir))
    sys.stdout.flush()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saver.save(sess, checkpoint_path, global_step=step)
    logger.info('Done.')