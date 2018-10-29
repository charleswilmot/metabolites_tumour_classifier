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
import fnmatch
import random
import ipdb


logger = log.getLogger("classifier")

## Find all the files in one directory with pattern in the filenames and get the total bytesize for train and test split
# @param directory str the directory of the files
# @param pattern str the file name pattern to match
# @return files list all the files match the file name pattern
# @return total_bytes int the total size of all the files in bytes
def find_files(directory, pattern='*.csv'):
    files = []
    total_bytes = 0
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
            total_bytes += os.path.getsize(os.path.join(root, filename))

    # random.shuffle(files)
    return files, total_bytes

## Map function to decode the .csv file in TextLineDataset
# @param line object in TextLineDataset
# @return features decoded features
# @return label int decoded label
def decode_csv(line):
    defaults = [[0.0]]*289   ## 301 depends on the real shape of the dataset
    csv_row = tf.decode_csv(line, record_defaults=defaults)#dydfias032122

    label = tf.cast(csv_row[0], tf.int64)  # also depend on the final position
    del csv_row[0]   #delete the label
    features = tf.stack(csv_row)

    return label, features

## Get batches of data in tf.dataset
# @param args the arguments passed to the software
# @key pattern str the file name pattern to match
# @key itemsize 8 the bytesize of the element
# @key data_dim int the number of features + 1 for label
def get_data_tensors(args, pattern='rand*.csv', itemsize=8, data_dim=289):
    data = {}
    filenames, total_bytes= find_files(args.input_dir, pattern=pattern)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename).skip(0).map(decode_csv))   ## skip the header, depend on wheather there is any

    # train and test split
    num_train = tf.cast(0.01 * 80 * total_bytes / itemsize*data_dim, tf.int64) # args train_portion
    train_ds = dataset.take(num_train).repeat(np.int(args.number_of_epochs)).shuffle(buffer_size=10000).batch(args.maximum_batch_size)
    test_ds = dataset.skip(num_train).repeat(np.int(args.number_of_epochs)).shuffle(buffer_size=10000).batch(args.maximum_batch_size)

    train_ds = train_ds.prefetch(1)
    iter_train = train_ds.make_initializable_iterator()
    batch_train = iter_train.get_next()
    # ipdb.set_trace()
    data["train_features"], data["train_labels"] = batch_train[0], batch_train[1:]
    data["train_iter"] = iter_train

    test_ds = test_ds.prefetch(1)
    iter_test = test_ds.make_initializable_iterator()
    batch_test = iter_test.get_next()
    data["test_features"], data["test_labels"] = batch_test[0], batch_test[1:]
    data["test_iter"] = iter_test
    logger.info("Input data read")

    return data

def get_data(args):
    data = {}   # train and test are already keywords. iftrain?
    logger.info("Reading input data")
    values = np.loadtxt(args.input_dir, delimiter=',')   # original shape (288, 77), is 77 samples each with 288 dimensions
    values = values.T  #
    train_v, test_v = train_test_split(values, test_size=0.1)   # 0.1 args.test_ratio: test / whole
    data["train_data"], data["train_lb"] = np.int(train_v[:, 0]), train_v[:, 1:]  # first col is label
    data["test_data"], data["test_lb"] = np.int(test_v[:, 0]), test_v[:, 1:]   # 0.1 args.test_ratio: test / whole

    logger.info("Input data read")
    return data


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


def get_data(args):
    data = {}
    logger.info("Reading input data")
    data["test"] = np.random.uniform(size=(10000 * int(args.train_test_split_ratio), 100))
    if args.test_or_train == "train":
        data["train"] = np.random.uniform(size=(10000 * (100 - int(args.train_test_split_ratio)), 100))
    logger.info("Input data read")
    return data


def save(sess, args, output_data, graph):
    logger.info("Saving output data")
    plot.all_figures(args, output_data)
    logger.info("Output data saved to {}".format("TODO"))


def split(data):
    return data[:, :97], data[:, 97:]


def shuffle(data):
    logger.debug("Shuffling data")
    return data
