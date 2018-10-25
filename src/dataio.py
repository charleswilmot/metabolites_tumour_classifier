## @package dataio
#  This package handles the interface with the hard drive.
#
#  It can in particular read and write matlab or python matrices.
import numpy as np
import logging as log


logger = log.getLogger("classifier")


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
    logger.info("Output data saved to {}".format("TODO"))


def split(data):
    return data[:, :97], data[:, 97:]


def shuffle(data):
    logger.debug("Shuffling data")
    return data
