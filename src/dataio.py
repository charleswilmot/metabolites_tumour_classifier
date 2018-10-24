## @package dataio
#  This package handles the interface with the hard drive.
#
#  It can in particular read and write matlab or python matrices.
import numpy as np
import logging as log


logger = log.getLogger("classifier")


def get_data(args):
    logger.info("Reading input data")
    # input_data["train"]
    # input_data["test"]
    logger.info("Input data read")


def save(args, output_data, graph):
    logger.info("Saving output data")
    logger.info("Output data daved to {}".format("TODO"))
