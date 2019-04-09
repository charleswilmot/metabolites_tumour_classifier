## @package argument
#  This package processes the argument passed to the main program.
import argparse
import json
import os
import sys
# sys.path.insert(0, "../")

import logging as log
import tensorflow as tf
import dataio
import datetime
import ipdb

#
PADDING_SIZE = 35
logger = log.getLogger("classifier")


class Params():
    """Class that loads hyperparameters from a json file.
    https://github.com/cs230-stanford/cs230-code-examples/blob/master/tensorflow/vision/model/utils.py
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        # type: (object) -> object
        # self.update(json_path)
        pass
    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path, mode=None):
        """Loads parameters from json file. if specify a modelkey, only load the params under thta modelkey"""
        with open(json_path) as f:
            dicts = json.load(f)
            if not mode:
                self.__dict__.update(dicts)
            elif mode == "train" or mode == "test":
                general_params = dicts["train_or_test"]["general"]
                exp_params = dicts["train_or_test"][mode]
                self.__dict__.update(general_params)
                self.__dict__.update(exp_params)
            else:
                model_params = dicts["model"][mode]
                self.__dict__.update(model_params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def padding(message):
    return " " + "." * (PADDING_SIZE - len(message)) + " "


## Interprets the string passed as an argument to the option --train-test-split-ratio
# @param s the string
def split_ratio(s):
    f = float(s)
    if f < 5:
        logger.critical("At least 5%% of the data must be used for testing (asked {})".format(f))
        raise argparse.ArgumentTypeError("At least 5%% of the data must be used for testing (asked {})".format(f))
    if f > 95:
        logger.critical("At most 95%% of the data can be used for testing (asked {})".format(f))
        raise argparse.ArgumentTypeError("At most 95%% of the data can be used for testing (asked {})".format(f))
    return f


## Interprets the string passed as an argument to the option --train-test-compute-time-ratio
# @param s the string
def compute_ratio(s):
    f = float(s)
    if f < 5:
        logger.critical("At least 5%% of the time must be spent testing (asked {})".format(v))
        raise argparse.ArgumentTypeError("At least 5%% of the time must be spent testing (asked {})".format(v))
    if f > 95:
        logger.critical("At most 95%% of the time can be spent testing (asked {})".format(v))
        raise argparse.ArgumentTypeError("At most 95%% of the time can be spent testing (asked {})".format(v))
    return f


_layer_number_dropout = 1
## Interprets the string passed as an argument to the option --dropout-probs
# @param s the string
def dropout_prob(s):
    global _layer_number_dropout
    f = float(s)
    if f < 0:
        logger.warning("Invalid dropout value. Got {}, setting to 0".format(s))
        return 0
    if f >= 1:
        logger.warning("Invalid dropout value. Got {}, setting to 0.99".format(s))
        return 0.99
    message = "Layer {} dropout probability:".format(_layer_number_dropout)
    pad = padding(message)
    logger.debug(message + pad + s)
    _layer_number_dropout += 1
    return f


_layer_number_batch_norm = 1
## Interprets the string passed as an argument to the option --batch-norms
# @param s the string
def batch_norm(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        logger.critical("Batch normalization got invalid parameter: {}".format(v))
        raise argparse.ArgumentTypeError('Boolean value expected.')
    message = "Layer {} batch normalization:".format(_layer_number_batch_norm)
    pad = padding(message)
    logger.debug(message + pad + s)
    _layer_number_batch_norm += 1


_layer_number_activation = 1
## Interprets the string passed as an argument to the option --activations
# @param s the string
def activation(s):
    if s == 'relu':
        return tf.nn.relu
    elif s == 'lrelu':
        return tf.nn.leaky_relu
    elif s == 'tanh':
        return tf.tanh
    elif s == 'sigmoid':
        return tf.nn.sigmoid
    elif s == 'None':
        return None
    else:
        logger.critical("Activation function not recognized: {}".format(s))
        raise argparse.ArgumentTypeError("Activation function not recognized: {}".format(s))
    message = "Layer {} activation:".format(_layer_number_activation)
    pad = padding(message)
    logger.debug(message + pad + s)
    _layer_number_activation += 1


_layer_number_dim = 0
## Interprets the string passed as an argument to the option --layers-dims
# @param s the string
def layer_dim(s):
    global _layer_number_dim
    i = int(s)
    if i <= 0:
        logger.critical("Layer size must be positive. Got {}".format(s))
        raise argparse.ArgumentTypeError("Layer size must be positive. Got {}".format(s))
    if _layer_number_dim == 0:
        message = "Input size:"
    else:
        message = "Layer {} size:".format(_layer_number_dim)
    pad = padding(message)
    logger.debug(message + pad + s)
    _layer_number_dim += 1
    return i



## Helper function to produce the DEBUG messages during argument parsing
# @param type_constructor python built-in type or interpreter function
# @param message a string containing one placeholder that is printed when parsing the argument
def log_debug_arg(type_constructor, message):
    def f(s):
        r = type_constructor(s)
        pad = padding(message)
        logger.debug(message + pad + s)
        return r
    return f


parser = argparse.ArgumentParser()
parser.add_argument(
    '-v', '--verbose', action='count', default=1,
    help="Verbosity level. Use -v, -vv, -vvv -vvvv."
)
parser.add_argument(
    '-resplit_data', default="../data",
    help="Verbosity level. Use -v, -vv, -vvv -vvvv."
)
parser.add_argument(
    '-T', dest='separator', action='store_true',
    help="Separator in case the option before train/test is a list. See https://bugs.python.org/issue9338 ."
)
parser.add_argument(
    '-exp_config', default="./exp_parameters.json",
    help="Json file path for experiment parameters"
)
parser.add_argument(
    '-model_config', default="./model_parameters.json",
    help="Json file path for model parameters"
)
parser.add_argument(
    '-A', '--activations', type=activation, action='store', nargs='+',
    default=['lrelu', 'lrelu', 'lrelu', 'lrelu', 'sigmoid'],
    help="Activation functions for every layer. Taken in 'lrelu', 'relu', 'sigmoid', 'tanh'"
)

subparsers = parser.add_subparsers(dest="test_or_train")

train_parser = subparsers.add_parser("train")
# train_parser.add_argument(
#     'output_path', metavar='OUTPUT',
#     type=log_debug_arg(str, "Output path:"),
#     nargs='?', default='../results',
#     help="Path to the output data."
# )

train_parser.add_argument(
    '--restore_from', type=log_debug_arg(str, "Restore model from:"),
    nargs='?', default= None,
    help="Path to a previously trained model."
)

train_parser.add_argument(
    '-a', '--data-augmentation', action='store_true',
    help="Set this flag if the algorithm should perform data augmentation."
)


test_parser = subparsers.add_parser("test")
test_parser.add_argument(
    'restore_from', metavar='MODEL',
    type=log_debug_arg(str, "Restore model from:"),
    help="Path to a previously trained model."
)

# test_parser.add_argument(
#     'output_path', metavar='OUTPUT',
#     type=log_debug_arg(str, "Output path:"),
#     nargs='?', default='./',
#     help="Path to the output data."
# )

test_parser.add_argument(
    '-x', '--data-not-labeled', action='store_true',
    help="Set this flag if the data you want to classify is not labeled."
)


## Read arguments once to get the verbosity level
args = parser.parse_args()
_layer_number_dim = 0
_layer_number_dropout = 1
_layer_number_batch_norm = 1
_layer_number_activation = 1

# Re-read the arguments after the verbosity has been set correctly
args = parser.parse_args()

## Load experiment parameters and model parameters
json_path = args.exp_config  # exp_param stores general training params
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = Params(json_path)
params.update(json_path, mode=args.test_or_train)

# load model specific parameters
json_path = args.model_config
assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
params.update(json_path, mode=params.model_name) # update params with the model configuration

# specify some params
time_str = '{0:%Y-%m-%dT%H-%M-%S-}'.format(datetime.datetime.now())

params.output_path = os.path.join(params.output_path,
                                time_str + "data-{}-class-{}-{}-{}".format(os.path.basename(params.input_data[0:-4]), params.num_classes, params.model_name, args.test_or_train))
params.model_save_dir = os.path.join(params.output_path, "network")
params.resplit_data = args.resplit_data
params.restore_from = args.restore_from
params.test_or_train = args.test_or_train
params.resume_training = (args.restore_from != None)


# Make the output directory
dataio.make_output_dir(params)


# Verbosity level:
level = 50 - (args.verbose * 10) + 1
logger.setLevel(level)
ch = log.StreamHandler()
ch.setLevel(level)
formatter = log.Formatter(" " * 12 + '%(message)s\r' + '[\033[1m%(levelname)s\033[0m]')
ch.setFormatter(formatter)
fh = log.FileHandler(params.output_path + "/summary.log")
fh.setLevel(1)
formatter = log.Formatter('[%(levelname)s] %(message)s\r')
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)
