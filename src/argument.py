## @package argument
#  This package processes the argument passed to the main program.
import argparse
import json
import os

import logging as log
import tensorflow as tf

import dataio

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
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path, model_key=None):
        """Loads parameters from json file. if specify a modelkey, only load the params under thta modelkey"""
        with open(json_path) as f:
            params = json.load(f)
            if not model_key:
                self.__dict__.update(params)
            else:
                model_params = params["model"][model_key]
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
    '-T', dest='separator', action='store_true',
    help="Separator in case the option before train/test is a list. See https://bugs.python.org/issue9338 ."
    )

parser.add_argument(
    '-exp_config', default='./exp_parameters.json',
    help="The jason file storing parameters for experiment."
    )

parser.add_argument(
    '-model_config', default='./model_parameters.json',
    help="The jason file storing parameters for network structure."
    )

parser.add_argument(
    '-model_name', default='MLP',
    help="The jason file storing parameters for network structure."
    )

subparsers = parser.add_subparsers(dest="test_or_train")
test_parser = subparsers.add_parser('test')
test_parser.add_argument(
    'model_path', metavar='MODEL',
    type=log_debug_arg(str, "Model path:"),
    help="Path to a previously trained model."
    )

test_parser.add_argument(
    'input_data', metavar='INPUT',
    type=log_debug_arg(str, "Data path:"),
    help="Path to the input data file."
    )

test_parser.add_argument(
    'output_path', metavar='OUTPUT',
    type=log_debug_arg(str, "Output path:"),
    nargs='?', default='./',
    help="Path to the output data."
    )

test_parser.add_argument(
    '-x', '--data-not-labeled', action='store_true',
    help="Set this flag if the data you want to classify is not labeled."
    )

train_parser = subparsers.add_parser('train')
train_parser.add_argument(
    'input_data', metavar='INPUT',
    type=log_debug_arg(str, "Data path:"),
    help="Path to the input data file."
    )

train_parser.add_argument(
    'output_path', metavar='OUTPUT',
    type=log_debug_arg(str, "Output path:"),
    default='./',
    help="Path to the output data."
    )

train_parser.add_argument(
    '-a', '--data-augmentation', action='store_true',
    help="Set this flag if the algorithm should perform data augmentation."
    )

train_parser.add_argument(
    '-S', '--train-test-split-ratio',
    type=log_debug_arg(split_ratio, "Train/Test split ratio:"),
    action='store', default='10', metavar='RATIO',
    help="Determines which percentage of the training data should be used for testing. It can be set to 0 percent, in which case the software does not produce any plot of the training."
    )

train_parser.add_argument(
    '-l', '--learning-rate',
    type=log_debug_arg(float, "Learning rate:"),
    action='store', default=1e-3,
    help="Set the learning rate."
    )

train_parser.add_argument(
    '-L', '--loss-type',
    type=log_debug_arg(str, "Loss type:"),
    action='store', default='softmax_ce', choices=['mse', 'rmse', 'softmax_ce'],
    help="Choses a loss type. Can be any of 'mse', 'rmse', 'softmax_ce'"
    )

train_parser.add_argument(
    '-O', '--optimizer-type',
    type=log_debug_arg(str, "Optimizer type:"),
    action='store', default='adam', choices=['adam', 'rmsprop', 'gradient_descent'],
    help="Choses an optimizer type. Can be any of 'adam', 'rmsprop', 'gradient_descent'"
    )

train_parser.add_argument(
    '-e', '--number-of-epochs',
    type=log_debug_arg(int, "Number of epoch:"),
    action='store', default='-1', metavar='N',
    help="Determines the amount of times the algorithm will see each of the training examples. If this flag is not set, the algorithm trains until convergence."
    )

train_parser.add_argument(
    '-t', '--test-every',
    type=log_debug_arg(int, "Number train batches:"),
    action='store', default=50, metavar='N_BATCHES',
    help="Determines which amount of time should be spent training the network, and which amount of time should be spent testing. The more time is spent testing, the better the plot will look."
    )

train_parser.add_argument(
    '-r', '--resume-training',
    type=log_debug_arg(str, "Model path:"),
    action='store', default=None, metavar='PATH',
    help="Path to a pretrained model. Resumes the training from the last checkpoint."
    )

## Read arguments once to get the verbosity level
args = parser.parse_args()
_layer_number_dim = 0
_layer_number_dropout = 1
_layer_number_batch_norm = 1
_layer_number_activation = 1

## Load experiment parameters and model parameters
json_path = args.exp_config  # exp_param stores general training params
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = Params(json_path)

# specify some params
params.model_path = args.model_path
params.model_name = args.model_name

## Verbosity level:
level = 50 - (args.verbose * 10) + 1
logger.setLevel(level)
ch = log.StreamHandler()
ch.setLevel(level)
formatter = log.Formatter(" " * 12 + '%(message)s\r' + '[\033[1m%(levelname)s\033[0m]')
ch.setFormatter(formatter)
dataio.make_output_dir(args)
fh = log.FileHandler(args.output_path + "/summary.log")
fh.setLevel(1)
formatter = log.Formatter('[%(levelname)s] %(message)s\r')
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)
## Re-read the arguments after the verbosity has been set correctly
args = parser.parse_args()
