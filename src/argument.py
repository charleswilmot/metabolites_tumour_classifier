## @package argument
#  This package processes the argument passed to the main program.
import argparse
import json
import os
import sys
# sys.path.insert(0, "../")

import logging as log
import tensorflow as tf
from utils import Params
import dataio
import datetime
import ipdb

#
PADDING_SIZE = 35
logger = log.getLogger("classifier")


# class Params():
#     """Class that loads hyperparameters from a json file.
#     https://github.com/cs230-stanford/cs230-code-examples/blob/master/tensorflow/vision/model/utils.py
#     Example:
#     ```
#     params = Params(json_path)
#     print(params.learning_rate)
#     params.learning_rate = 0.5  # change the value of learning_rate in params
#     ```
#     """
#
#     def __init__(self, json_path):
#         # type: (object) -> object
#         # self.update(json_path)
#         pass
#     def save(self, json_path):
#         """Saves parameters to json file"""
#         with open(json_path, 'w') as f:
#             json.dump(self.__dict__, f, indent=4)
#
#     def update(self, json_path, mode=None):
#         """Loads parameters from json file. if specify a modelkey, only load the params under thta modelkey"""
#         with open(json_path) as f:
#             dicts = json.load(f)
#             if not mode:
#                 self.__dict__.update(dicts)
#             elif mode == "train" or mode == "test":
#                 general_params = dicts["train_or_test"]["general"]
#                 exp_params = dicts["train_or_test"][mode]
#                 self.__dict__.update(general_params)
#                 self.__dict__.update(exp_params)
#             else:
#                 model_params = dicts["model"][mode]
#                 self.__dict__.update(model_params)
#     def update(self, json_path, mode=None):
#         """Loads parameters from json file. if specify a modelkey, only load the params under thta modelkey"""
#         with open(json_path) as f:
#             dicts = json.load(f)
#             if not mode:
#                 self.__dict__.update(dicts)
#             elif mode == "train" or mode == "test":
#                 # general_params = dicts["train_or_test"]["general"]
#                 general_params = dicts["general"]
#                 exp_params = dicts[mode]
#                 self.__dict__.update(general_params)
#                 self.__dict__.update(exp_params)
#             else:
#                 model_params = dicts["model"][mode]
#                 self.__dict__.update(model_params)
#
#     @property
#     def dict(self):
#         """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
#         return self.__dict__


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


## Interprets the string passed as an argument to the option --number-of-epochs
# @param s the string
def nepoch(s):
    f = float(s)
    if f < 0.05 and f != -1:
        logger.critical("Not enough epochs (asked {}, minimum is 0.05)".format(s))
        raise argparse.ArgumentTypeError("Not enough epochs (asked {}, minimum is 0.05)".format(s))
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


## custom activation function
def lrelu(x):
    a = 1 / 3
    return tf.nn.relu(x) * (1 - a) + a * x


_layer_number_activation = 1
## Interprets the string passed as an argument to the option --activations
# @param s the string
def activation(s):
    if s == 'relu':
        return tf.nn.relu
    elif s == 'lrelu':
        return lrelu
    elif s == 'tanh':
        return tf.tanh
    elif s == 'sigmoid':
        return tf.nn.sigmoid
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
# parser.add_argument(
#     '-v', '--verbose', action='count', default=1,
#     help="Verbosity level. Use -v, -vv, -vvv -vvvv."
# )
# parser.add_argument(
#     '-T', dest='separator', action='store_true',
#     help="Separator in case the option before train/test is a list. See https://bugs.python.org/issue9338 ."
# )
parser.add_argument(
    '--output_path', default="./exp_parameters.json",
    help="Json file path for experiment parameters"
)
parser.add_argument(
    '--exp_config', default="./exp_parameters.json",
    help="Json file path for experiment parameters"
)
parser.add_argument(
    '--model_config', default="./model_parameters.json",
    help="Json file path for model parameters"
)


def load_all_params(args):
# args = parser.parse_args()
    json_path = args.exp_config  # exp_param stores general training params
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    args = Params(json_path)
    args.update(json_path, mode=args.test_or_train)

    # load model specific parameters
    json_path = args.model_config
    assert os.path.isfile(json_path), "No json file found at {}".format(json_path)

    args.update(json_path, mode=args.model_name) # update params with the model configuration
# args.from_clusterpy = args.from_clusterpy
# args.certain_dir = args.certain_dir
#
#
# if args.data_mode == "mnist" or args.data_mode == "MNIST":
#     args.width = 28
#     args.height = 28
#     args.data_source = "mnist"
#     args.noise_ratio = args.noise_ratio
# elif args.data_mode == "metabolite" or args.data_mode == "metabolites":
#     args.width = 1
#     args.height = 288
#     # TODO, cluster and param.json all give this parameter
#
# if not args.from_clusterpy:

#     print("Not run from cluster.py params.input data dir: ", args.input_data)
#     if args.data_mode == "metabolite":
#         args.data_source = os.path.basename(args.input_data).split("_")[-1].split(".")[0]
#     elif args.data_mode == "mnist" or args.data_mode == "MNIST":
#         args.data_source = "mnist"
#
#     # specify some params
#     time_str = '{0:%Y-%m-%dT%H-%M-%S-}'.format(datetime.datetime.now())
#     if args.restore_from is None:  # and args.output_path is None:  #cluster.py
#         postfix = "100rns-" + args.test_or_train if args.if_single_runs else args.test_or_train
#         args.output_path = os.path.join(args.output_root,
#                                           "{}-{}-{}x{}-factor-{}-from-{}-certain{}-theta-{}-s{}-{}".format(
#                                               time_str, args.model_name, args.aug_method, args.aug_folds,
#                                               args.aug_scale, args.data_source,
#                                               args.if_from_certain, args.theta_thr,
#                                               args.randseed, postfix))
#         # params.postfix = "-test"
#     # elif args.restore_from is None and args.output_path is not None:
#     #     params.output_path = args.output_path
#     elif args.restore_from is not None:  # restore a model
#         args.output_path = os.path.dirname(args.restore_from) + "-on-{}-{}".format(args.data_source, "test")
#         args.postfix = "-test"
#     dataio.make_output_dir(args, sub_folders=["AUCs", "CAMs", 'CAMs/mean', "wrong_examples", "certains"])
# else:
#     print("Run from cluster.py args.input data dir: ", args.input_data)
#     if args.data_mode == "metabolite":
#         args.data_source = os.path.basename(args.input_data).split("_")[-1].split(".")[0]
#     elif args.data_mode == "mnist" or args.data_mode == "MNIST":
#         args.data_source = "mnist"
#     args.output_path = args.output_path
#
# # params.resplit_data = args.resplit_data
# args.restore_from = args.restore_from
# args.test_or_train = args.test_or_train
# args.resume_training = (args.restore_from != None)
# args.randseed = args.randseed
# args.if_single_runs = False
# print("argument.py, params.if_single_runs: ", args.if_single_runs)
#
# args.model_save_dir = os.path.join(args.output_path, "network")
# print("output dir: ", args.output_path)
#
#
# if args.test_or_train == "test":
#     args.if_from_certain = False
#     args.if_save_certain = False
# elif args.test_or_train == "train":
#     args.aug_scale = args.aug_scale
#     args.aug_method = args.aug_method
#     args.aug_folds = args.aug_folds
#     args.theta_thr = args.theta_thr

# Verbosity level:
# level = 50 - (args.verbose * 10) + 1
# logger.setLevel(level)
# ch = log.StreamHandler()
# ch.setLevel(level)
# formatter = log.Formatter(" " * 12 + '%(message)s\r' + '[\033[1m%(levelname)s\033[0m]')
# ch.setFormatter(formatter)
# dataio.make_output_dir(args)
# fh = log.FileHandler(args.output_path + "/summary.log")
# fh.setLevel(1)
# formatter = log.Formatter('[%(levelname)s] %(message)s\r')
# fh.setFormatter(formatter)
# logger.addHandler(ch)
# ## Re-read the arguments after the verbosity has been set correctly
# args = parser.parse_args()
