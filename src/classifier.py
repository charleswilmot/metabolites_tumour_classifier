## @package classifier
#  This is the main file of the software.
# import matplotlib
# matplotlib.use('Agg')
import sys
sys.path.append("..")
import graph
import os
import ipdb
import dataio as dataio
import utils
import argparse
import procedure
import logging as log
import tensorflow as tf
import numpy as np
import datetime
import logging
from tensorflow.python.client import device_lib
tf.compat.v1.disable_v2_behavior()
import tracemalloc

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("Using GPU, set memory growth to True")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    logging.info("-------------------Available GPU-----------------------")
    logging.info([x.name for x in local_device_protos if x.device_type == 'GPU'])

parser = argparse.ArgumentParser()
parser.add_argument(
    '--output_path', default=None,
    help="output dir"
)
parser.add_argument(
    '--exp_config', default="./exp_parameters.yaml",
    help="Json file path for experiment parameters"
)
parser.add_argument(
    '--model_config', default="./model_parameters.yaml",
    help="Json file path for model parameters"
)
logger = log.getLogger("classifier")

# track the memory usage while running the script
tracemalloc.start()

params = parser.parse_args()

args = utils.load_all_params_yaml(params.exp_config, params.model_config)
args.rand_seed = np.random.randint(0, 9999)

# switch the directory in different platforms
if args.platform == "laptop":
    args.root_of_root = "C:/Users/LDY/Desktop/metabolites-0301/metabolites_tumour_classifier"
    if args.data_mode == "metabolite" or args.data_mode == "metabolites":
        args.data_root = os.path.join(args.root_of_root, "data", "20190325")
    elif args.data_mode == "mnist" or args.data_mode == "MNIST":
        args.data_root = os.path.join(args.root_of_root, "data", "noisy_mnist")
    args.input_data = os.path.join(args.data_root, args.input_data)
    args.output_root = os.path.join(args.root_of_root, "results")


elif args.platform == "FIAS":
    args.root_of_root = "/home/epilepsy-data/data/metabolites"
    if args.data_mode == "metabolite" or args.data_mode == "metabolites":
        args.data_root = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data"
    elif args.data_mode == "mnist" or args.data_mode == "MNIST":
        args.data_root = "/home/epilepsy-data/data/metabolites/noisy-MNIST"
    args.input_data = os.path.join(args.data_root, args.input_data)
    args.output_root = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review"

# if the job is NOT submitted with cluster.py, only locally, then
if not args.from_clusterpy:
    args = utils.generate_output_path(args)
    dataio.make_output_dir(args, sub_folders=["AUCs", "CAMs", 'CAMs/mean', "wrong_examples", "certains"])
    dataio.save_command_line(args.model_save_dir)

logger.info("Taking in config files: {}\n{}\n{}".format(params.output_path, params.exp_config, params.model_config))
get_available_gpus()

np.random.seed(seed=np.int(args.rand_seed))
tf.compat.v1.set_random_seed(np.int(args.rand_seed))

## use to generate noisy mnist
# dataio.generate_mnist_with_noise(args)

#
if args.if_single_runs:  ## 100 single-epoch training
    args.if_save_certain = True
    if args.data_mode == "metabolites" or args.data_mode == "metabolite":
        data_tensors, args = dataio.get_single_ep_training_data_tensors(args)
    elif args.data_mode == "mnist" or args.data_mode == "MNIST":
        args.num_classes = 10
        data_tensors, args = dataio.get_single_ep_training_data_tensors(args)

elif args.if_from_certain and args.train_or_test == "train":  # use distillation to augment data
    certain_files = dataio.find_files(args.certain_dir,
                                      pattern="full_summary*.csv")
    if args.data_mode == "metabolites" or args.data_mode == "metabolite":
        data_tensors, args = dataio.get_data_tensors(args,
                                                     certain_fns=certain_files[0])
    elif args.data_mode == "mnist" or args.data_mode == "MNIST":
        args.num_classes = 10
        data_tensors, args = dataio.get_noisy_mnist_data_tensors(args, certain_fns=certain_files[0])
else:                # normal training
    if args.data_mode == "metabolites" or args.data_mode == "metabolite":
        data_tensors, args = dataio.get_data_tensors(args)
    elif args.data_mode == "mnist" or args.data_mode == "MNIST":
        args.num_classes = 10
        data_tensors, args = dataio.get_noisy_mnist_data_tensors(args)

logger.info("------------Successfully get data tensors----------------------")

graph = graph.get_graph(args, data_tensors)

with tf.compat.v1.Session() as sess:
    output_data = procedure.main_train(sess, args, graph)

snapshot = tracemalloc.take_snapshot()
utils.display_top(snapshot, key_type='lineno', limit=20)

if args.train_or_test == "test":
    dataio.rename_test_fold_on_the_fly(args)