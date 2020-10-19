## @package classifier
#  This is the main file of the software.
import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append("..")
import graph
import dataio
import utils
import argparse
import procedure
import logging as log
import tensorflow as tf
import numpy as np
import datetime
import logging
from tensorflow.python.client import device_lib
from dataio import make_output_dir
tf.compat.v1.disable_v2_behavior()



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
    '--exp_config', default="/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/src/exp_parameters.json",
    help="Json file path for experiment parameters"
)
parser.add_argument(
    '--model_config', default="/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/src/model_parameters.json",
    help="Json file path for model parameters"
)

# args = argument.params
params = parser.parse_args()

args = utils.load_all_params(params.exp_config, params.model_config)

# if the job is NOT submitted with cluster.py, only locally, then
if not args.from_clusterpy:
    args = utils.generate_output_path(args)
    make_output_dir(args, sub_folders=["AUCs", "CAMs", 'CAMs/mean', "wrong_examples", "certains"])
    dataio.save_command_line(args.model_save_dir)

print("Taking in config files: {}\n{}\n{}".format(params.output_path, params.exp_config, params.model_config))
logger = log.getLogger("classifier")
get_available_gpus()

if not args.rand_seed:
    temp_seed = np.random.randint(0, 9999)
    args.rand_seed = temp_seed

np.random.seed(seed=np.int(args.rand_seed))
tf.compat.v1.set_random_seed(np.int(args.rand_seed))

# Get augmentation of the data
if args.if_from_certain and args.train_or_test == 'train':
    if args.distill_old:
        certain_dir = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2020-09-01T21-35-25_Nonex0_factor_0_from-ep_0_from-lout40_data5-theta_0.9-train/certains"
        logger.info("______________________________________________")
        print(certain_dir)
        logger.info("______________________________________________")
        certain_files = dataio.find_files(certain_dir, pattern="*_epoch_{}_*.csv".format(args.from_epoch))
        print("certain_files", certain_files)
        data_tensors, args = dataio.get_data_tensors(args, certain_fns=certain_files)
    else:
        certain_files = dataio.find_files(args.certain_dir, pattern="full_summary*.csv")
        print("certain_files", certain_files)
        data_tensors, args = dataio.get_data_tensors(args, certain_fns=certain_files[0])
else:
    if args.data_mode == "mnist" or args.data_mode == "MNIST":
        data_tensors, args = dataio.get_noisy_mnist_data(args)
    elif args.if_single_runs:
        data_tensors, args = dataio.get_single_ep_training_data_tensors(args)
    else:
        data_tensors, args = dataio.get_data_tensors(args)

print("------------Successfully get data tensors")

graph = graph.get_graph(args, data_tensors)

with tf.compat.v1.Session() as sess:
    output_data = procedure.main_train(sess, args, graph)

if args.if_from_certain:
    print("certain_files", certain_files)