## @package classifier
#  This is the main file of the software.

# import matplotlib
# matplotlib.use('Agg')
import sys
sys.path.append("..")
import graph
import dataio
import argument
import procedure
import logging as log
import tensorflow as tf
import numpy as np
import logging
from tensorflow.python.client import device_lib
tf.compat.v1.disable_v2_behavior()

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    logging.info("-------------------Available GPU-----------------------")
    logging.info([x.name for x in local_device_protos if x.device_type == 'GPU'])


logger = log.getLogger("classifier")
args = argument.params
dataio.save_command_line(args)
get_available_gpus()

if not args.randseed:
    temp_seed = np.random.randint(0, 9999)
    args.randseed = temp_seed

np.random.seed(seed=np.int(args.randseed))
tf.compat.v1.set_random_seed(np.int(args.randseed))

# C:\Users\LDY\Desktop\metabolites-0301\metabolites_tumour_classifier\results
# /home/epilepsy-data/data/metabolites/results/

# Get augmentation of the data
if args.if_from_certain and args.test_or_train == 'train':
    # certain_dir ="/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2020-09-01T21-35-28-None-meanx0-factor-0-from-ep-0-from-lout40-data5-theta-0.95-train/certains"
    logger.info("______________________________________________")
    print(args.certain_dir)
    logger.info("______________________________________________")
    certain_files = dataio.find_files(args.certain_dir, pattern="full_summary*.csv")
    print("certain_files", certain_files)
    data_tensors, args = dataio.get_data_tensors(args, certain_fns=certain_files[0])
else:
    if args.data_mode == "mnist" or args.data_mode == "MNIST":
        data_tensors, args = dataio.get_noisy_mnist_data(args)
    else:
        data_tensors, args = dataio.get_data_tensors(args)

print("------------Successfully get data tensors")
print("---args.if_single_runs--: ", args.if_single_runs)


graph = graph.get_graph(args, data_tensors)


with tf.compat.v1.Session() as sess:
    output_data = procedure.main_train(sess, args, graph)

print("args.seed", args.seed)
if args.if_from_certain:
    print("certain_files", certain_files)