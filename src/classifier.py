## @package classifier
#  This is the main file of the software.
#
#  For more details, try:\n
#  python3 classifier.py -h\n
#  python3 classifier.py train -h\n
#  python3 classifier.py test -h\n
#  example: python3 classifier.py -b 20 -s 17845 train ../data/ ../results/ -e 100
import matplotlib
matplotlib.use('Agg')
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


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    logging.info("-------------------Available GPU-----------------------")
    logging.info([x.name for x in local_device_protos if x.device_type == 'GPU'])


logger = log.getLogger("classifier")
args = argument.params
dataio.save_command_line(args)
get_available_gpus()
# if args.seed is not None:
#     np.random.seed(seed=args.seed)
#     tf.compat.v1.set_random_seed(args.seed)

# if args.seed != 2594:   # every time change the random seed should save the data again
## get leave-out train and test sets: dataio.split_data_for_lout_val(args)\ dataio.split_data_for_val(args)
# Get augmentation of the data

data_tensors, args = dataio.get_data_tensors(args)

graph = graph.get_graph(args, data_tensors)
with tf.Session() as sess:
    output_data = procedure.main_train(sess, args, graph)
logger.info("Success")
