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
tf.compat.v1.disable_v2_behavior()

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    logging.info("-------------------Available GPU-----------------------")
    logging.info([x.name for x in local_device_protos if x.device_type == 'GPU'])


logger = log.getLogger("classifier")
args = argument.params
logger.info("------------args.aug_method", args.aug_method)
logger.info("------------args.theta_thr", args.theta_thr)
dataio.save_command_line(args)
get_available_gpus()

if not args.seed:
    temp_seed = np.random.randint(0, 9999)
    args.seed = temp_seed

    
np.random.seed(seed=args.seed)
tf.compat.v1.set_random_seed(args.seed)

# if args.seed != 2594:   # every time change the random seed should save the data again
## get leave-out train and test sets: dataio.split_data_for_lout_val(args)\ dataio.split_data_for_val(args)

# Get augmentation of the data
if args.if_from_certain:
    certain_dir ="/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-10-30T15-18-21-data-lout40-data5-1d-class2-Res_ECG_CAM-step1-collect-aug_ops_meanx0-0-train-0.889/certains"
    logger.info("______________________________________________")
    print(certain_dir)
    logger.info("______________________________________________")
    certain_files = dataio.find_files(certain_dir, pattern="*_epoch_{}_*.csv".format(args.from_epoch))
    print("certain_files", certain_files)
    data_tensors, args = dataio.get_data_tensors(args, certain_fns=certain_files)
else:
    data_tensors, args = dataio.get_data_tensors(args)

logger.info("------------Successfully get data tensors")


graph = graph.get_graph(args, data_tensors)

with tf.compat.v1.Session() as sess:
    output_data = procedure.main_train(sess, args, graph)
logger.info("Success")
print("args.seed", args.seed)
if args.if_from_certain:
    print("certain_files", certain_files)