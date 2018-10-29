## @package classifier
#  This is the main file of the software.
#
#  For more details, try:\n
#  python3 classifier.py -h\n
#  python3 classifier.py train -h\n
#  python3 classifier.py test -h\n
import graph
import dataio
import argument
import procedure
import logging as log
import tensorflow as tf
import numpy as np


logger = log.getLogger("classifier")
args = argument.args
dataio.save_command_line(args)
if args.seed is not None:
    np.random.seed(seed=args.seed)
    tf.set_random_seed(args.seed)
data_tensors = dataio.get_data_tensors(args)
graph = graph.get_graph(args, data_tensors)
with tf.Session() as sess:
    for key in iters.keys(): # init the iterator for the dataset
        if "iter" in key:
            sess.run(graph[key].initializer)
    output_data = procedure.run(sess, args, graph)
    dataio.save(sess, args, output_data, graph)
logger.info("Success")
