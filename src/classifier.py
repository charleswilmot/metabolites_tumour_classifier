## @package classifier
#  This is the main file of the software.
#
#  Run python3 classifier.py -h for more details.
import graph
import dataio
import argument
import procedure
import logging as log
import tensorflow as tf


logger = log.getLogger("classifier")
args = argument.args
graph = graph.get_graph(args)
input_data = dataio.get_data(args)
with tf.Session() as sess:
    output_data = procedure.run(sess, args, graph, input_data)
    dataio.save(sess, args, output_data, graph)
logger.info("Success")
