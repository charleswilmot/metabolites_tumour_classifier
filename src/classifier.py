## @package classifier
#  This is the main file of the software.
#
#  For more details, try:\n
#  python3 classifier.py -h\n
#  python3 classifier.py train -h\n
#  python3 classifier.py test -h\n
#  example: python3 classifier.py -b 20 -s 17845 train ../data/ ../results/ -e 100
import graph
import dataio
import argument
import procedure
import logging as log
import tensorflow as tf


logger = log.getLogger("classifier")
args = argument.args
data_tensors = dataio.get_data_tensors(args)
# data_tensors = {
#     "train_features": wlvnfkjvn,
#     "train_labels": wlvnfkjvn,
#     "test_features": wlvnfkjvn,
#     "test_labels": wlvnfkjvn
#     "iter_train": jlkjpoi
#     "iter_test": jlkjpoi
# }
graph, iters = graph.get_graph(args, data_tensors)
with tf.Session() as sess:
    for key in iters.keys(): # init the iterator for the dataset
        sess.run(iters[key].initializer)
    output_data = procedure.run(sess, args, graph)
    dataio.save(sess, args, output_data, graph)
logger.info("Success")
