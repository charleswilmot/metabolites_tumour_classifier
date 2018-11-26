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
import numpy as np


logger = log.getLogger("classifier")
args = argument.params
dataio.save_command_line(args)

if args.seed is not None:
    np.random.seed(seed=args.seed)
    tf.set_random_seed(args.seed)

# if args.seed != 2594:   # every time change the random seed should save the data again
# dataio.split_data_for_val(args)

spectra, labels = dataio.get_data(args)
data_tensors = dataio.get_data_tensors(args, spectra, labels)
graph = graph.get_graph(args, data_tensors)
with tf.Session() as sess:
    procedure.initialize(sess, graph, args.test_or_train == 'test')
    output_data = procedure.run(sess, args, graph)
    if args.test_or_train == "train":
        dataio.save_plots(sess, args, output_data, training=True)
    else:
        dataio.save_plots(sess, args, output_data, training=False)
logger.info("Success")
