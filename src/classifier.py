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
args = argument.args
dataio.save_command_line(args)
if args.seed is not None:
    np.random.seed(seed=args.seed)
    tf.set_random_seed(args.seed)
with tf.Session() as sess:
    graph = graph.get_graph(args)
    procedure.initialize(sess, graph, args.test_or_train == 'test')
    output_data = procedure.run(sess, args, graph)
    dataio.save(sess, args, output_data)
logger.info("Success")
