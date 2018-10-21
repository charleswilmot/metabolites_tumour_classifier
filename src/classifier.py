## @package classifier
#  This is the main file of the software.
#
#  Run python3 classifier.py -h for more details.
import network
import dataio
import argument
import training
import testing
import logging as log
import tensorflow as tf


log.debug("Packages imported")
# parse args
args = argument.parser.parse_args()
log.debug("Program arguments read")
# define network
net = network.MLP(args.network_conf)
log.info("Network defined")
# define loss if training
if test_or_train == "train":
    net_output_true, loss, optimizer = network.get_loss_and_target_phd(net.out, args.loss_type)
    log.info("Loss function defined")
# load data training or testing ...
if test_or_train == "train":
    x_data_train, y_data_train = dataio.get_data_train(args.train_test_split_ratio, augmentation=args.data_augmentation)
    log.info("Training data loaded in RAM")
    x_data_test, y_data_test = dataio.get_data_test(args.train_test_split_ratio)
    log.info("Testing data loaded in RAM")
if test_or_train == "test":
    x_data_test, y_data_test = dataio.get_data_test(100, no_label=args.data_not_labeled)
    if args.data_not_labeled:
        log.info("Testing data loaded in RAM (data is NOT labeled)")
    else:
        log.info("Testing data loaded in RAM (data is labeled)")
# initialize variables
saver = tf.train.Saver()
with tf.Session() as sess:
    if test_or_train == "train" and args.resume_training != "":
        saver.restore(sess, args.resume_training)
        log.info("Training is resumed from\t" + args.resume_training)
    if test_or_train == "train" and args.resume_training == "":
        sess.run(tf.global_variables_initializer())
        log.info("Weights initialized with random numbers")
    if test_or_train == "test":
        saver.restore(sess, args.model_path)
        log.info("Model restored from\t" + args.model_path)
    # train procedure or test procedure
    if test_or_train == "train":
        log.info("Starting training procedure")
        training.procedure(args, net, net_output_true, loss, optimizer)
    if test_or_train == "test":
        log.info("Classifying data")
        testing.procedure(args, net)
