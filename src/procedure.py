## @package procedure
#  Testing and training procedures.
import dataio
import numpy as np
import logging as log
import tensorflow as tf


logger = log.getLogger("classifier")


## Processes the output of compute (cf See also) to calculate the average loss and accuracy
# @param ret dictionary containing the keys "ncorrect", "loss_sum" and "batch_size"
# @param N number of examples computed by compute
# @see compute
def reduce_mean_loss_accuracy(ret):
    N = sum([b["batch_size"] for b in ret])
    loss = sum([b["loss_sum"] for b in ret]) / N
    accuracy = sum([b["ncorrect"] for b in ret]) / N
    return loss, accuracy


def limit(max_batches):
    if max_batches is None:
        def iterator():
            while True:
                yield
        return iterator()
    else:
        return range(max_batches)


def compute(sess, graph, fetches, max_batches=None):
    # return loss / accuracy for the complete set
    ret = []
    tape_end = False
    for _ in limit(max_batches):
        try:
            ret.append(sess.run(fetches))
        except tf.errors.OutOfRangeError:
            tape_end = True
    return ret, tape_end


## Testing phase
# @param sess a tensorflow Session object
# @param graph the graph (cf See also)
# @return a dictionary containing the average loss and accuracy on the data
# @see get_graph
def testing(sess, graph):
    # return loss / accuracy for the complete set
    fetches = {
        "ncorrect": graph["test_ncorrect"],
        "loss_sum": graph["test_loss_sum"],
        "batch_size": graph["test_batch_size"]
    }
    logger.critical("TODO: initialize testing iterator here...")
    raise(NotImplementedError("Please implement me  !!!"))
    ret, tape_end = compute(sess, graph, fetches)
    loss, accuracy = reduce_mean_loss_accuracy(ret)
    return {"accuracy": accuracy, "loss": loss}


test_phase = testing


def train_phase(sess, graph, nbatches):
    fetches = {
        "ncorrect": graph["train_ncorrect"],
        "loss_sum": graph["train_loss_sum"],
        "batch_size": graph["train_batch_size"],
        "train_op": graph["train_op"]
    }
    ret, tape_end = compute(sess, graph, fetches, max_batches=nbatches)
    loss, accuracy = reduce_mean_loss_accuracy(ret)
    return {"accuracy": accuracy, "loss": loss, "end": tape_end}


## Termination contition of the training
#
# Training terminates after a fixed amount of epoches if the option
# --number-of-epochs is set. Otherwise, it terminates when the test accuracy
# starts increasing (early stoping)
# @param nepoch target number of epoch
# @param epoch_number current epoch number
# @param output_data output data of the training (contains the test accuracy)
# @return False if the termination condition is fulfilled
# @see training
def condition(end, output_data):
    if end:
        return False
    if len(output_data["test_accuracy"]) < 2:
        return True
    else:
        c = output_data["test_accuracy"][-2] >= output_data["test_accuracy"][-1]
        if not c:
            logger.info("Termination condition fulfilled: accuracy t-1 {} < {} accuracy t0".format(output_data["test_accuracy"][-2], output_data["test_accuracy"][-1]))
        return c


## Complete training procedure
#
# The training procedure uses the training and testing data to completely
# train the network, until a termination condition is fulfilled.
# It alternates between training phases and testing phases. The frequency at
# which these phases alternates is determined by the option
# --train-test-compute-time-ratio.
#
# @param sess a tensorflow Session object
# @param args the arguments passed to the software
# @param graph the graph (cf See also)
# @param input_data training and testing data
# @return output_data the loss and accuracy of training and testing
def training(sess, args, graph, input_data):
    logger.info("Starting training procedure")
    if args.resume_training:
        raise(NotImplementedError("Restore weights here"))
    else:
        raise(NotImplementedError("Initialize train iterator here..."))
        logger.info("Initializing network with random weights")
        sess.run(tf.global_variables_initializer())
    output_data = {}
    output_data["train_loss"] = []
    output_data["train_accuracy"] = []
    output_data["test_loss"] = []
    output_data["test_accuracy"] = []
    end = False
    savet = tf.train.Saver()
    while condition(end, output_data):
        # train phase
        ret = train_phase(sess, graph)
        output_data["train_loss"].append(ret["loss"])
        output_data["train_accuracy"].append(ret["accuracy"])
        end = ret["end"]
        logger.debug("Training phase done")
        saver.save(sess, args.output_path + "/network/model.ckpt")
        logger.debug("Model saved")
        # test phase
        ret = test_phase(sess, graph)
        output_data["test_loss"].append(ret["loss"])
        output_data["test_accuracy"].append(ret["accuracy"])
        logger.debug("Testing phase done")
    logger.info("Training procedure done")
    return output_data


## Dispatches the call between test and train
# @param sess a tensorflow Session object
# @param args the arguments passed to the software
# @param graph the graph (cf See also)
# @param input_data training and testing data
def run(sess, args, graph, input_data):
    if args.test_or_train == 'train':
        return training(sess, graph)
    else:
        return testing(sess, graph)
