## @package procedure
#  Testing and training procedures.
import dataio
import numpy as np
import logging as log
import tensorflow as tf


logger = log.getLogger("classifier")


## This helper function is an interface to the tensorflow graph
#
# It allows to specify which tensors have to be computed
# @param sess a tensorflow session object
# @param args arguments passed to the command line (used to get the maximum batch size)
# @param graph the graph (cf See also)
# @param data the data to be processed
# @param fetches the tensors and ops to be computed
# @param training a flag specifying if the computation should be run in train mode (see batch normalization)
# @return a list of structures similar to fetches
# @see get_graph
def compute(sess, args, graph, data, fetches, training):
    ret = []
    nsample = data.shape[0]
    caesuras = list(range(0, nsample, args.maximum_batch_size))
    if nsample % args.maximum_batch_size != 0:
        caesuras = caesuras + [nsample]
    caesuras = np.array(caesuras)
    net = graph["network"]
    for start, stop in zip(caesuras, caesuras[1:]):
        feed_dict = {}
        inp, out = dataio.split(data[start:stop])
        feed_dict[net.inp] = inp
        feed_dict[net.out_true] = out
        feed_dict[net.training] = training
        ret.append(sess.run(fetches, feed_dict=feed_dict))
    return ret


## Processes the output of compute (cf See also) to calculate the average loss and accuracy
# @param ret output of compute
# @param N number of examples computed by compute
# @see compute
def reduce_mean_loss_accuracy(ret, N):
    loss = sum([b["loss_sum"] for b in ret]) / N
    accuracy = sum([b["ncorrect"] for b in ret]) / N
    return loss, accuracy


## Processes the output of compute (cf See also) to calculate the sum of the loss and the total number of sample correctly classified
# @param ret output of compute
# @see compute
def reduce_sum_loss_accuracy(ret):
    loss = sum([b["loss_sum"] for b in ret])
    accuracy = sum([b["ncorrect"] for b in ret])
    return loss, accuracy


## Testing phase
# @param sess a tensorflow Session object
# @param args the arguments passed to the software
# @param graph the graph (cf See also)
# @param test_data the data to be tested
# @return a dictionary containing the average loss and accuracy on the data
# @see get_graph
def testing(sess, args, graph, test_data):
    # return loss / accuracy for the complete set
    fetches = {
        "ncorrect": graph["ncorrect"],
        "loss_sum": graph["loss_sum"]
    }
    ret = compute(sess, args, graph, test_data, fetches, False)
    N = test_data.shape[0]
    loss, accuracy = reduce_mean_loss_accuracy(ret, N)
    return {"accuracy": accuracy, "loss": loss}


test_phase = testing


## Training phase
#
# Does 3 things:
# - train the network on the training data, using a precise amount of sample (see how_many)
# - if the cursor hits the end of the training set, shuffles the data and place the cursor at its begining
# - reports how many times the cursor hit the end of the training set during that phase
# - returns the average loss and accuracy on these 'how_many' samples
#
# This is probably the most complicated function. The implementtion might be sub-optimal.
# This function call a recursive function under the hood. See inline comment of the function
# _rec_train_phase for more details.
# @param sess a tensorflow Session object
# @param args the arguments passed to the software
# @param graph the graph (cf See also)
# @param train_data the data the network should be trained on
# @param cursor index in the training set where the training should start
# @param how_many amount of training sample to use for training
# @param epoch_number current epoch number. Used to determine if training should stop prematurely
# @see _rec_train_phase
def train_phase(sess, args, graph, train_data, cursor, how_many, epoch_number):
    _ret = _rec_train_phase(sess, args, graph, train_data, cursor, how_many, epoch_number)
    N = _ret["N"]
    ret = {
        "accuracy": _ret["ncorrect"] / N,
        "loss": _ret["loss_sum"] / N,
        "epoch_number": _ret["epoch_number"],
        "cursor": _ret["cursor"]
    }
    return ret


## Recurrent implementation of the training phase
# @param sess a tensorflow Session object
# @param args the arguments passed to the software
# @param graph the graph (cf See also)
# @param train_data the data the network should be trained on
# @param cursor index in the training set where the training should start
# @param how_many amount of training sample to use for training
# @param epoch_number current epoch number. Used to determine if training should stop prematurely
# @see train_phase
# @todo there is room for improvement here...
def _rec_train_phase(sess, args, graph, train_data, cursor, how_many, epoch_number):
    fetches = {
        "ncorrect": graph["ncorrect"],
        "loss_sum": graph["loss_sum"],
        "train_op": graph["train_op"]
    }
    if cursor + how_many < train_data.shape[0]:
        # There are more than how_many samples remaining in the training set
        data = train_data[cursor: cursor + how_many]  # get the data that should be trained
        logger.debug("Processing training set from {} to {}".format(cursor, cursor + how_many))
        ret = compute(sess, args, graph, data, fetches, True)  # train the network
        loss_sum, ncorrect = reduce_sum_loss_accuracy(ret)  # retrieve sum of loss and accuracy
        return {"ncorrect": ncorrect, "loss_sum": loss_sum, "epoch_number": epoch_number, "cursor": cursor + how_many, "N": how_many}
    else:
        # There are less than how_many samples remaining in the training set
        # in that case, we:
        # - (1) train the network with this data,
        # - (2) notify that an epoch has been completed,
        # - (2bis) if the requested number of epoch is reached, prematurely exit
        # - (3) suffle the training set
        # - (4) continue training from the begining of the suffled data until 'how_many' is reached
        data = train_data[cursor:]  # get the remaining data in the training set
        logger.debug("Processing training set from {} to the end ({})".format(cursor, len(train_data)))
        ret = compute(sess, args, graph, data, fetches, True)  # (1) train the network on the remaining data
        N = data.shape[0]
        loss_sum, ncorrect = reduce_sum_loss_accuracy(ret)
        logger.info("Epoch number {} done".format(epoch_number + 1))  # (2)
        if epoch_number + 1 == args.number_of_epochs:
            return {"ncorrect": ncorrect, "loss_sum": loss_sum, "epoch_number": epoch_number + 1, "cursor": 0, "N": N}
        train_data = dataio.shuffle(train_data)  # (3)
        if how_many - N > 0:
            rec_ret = _rec_train_phase(sess, args, graph, train_data, 0, how_many - N, epoch_number + 1)  # (4)
            rec_ret["loss_sum"] += loss_sum
            rec_ret["ncorrect"] += ncorrect
            rec_ret["N"] += N
            return rec_ret
        else:
            return {"ncorrect": ncorrect, "loss_sum": loss_sum, "epoch_number": epoch_number + 1, "cursor": 0, "N": N}


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
def condition(nepoch, epoch_number, output_data):
    if nepoch != -1:
        c = epoch_number < nepoch
        if not c:
            logger.info("Termination condition fulfilled: epoch number {}".format(epoch_number))
        return c
    else:
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
        logger.info("Initializing network with random weights")
        sess.run(tf.global_variables_initializer())
    output_data = {}
    output_data["train_loss"] = []
    output_data["train_accuracy"] = []
    output_data["test_loss"] = []
    output_data["test_accuracy"] = []
    train_data = input_data["train"]
    test_data = input_data["test"]
    epoch_number = 0
    cursor = 0
    l = input_data["test"].shape[0]
    r = args.train_test_compute_time_ratio
    train_phase_size = int(l * (100 - r) / r)
    while condition(args.number_of_epochs, epoch_number, output_data):
        # train phase
        ret = train_phase(sess, args, graph, train_data, cursor, train_phase_size, epoch_number)
        epoch_number = ret["epoch_number"]
        cursor = ret["cursor"]
        output_data["train_loss"].append(ret["loss"])
        output_data["train_accuracy"].append(ret["accuracy"])
        # test phase
        ret = test_phase(sess, args, graph, test_data)
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
        return training(sess, args, graph, input_data)
    else:
        return testing(sess, args, graph, input_data)
