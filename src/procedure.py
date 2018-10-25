import dataio
import numpy as np
import logging as log
import tensorflow as tf


logger = log.getLogger("classifier")


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
        feed_dict[net.training] = False
        ret.append(sess.run(fetches, feed_dict=feed_dict))
    return ret


def reduce_mean_loss_accuracy(ret, N):
    loss = sum([b["loss_sum"] for b in ret]) / N
    accuracy = sum([b["ncorrect"] for b in ret]) / N
    return loss, accuracy


def reduce_sum_loss_accuracy(ret):
    loss = sum([b["loss_sum"] for b in ret])
    accuracy = sum([b["ncorrect"] for b in ret])
    return loss, accuracy


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


def _rec_train_phase(sess, args, graph, train_data, cursor, how_many, epoch_number):
    fetches = {
        "ncorrect": graph["ncorrect"],
        "loss_sum": graph["loss_sum"],
        "train_op": graph["train_op"]
    }
    if cursor + how_many < train_data.shape[0]:
        # enough data remaining for one phase
        data = train_data[cursor: cursor + how_many]
        logger.debug("Processing training set from {} to {}".format(cursor, cursor + how_many))
        ret = compute(sess, args, graph, data, fetches, True)
        loss_sum, ncorrect = reduce_sum_loss_accuracy(ret)
        return {"ncorrect": ncorrect, "loss_sum": loss_sum, "epoch_number": epoch_number, "cursor": cursor + how_many, "N": how_many}
    else:
        # not enough data remaining for one phase
        data = train_data[cursor:]
        logger.debug("Processing training set from {} to the end ({})".format(cursor, len(train_data)))
        ret = compute(sess, args, graph, data, fetches, True)
        N = data.shape[0]
        loss_sum, ncorrect = reduce_sum_loss_accuracy(ret)
        logger.info("Epoch number {} done".format(epoch_number + 1))
        if epoch_number + 1 == args.number_of_epochs:
            return {"ncorrect": ncorrect, "loss_sum": loss_sum, "epoch_number": epoch_number + 1, "cursor": 0, "N": N}
        train_data = dataio.shuffle(train_data)
        if how_many - N > 0:
            rec_ret = _rec_train_phase(sess, args, graph, train_data, 0, how_many - N, epoch_number + 1)
            rec_ret["loss_sum"] += loss_sum
            rec_ret["ncorrect"] += ncorrect
            rec_ret["N"] += N
            return rec_ret
        else:
            return {"ncorrect": ncorrect, "loss_sum": loss_sum, "epoch_number": epoch_number + 1, "cursor": 0, "N": N}


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


def run(sess, args, graph, input_data):
    if args.test_or_train == 'train':
        return training(sess, args, graph, input_data)
    else:
        return testing(sess, args, graph, input_data)
