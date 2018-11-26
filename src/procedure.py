## @package procedure
#  Testing and training procedures.
import numpy as np
import logging as log
import tensorflow as tf

from dataio import save_my_model, load_model, save_plots


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



## Processes the output of compute (cf See also) to calculate the sum of confusion matrices
# @param ret dictionary containing the keys "ncorrect", "loss_sum" and "batch_size"
# @param N number of examples computed by compute
# @see compute
def sum_confusion(ret):
    N = sum([b["batch_size"] for b in ret])
    confusion = sum([b["confusion"] for b in ret])
    return confusion


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
    i = 0
    for _ in limit(max_batches):
        try:
            ret.append(sess.run(fetches))
        except tf.errors.OutOfRangeError:
            tape_end = True
            break
    return ret, tape_end


def initialize(sess, graph, test_only=False):
    if test_only:
        fetches = graph["test_initializer"]
    else:
        fetches = [graph["test_initializer"], graph["train_initializer"]]
    sess.run(fetches)


def get_wrong_examples(fetches):
    """
    Get the wrongly classified examples with features and label
    :param sess:
    :param fetches:
    :return: wrong examples with features and their labels
    """

    num_classes = fetches[0]["test_labels"].shape[-1]
    data_len = fetches[0]["test_features"].shape[-1]
    features = np.empty((0, data_len))
    labels = np.empty((0, num_classes))
    for i in range(len(fetches)):
        features = np.vstack((features, fetches[i]["test_features"][np.where(fetches[i]["test_wrong_inds"] == 0)[0]]))
        labels = np.vstack((labels, fetches[i]["test_labels"][np.where(fetches[i]["test_wrong_inds"] == 0)[0]]))
    return features, labels

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
        "confusion": graph["test_confusion"],
        "batch_size": graph["test_batch_size"],
        "test_labels":graph["test_labels"],
        "test_features":graph["test_features"],
        "test_out":graph["test_out"],
        "test_wrong_inds": graph["test_wrong_inds"]
    }
    initialize(sess, graph, test_only=True)
    ret, tape_end = compute(sess, graph, fetches)
    loss, accuracy = reduce_mean_loss_accuracy(ret)
    confusion = sum_confusion(ret)
    wrong_features, wrong_labels = get_wrong_examples(ret)
    return {"accuracy": accuracy, "loss": loss, "confusion": confusion, "test_wrong_features": wrong_features, "test_wrong_labels": wrong_labels, "current_step": "test"}


test_phase = testing

##
#
def train_phase(sess, graph, nbatches): # change
    fetches = {
        "ncorrect": graph["train_ncorrect"],
        "loss_sum": graph["train_loss_sum"],
        "confusion": graph["train_confusion"],
        "batch_size": graph["train_batch_size"],
        "train_op": graph["train_op"]
    }

    ret, tape_end = compute(sess, graph, fetches, max_batches=nbatches)
    if tape_end:
        return None
    loss, accuracy = reduce_mean_loss_accuracy(ret)
    confusion = sum_confusion(ret)
    return {"accuracy": accuracy, "loss": loss, "confusion": confusion}


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
def condition(end, output_data, number_of_epochs):
    if end:
        return False
    if len(output_data["test_accuracy"]) < 1 or number_of_epochs != -1:
        return True
    else:
        best_accuracy = max(output_data["test_accuracy"])
        c = (np.array(output_data["test_accuracy"])[-5:] < best_accuracy).all()
        if c:
            logger.info("Termination condition fulfilled")
        return not c


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
def training(sess, args, graph, saver):
    logger.info("Starting training procedure")
    best_saver = tf.train.Saver(max_to_keep=3)   # keep the top 3 best models

    output_data = {}
    output_data["train_loss"] = []
    output_data["train_accuracy"] = []
    output_data["train_confusion"] = 0
    output_data["test_loss"] = []
    output_data["test_accuracy"] = []
    output_data["test_confusion"] = 0
    output_data["current_step"] = 0
    end = False
    best_accuracy = 0

    while condition(end, output_data, args.number_of_epochs):
        # train phase
        ret = train_phase(sess, graph, args.test_every)
        if ret is not None:
            output_data["train_loss"].append(ret["loss"])
            output_data["train_accuracy"].append(ret["accuracy"])
        else:
            end = True
        logger.debug("Training phase done")
        # test phase
        ret = test_phase(sess, graph)
        output_data["test_loss"].append(ret["loss"])
        output_data["test_accuracy"].append(ret["accuracy"])
        output_data["current_step"] += 1
        output_data["test_confusion"] = ret["confusion"]
        output_data["test_wrong_features"] = ret["wrong_features"]
        output_data["test_wrong_labels"] = ret["wrong_labels"]


        logger.debug("Testing phase done\t({})".format(ret["accuracy"]))

        # save model
        if output_data["test_accuracy"][-1] > best_accuracy:
            print("Best accuracy {}".format(output_data["test_accuracy"][-1]))
            best_accuracy = output_data["test_accuracy"][-1]
            save_my_model(best_saver, sess, args.model_save_dir, len(output_data["test_accuracy"]), name=np.str("{:.4f}".format(best_accuracy)))
            save_plots(sess, args, output_data)

    logger.info("Training procedure done")
    return output_data


## Dispatches the call between test and train
# @param sess a tensorflow Session object
# @param args the arguments passed to the software
# @param graph the graph (cf See also)
# @param input_data training and testing data
def run(sess, args, graph):
    saver = tf.train.Saver()
    if args.restore_from:
        global_step = load_model(saver, sess, args.restore_from)
        logger.info("Restore model Done! Global step is {}".format(global_step))
    else:
        # raise(NotImplementedError("Initialize train iterator here..."))
        logger.info("Initializing network with random weights")
        sess.run(tf.global_variables_initializer())
    if args.test_or_train == 'train':
        return training(sess, args, graph, saver)
    else:
        return testing(sess, graph)
