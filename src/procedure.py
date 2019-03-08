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


def compute(sess, fetches, max_batches=None, learning_rate=0.0005):
    # return loss / accuracy for the complete set
    ret = []
    tape_end = False
    i = 0
    for _ in limit(max_batches):
        try:
            if "train_op" in fetches.keys():
                ret.append(sess.run(fetches, feed_dict={fetches["learning_rate_op"]: learning_rate}))
            else:
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


## Processes the output of compute (cf See also) to calculate the sum of confusion matrices
# @param ret dictionary containing the keys "ncorrect", "loss_sum" and "batch_size"
# @param N number of examples computed by compute
# @see compute
def get_activity(ret):
    activities = np.empty((0, 200))
    for b in ret:
        activities = np.vstack((activities, b["test_activity"]))

    return activities


## Concat data in one training/test set
# @param ret dictionary containing the keys "ncorrect", "loss_sum" and "batch_size"
# @param key, the key of the interested data in the dict that you want to extract and concat
# @return array of con
def concat_labels(ret, key="labels"):
    interest = np.empty((0, ret[0][key].shape[1]))
    for b in ret:
        interest = np.vstack((interest, b[key]))

    return np.array(interest)


def get_learning_rate(epoch):
    learning_rate = 0.01
    if epoch > 100:
        learning_rate *= 5 * 1e-1
    elif epoch > 80:
        learning_rate *= 5 * 1e-1
    elif epoch > 40:
        learning_rate *= 5 * 1e-1
    elif epoch > 10:
        learning_rate *= 5 * 1e-1
    return learning_rate
########################################################################

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
        "test_labels": graph["test_labels"],
        "test_features": graph["test_features"],
        "test_out": graph["test_out"],
        "test_wrong_inds": graph["test_wrong_inds"],
    }
    initialize(sess, graph, test_only=True)
    ret, tape_end = compute(sess, fetches)
    loss, accuracy = reduce_mean_loss_accuracy(ret)
    confusion = sum_confusion(ret)
    wrong_features, wrong_labels = get_wrong_examples(ret)
    # activity = get_activity(ret)
    test_labels = concat_labels(ret, key="test_labels")
    test_pred = concat_labels(ret, key="test_out")
    return {"test_accuracy": accuracy,
            "test_loss": loss,
            "test_confusion": confusion,
            "test_wrong_features": wrong_features,
            "test_wrong_labels": wrong_labels,
            "test_labels": test_labels,
            "test_pred": test_pred}


test_phase = testing
##
#
def train_phase(sess, graph, nbatches, epoch): # change
    fetches = {
        "ncorrect": graph["train_ncorrect"],
        "loss_sum": graph["train_loss_sum"],
        "confusion": graph["train_confusion"],
        "batch_size": graph["train_batch_size"],
        "train_op": graph["train_op"],
        "learning_rate_op": graph["learning_rate_op"]
    }
    lr = get_learning_rate(epoch)
    ret, tape_end = compute(sess, fetches, max_batches=nbatches, learning_rate=lr)
    if tape_end:
        return None
    loss, accuracy = reduce_mean_loss_accuracy(ret)
    confusion = sum_confusion(ret)
    num_trained = sum([b["batch_size"] for b in ret])
    return {"accuracy": accuracy, "loss": loss, "confusion": confusion, "num_trained": num_trained}


## Termination contition of the training
#
# Training terminates after a fixed amount of epoches if the option
# --number-of-epochs is set. Otherwise, it terminates when the test accuracy
# starts decreasing (early stoping)
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
    output_data["test_pred"] = []
    output_data["current_step"] = 0
    end = False
    best_accuracy = 0
    epoch = 0
    num_trained = 0

    while condition(end, output_data, args.number_of_epochs):
        # train phase
        
        ret = train_phase(sess, graph, args.test_every, epoch)
        if ret is not None:
            output_data["train_loss"].append(ret["loss"])
            output_data["train_accuracy"].append(ret["accuracy"])
        else:
            end = True
        num_trained += ret["num_trained"]
        epoch = num_trained * 1.0 / args.num_train
        logger.debug("Training phase done")
        
        # test phase
        ret = test_phase(sess, graph)
        output_data["test_loss"].append(ret["test_loss"])
        output_data["test_accuracy"].append(ret["test_accuracy"])

        output_data["test_confusion"] = ret["test_confusion"]
        output_data["test_wrong_features"] = ret["test_wrong_features"]
        output_data["test_wrong_labels"] = ret["test_wrong_labels"]
        # output_data["test_activity"] = ret["test_activity"]
        output_data["test_labels"] = ret["test_labels"]
        output_data["test_pred"] = ret["test_pred"]
        output_data["current_step"] += 1
        # TODO: how to simplify the collecting of data for future plot? Don't need to fetch labels every epoch
        logger.debug("Epoch {}, Testing phase done\t({})".format(epoch, ret["test_accuracy"]))

        # save model
        if output_data["test_accuracy"][-1] > best_accuracy:
            print("Epoch {:0.1f} Best accuracy {}".format(epoch, output_data["test_accuracy"][-1]))
            best_accuracy = output_data["test_accuracy"][-1]
            save_my_model(best_saver, sess, args.model_save_dir, len(output_data["test_accuracy"]), name=np.str("{:.4f}".format(best_accuracy)))
            save_plots(sess, args, output_data, training=True, epoch=epoch)

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
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    if args.test_or_train == 'train':
        initialize(sess, graph, test_only=False)
        return training(sess, args, graph, saver)
    else:
        initialize(sess, graph, test_only=True)
        return testing(sess, graph)
        
