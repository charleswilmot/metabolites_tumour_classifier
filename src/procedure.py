## @package procedure
#  Testing and training procedures.
import numpy as np
import logging as log
import tensorflow as tf
from tqdm import tqdm
from dataio import save_my_model, load_model, save_plots
import plot as plot
import pickle
import dataio
import ipdb
from sklearn import metrics
logger = log.getLogger("classifier")
# initializer = tf.glorot_uniform_initializer()
initializer = tf.keras.initializers.he_normal(seed=845)
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
    temp = 0
    for ii in tqdm(range(max_batches)):
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
def concat_data(ret, key="labels"):
    if len(ret[0][key].shape) > 1:
        sh = np.array(ret[0][key].shape)
        sh[0] = 0

        interest = np.empty((sh))
        for b in ret:
            interest = np.vstack((interest, b[key]))
    else:
        interest = np.empty((0))
        for b in ret:
            interest = np.append(interest, b[key])
    # print("compute", key, interest.shape)

    return np.array(interest).astype(np.float32)


def get_learning_rate(epoch):
    learning_rate = 0.05
    if epoch > 300:
        learning_rate *= np.power(0.5, 7)
    elif epoch > 200:
        learning_rate *= np.power(0.5, 6)
    elif epoch > 150:
        learning_rate *= np.power(0.5, 5)
    elif epoch > 100:
        learning_rate *= np.power(0.5, 4)
    elif epoch > 80:
        learning_rate *= np.power(0.5, 3)
    elif epoch > 60:
        learning_rate *= np.power(0.5, 2)
    elif epoch > 25:
        learning_rate *= 0.5
    return learning_rate


def reduce_lr_on_plateu(lr, acc_history, factor=0.1, patience=4,
                        epsilon=1e-02, min_lr=10e-8):
    """
    Reduce learning rate by factor when it didn't increase for patience number of epochs
    :param lr:, float, the learing rate
    :param acc_history:lr, float, the learing rate
    :param factor: float, new_lr = lr * factor
    :param patience: number of epoch that can tolerant with no increase
    :param epsilon: only focus on significant changes
    :param min_lr: lower bound on the learning rate.
    :return:
    """
    # if there are patience epochs with a decreasing accuracy and the decrease is bigger than epsilon, then reduce
    if np.sum((acc_history[1:] - acc_history[0:-1]) <= 0) >= patience \
            and np.abs(np.mean((acc_history[1:] - acc_history[0:-1]))) > epsilon:
        if lr > min_lr:
            new_lr = lr * factor
            new_lr = max(new_lr, min_lr)
        else:
            new_lr = lr
    else:
        new_lr = lr
    return new_lr


def get_fetches(model_aspect, names, train_or_test='test'):
    """
    Get fetches given key-words
    :param model_aspect: with all the train and test attributes
    :param names: the short key word from the attributes
    :param train_or_test: str, indicate which phase it is in
    :return: fetches, dict
    """
    fetches = {}
    for key in names:
        if key == 'train_op':
            fetches[key] = model_aspect[key]
        else:
            fetches[key] = model_aspect["{}_".format(train_or_test) + key]
    return fetches
########################################################################

## Testing phase
# @param sess a tensorflow Session object
# @param graph the graph (cf See also)
# @return a dictionary containing the average loss and accuracy on the data
# @see get_graph
def testing(sess, graph, model_name):
    # return loss / accuracy for the complete set
    fetches = {
        "ncorrect": graph["test_ncorrect"],
        "loss_sum": graph["test_loss_sum"],
        "confusion": graph["test_confusion"],
        "batch_size": graph["test_batch_size"],
        "test_labels": graph["test_labels"],
        "test_ids": graph["test_ids"],
        "test_features": graph["test_features"],
        "test_out": graph["test_out"],
        "test_wrong_inds": graph["test_wrong_inds"]
    }
    if "CAM" in model_name:
        fetches.update({"test_conv": graph["test_conv"],
                        "test_gap_w": graph["test_gap_w"]})
    initialize(sess, graph, test_only=True)
    ret, tape_end = compute(sess, fetches, max_batches=graph["test_batches"])
    loss, accuracy = reduce_mean_loss_accuracy(ret)
    confusion = sum_confusion(ret)
    wrong_features, wrong_labels = get_wrong_examples(ret)
    # activity = get_activity(ret)
    test_labels = concat_data(ret, key="test_labels")
    test_ids = concat_data(ret, key="test_ids")
    test_pred = concat_data(ret, key="test_out")
    test_features = concat_data(ret, key="test_features")
    if "CAM" in model_name:
        test_conv = concat_data(ret, key="test_conv")
        test_gap_w = ret[0]["test_gap_w"]
        return {"test_accuracy": accuracy,
                "test_loss": loss,
                "test_confusion": confusion,
                "test_wrong_features": wrong_features,
                "test_wrong_labels": wrong_labels,
                "test_labels": test_labels,
                "test_ids": test_ids,
                "test_features": test_features,
                "test_conv": test_conv,
                "test_gap_w": test_gap_w,
                "test_pred": test_pred}
    else:
        return {"test_accuracy": accuracy,
                "test_loss": loss,
                "test_confusion": confusion,
                "test_wrong_features": wrong_features,
                "test_wrong_labels": wrong_labels,
                "test_labels": test_labels,
                "test_ids": test_ids,
                "test_features": test_features,
                "test_pred": test_pred}


test_phase = testing
##
#
def train_phase(sess, graph, nbatches, epoch, lr=0.001): # change
    fetches = {
        "ncorrect": graph["train_ncorrect"],
        "loss_sum": graph["train_loss_sum"],
        "confusion": graph["train_confusion"],
        "batch_size": graph["train_batch_size"],
        "train_op": graph["train_op"],
        "learning_rate_op": graph["learning_rate_op"]
    }
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
def condition(end, output_data, epoch, number_of_epochs):
    if end:
        return False
    if epoch > number_of_epochs:
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
    lr = args.learning_rate

    # while condition(end, output_data, epoch, args.number_of_epochs):
    for epoch in range(args.number_of_epochs):
        # train phase
        if len(output_data["test_accuracy"]) > 4:  # if the test_acc keeps dropping for 3 steps, reduce the learning rate
            lr = reduce_lr_on_plateu(
                lr,
                np.array(output_data["test_accuracy"][-3-1:]),
                factor=0.5, patience=3,
                epsilon=1e-04, min_lr=10e-8)

        if epoch > 20 and epoch % 20 == 0:
            print("Epoch: ", epoch)
        ret_train = train_phase(sess, graph, args.test_every, epoch, lr=lr)

        if ret_train is not None:
            output_data["train_loss"].append(ret_train["loss"])
            output_data["train_accuracy"].append(ret_train["accuracy"])
        else:
            end = True
        num_trained += ret_train["num_trained"]
        epoch = num_trained * 1.0 / args.num_train
        logger.debug("Training phase done")
        
        # test phase
        ret_test = test_phase(sess, graph, model_name=args.model_name)
        output_data["test_loss"].append(ret_test["test_loss"])
        output_data["test_accuracy"].append(ret_test["test_accuracy"])
        output_data["test_confusion"] = ret_test["test_confusion"]
        output_data["test_wrong_features"] = ret_test["test_wrong_features"]
        output_data["test_wrong_labels"] = ret_test["test_wrong_labels"]
        # output_data["test_activity"] = ret["test_activity"]
        output_data["test_labels"] = ret_test["test_labels"]
        output_data["test_pred"] = ret_test["test_pred"]
        output_data["test_ids"] = ret_test["test_ids"]
        # output_data["test_conv"] = ret["test_conv"]
        output_data["current_step"] += 1
        # TODO: how to simplify the collecting of data for future plot? Don't need to fetch labels every epoch
        logger.debug("Epoch {}, Testing phase done\t({})".format(epoch, ret_test["test_accuracy"]))

        # save model
        if output_data["test_accuracy"][-1] > best_accuracy:
            print("Epoch {:0.1f} Best test accuracy {}, test AUC {}\n Test Confusion:\n{}".format(epoch, output_data["test_accuracy"][-1], metrics.roc_auc_score(ret_test["test_labels"], ret_test["test_pred"]), ret_test["test_confusion"]))
            best_accuracy = output_data["test_accuracy"][-1]
            save_my_model(best_saver, sess, args.model_save_dir, len(output_data["test_accuracy"]), name=np.str("{:.4f}".format(best_accuracy)))
            save_plots(sess, args, output_data, training=True, epoch=epoch)
            if "CAM" in args.model_name:
                class_maps, rand_inds = plot.get_class_map(ret_test["test_labels"],
                                                ret_test["test_conv"],
                                                ret_test["test_gap_w"],
                                                args.data_len, 1, number2use = 200)
                plot.plot_class_activation_map(sess, class_maps,
                                               ret_test["test_features"][rand_inds],
                                               ret_test["test_labels"][rand_inds],
                                               ret_test["test_pred"][rand_inds],
                                               epoch, args)

    logger.info("Training procedure done")
    save_plots(sess, args, output_data, training=True, epoch=epoch)
    return output_data


## Dispatches the call between test and train
# @param sess a tensorflow Session object
# @param args the arguments passed to the software
# @param graph the graph (cf See also)
# @param input_data training and testing data
def main_train(sess, args, graph):
    saver = tf.train.Saver()
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
        output_data = training(sess, args, graph, saver)
        dataio.save_plots(sess, args, output_data, training=True)
    else:
        initialize(sess, graph, test_only=True)
        output_data = testing(sess, graph)

        with open(args.output_path + '/Data{}_class{}_model{}_test_return_data_acc_{:.3f}.txt'.format(
                    args.data_source, args.num_classes, args.model_name,
                    output_data["test_accuracy"]), 'wb') as ff:
            pickle.dump({"output_data": output_data}, ff)
        dataio.save_plots(sess, args, output_data, training=False)
        
