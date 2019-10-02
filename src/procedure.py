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


# def compute(sess, fetches, max_batches=None, learning_rate=0.0005):
#     # return loss / accuracy for the complete set
#     ret = []
#     tape_end = False
#     temp = 0
#     for ii in tqdm(range(max_batches)):
#         try:
#             if "train_op" in fetches.keys():
#                 ret.append(sess.run(fetches, feed_dict={fetches["learning_rate_op"]: learning_rate}))
#             else:
#                 ret.append(sess.run(fetches))
#         except tf.errors.OutOfRangeError:
#             tape_end = True
#             break
#     return ret, tape_end


def compute(sess, fetches, compute_batches=100, lr=0.0005,
            if_get_wrong=False, if_get_certain=False):
    """
    Compute the interested tensors and ops
    :param sess:
    :param fetches:
    :param compute_batches:
    :param lr:
    :param if_get_wrong:
    :param if_get_certain:
    :return:
    """

    results = {key: 0 for key, _ in fetches.items() if key != "train_op"}
    sum_keys = ["loss", "num_correct", "confusion", "batch_size"]

    if_check_cam = False
    if "conv" in fetches.keys():
        if_check_cam = True
        exp_keys = ["labels", "features", "sample_ids"
                    "logits", "conv", "gap_w"]
        if if_get_certain:
            exp_keys += ["certain_features", "certain_labels_int", "certain_sample_ids",
                         "certain_pred", "certain_conv"]
    else:
        exp_keys = ["labels", "features", "sample_ids",
                    "logits"]
        if if_get_certain:
            exp_keys += ["certain_features", "certain_labels",
                         "certain_sample_ids", "certain_pred"]
    if if_get_wrong:
        exp_keys += ["wrong_features", "wrong_labels"]

    for key in exp_keys:
        results[key] = []

    for i in tqdm(range(compute_batches)):
        if "train_op" in fetches.keys():
            run_all = sess.run(fetches, feed_dict={fetches["learning_rate_op"]: lr})
        else:
            run_all = sess.run(fetches)

        for _, key in enumerate(run_all.keys()):
            # Sum over all the sumable variables
            if key in sum_keys:
                if np.isnan(run_all[key]).any():
                    print("{}-th batch, {} contains NaN".format(i, key))
                else:
                    results[key] = results[key] + run_all[key]
            # only take the last batch example variables for further plotting
            elif key in exp_keys:
                if np.isnan(run_all[key]).any():
                    print("{}-th batch, {} contains NaN".format(i, key))
                else:
                    results[key] = run_all[key]
        if if_get_wrong:
            results = get_wrong_examples(results)
        if if_get_certain:

            results = get_most_cer_uncertain_samples(results, cer_thsh=0.999, if_check_cam=if_check_cam)
    return results


def get_most_cer_uncertain_samples(results, cer_thsh=0.999, if_check_cam=True):
    """
    Get num2get wrong examples
    :param results: dict, with keys "wrong_BL", "wrong_EPG"
    :param cer_thsh: how many examples to collect
    :return:
    """
    # labels_int = np.argmax(results["labels"], axis=1)
    sample_ids = results["certain_sample_ids"]
    certain_inds = np.where(np.sum((results["logits"] > cer_thsh), axis=1) == 1)[0].astype(np.int)
    # data_len = results["features"].shape[-1]

    if certain_inds.size != 0:
        ipdb.set_trace()
        if len(results["certain_features"]) == 0:
            results["certain_sample_ids"] = np.empty(0)
            if if_check_cam:
                results["certain_conv"] = np.empty(results["conv"].shape)

        results["certain_sample_ids"] = np.append(results["certain_sample_ids"], sample_ids[certain_inds]).astype(np.int)
        # if if_check_cam:
        #     results["certain_conv"] = np.vstack(
        #         (results["certain_conv"], results["conv"][certain_inds]))

    return results


def get_wrong_examples(results):
    """
    Get num2get wrong examples
    :param results: dict, with keys "wrong_BL", "wrong_EPG"
    :return:
    """
    labels_int = np.argmax(results["labels"], axis=1)
    wrong_inds = np.where(labels_int != results["pred_int"])[0]
    if len(results["features"].shape) == 2:
        shape = (0, results["features"].shape[-1])
    elif len(results["features"].shape) == 3:
        shape = (0, results["features"].shape[1], results["features"].shape[-1])

    if len(wrong_inds) != 0:
        if len(results["wrong_features"]) == 0:
            results["wrong_features"] = np.empty(shape)
            results["wrong_labels"] = np.empty(0)
            results["wrong_features"] = np.vstack((results["wrong_features"], results["features"][wrong_inds]))
            results["wrong_labels"] = np.append(results["wrong_labels"], labels_int[wrong_inds]).astype(np.int)
        else:
            results["wrong_features"] = np.vstack(
                (results["wrong_features"], results["features"][wrong_inds]))
            results["wrong_labels"] = np.append(results["wrong_labels"], labels_int[wrong_inds]).astype(np.int)

    return results



def initialize(sess, graph, test_only=False):
    if test_only:
        fetches = graph["test_initializer"]
    else:
        fetches = [graph["test_initializer"], graph["train_initializer"]]
    sess.run(fetches)


# def get_wrong_examples(fetches):
#     """
#     Get the wrongly classified examples with features and label
#     :param sess:
#     :param fetches:
#     :return: wrong examples with features and their labels
#     """
#     num_classes = fetches[0]["test_labels"].shape[-1]
#     data_len = fetches[0]["test_features"].shape[-1]
#     features = np.empty((0, data_len))
#     labels = np.empty((0, num_classes))
#     for i in range(len(fetches)):
#         features = np.vstack((features, fetches[i]["test_features"][np.where(fetches[i]["test_wrong_inds"] == 0)[0]]))
#         labels = np.vstack((labels, fetches[i]["test_labels"][np.where(fetches[i]["test_wrong_inds"] == 0)[0]]))
#     return features, labels


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


def get_returns(results, names, train_or_test='test'):
    """
    Get fetches given key-words
    :param results: dict, with all the train and test attributes
    :param names: the short key word from the attributes
    :param train_or_test: str, indicate which phase it is in
    :return: fetches, dict
    """
    ret = {}
    for k in names:
        if k == 'accuracy':
            ret["{}_accuracy".format(train_or_test)] = results["ncorrect"] / results["batch_size"]
        elif k == 'loss':
            ret["{}_loss_sum".format(train_or_test)] = results["loss_sum"] / results["batch_size"]
        elif k == "batch_size":
            ret["{}_num_trained"] = results["batch_size"]
        else:
            ret["{}_".format(train_or_test)+k] = results[k]
    return ret
########################################################################

## Testing phase
# @param sess a tensorflow Session object
# @param graph the graph (cf See also)
# @return a dictionary containing the average loss and accuracy on the data
# @see get_graph
def testing(sess, graph, compute_batches=100,
            if_check_cam=False, train_or_test='test'):
    # return loss / accuracy for the complete set
    initialize(sess, graph, test_only=True)
    if if_check_cam:
        names = ['loss', 'ncorrect', 'confusion',
                 'batch_size', 'labels', 'ids',
                 'logits', 'features', 'conv', 'gap_w']
        fetches = get_fetches(graph, names, train_or_test=train_or_test)
        results = compute(sess, fetches,
                          compute_batches=compute_batches,
                          if_get_wrong=True,
                          if_get_certain=True)

        # loss, accuracy = reduce_mean_loss_accuracy(ret)
        # confusion = sum_confusion(ret)
        # wrong_features, wrong_labels = get_wrong_examples(ret)
        # activity = get_activity(ret)
        # test_labels = concat_data(ret, key="test_labels")
        # test_ids = concat_data(ret, key="test_ids")
        # test_pred = concat_data(ret, key="test_out")
        # test_features = concat_data(ret, key="test_features")
        # certain_features = concat_data(ret, key="certain_features")
        # certain_labels = concat_data(ret, key="certain_labels")
        # certain_pred = concat_data(ret, key="certain_pred")
        return_names = ["accuracy", "loss", "confusion",
                        "labels", "ids", "features", "sample_ids", "logits",
                        "wrong_features", "wrong_labels",
                        "conv", "gap_w",
                        "certain_features", "certain_labels_int", "certain_pred",
                        "certain_conv", "certain_sample_ids"]
        ret = get_returns(results, return_names, train_or_test=train_or_test)
    else:
        names = ['loss', 'ncorrect', 'confusion',
                 'batch_size', 'labels', 'ids',
                 'logits', 'features']
        fetches = get_fetches(graph, names, train_or_test=train_or_test)
        results = compute(sess, fetches,
                          compute_batches=compute_batches,
                          if_get_wrong=True,
                          if_get_certain=True)
        return_names = ["accuracy", "loss", "confusion",
                        "labels", "ids", "features", "sample_ids", "logits",
                        "wrong_features", "wrong_labels",
                        "certain_features", "certain_labels_int",
                         "certain_pred", "certain_sample_ids"]

        ret = get_returns(results, return_names, train_or_test=train_or_test)

    return ret

    # if "CAM" in model_name:
    #     test_conv = concat_data(ret, key="test_conv")
    #     test_gap_w = ret[0]["test_gap_w"]
    #     return {"test_accuracy": accuracy,
    #             "test_loss": loss,
    #             "test_confusion": confusion,
    #             "test_wrong_features": wrong_features,
    #             "test_wrong_labels": wrong_labels,
    #             "test_labels": test_labels,
    #             "test_ids": test_ids,
    #             "test_features": test_features,
    #             "test_conv": test_conv,
    #             "test_gap_w": test_gap_w,
    #             "test_pred": test_pred,
    #             "test_pred": test_pred,
    #             "test_pred": test_pred,
    #             "certain_features":ghfdgh,
    #             "certain_labels":fhgfhd,
    #             "certain_pred":jk,
    #             }
    # else:
    #     return {"test_accuracy": accuracy,
    #             "test_loss": loss,
    #             "test_confusion": confusion,
    #             "test_wrong_features": wrong_features,
    #             "test_wrong_labels": wrong_labels,
    #             "test_labels": test_labels,
    #             "test_ids": test_ids,
    #             "test_features": test_features,
    #             "test_pred": test_pred}


test_phase = testing
##
#
def train_phase(sess, graph, nbatches, epoch, lr=0.001,
                if_get_wrong=False, if_get_certain=False,
                train_or_test="train"): # change
    """
    Train phase
    :param sess:
    :param graph:
    :param nbatches:
    :param epoch:
    :param lr:
    :param if_get_wrong:
    :param if_get_certain:
    :param train_or_test:
    :return:
    """
    names = ['loss', 'ncorrect', 'confusion',
             'batch_size', 'train_op', 'logits',
             'learning_rate_op']
    fetches = get_fetches(graph, names, train_or_test=train_or_test)

    ret = compute(sess, fetches, compute_batches=nbatches, lr=lr,
            if_get_wrong=if_get_wrong, if_get_certain=if_get_certain)

    # loss, accuracy = reduce_mean_loss_accuracy(ret)
    # confusion = sum_confusion(ret)
    # num_trained = sum([b["batch_size"] for b in ret])
    return_names = ["accuracy", "loss", "confusion", "batch_size"]

    return get_returns(results, return_names, train_or_test=train_or_test)



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
        ret_train = train_phase(sess, graph, args.test_every,
                                epoch, lr=lr, if_get_wrong=False,
                                if_get_certain=True)

        if ret_train is not None:
            output_data["train_loss"].append(ret_train["loss"])
            output_data["train_accuracy"].append(ret_train["accuracy"])
        else:
            end = True
        num_trained += ret_train["num_trained"]
        epoch = num_trained * 1.0 / graph["train_num_samples"]
        logger.debug("Epoch: {:.2f}, Training phase done".format(epoch))
        
        # test phase
        if_check_cam = True if "CAM" in args.model_name else False
        ret_test = test_phase(sess, graph,
                              if_check_cam=if_check_cam,
                              compute_batches=graph["test_num_batches"],
                              train_or_test="test")
        output_data["test_loss"].append(ret_test["test_loss"])
        output_data["test_accuracy"].append(ret_test["test_accuracy"])
        output_data["test_confusion"] = ret_test["test_confusion"]
        output_data["test_wrong_features"] = ret_test["test_wrong_features"]
        output_data["test_wrong_labels"] = ret_test["test_wrong_labels"]
        # output_data["test_activity"] = ret["test_activity"]
        output_data["test_labels"] = ret_test["test_labels"]  # collect all prediction
        output_data["test_pred"] = ret_test["test_pred"]
        output_data["test_ids"] = ret_test["test_ids"]
        # output_data["test_conv"] = ret["test_conv"]
        output_data["current_step"] += 1
        # TODO: how to simplify the collecting of data for future plot? Don't need to fetch labels every epoch
        logger.debug("Epoch {}, Testing phase done\t({})".format(epoch, ret_test["test_accuracy"]))

        # save model
        if output_data["test_accuracy"][-1] > best_accuracy:
            print("Epoch {:0.1f} Best test accuracy {}, test AUC {}\n Test Confusion:\n{}\n, saved_dir: {}".format(epoch, output_data["test_accuracy"][-1], metrics.roc_auc_score(ret_test["test_labels"], ret_test["test_pred"]), ret_test["test_confusion"], args.output_path))
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
    if args.restore_from:
        print("restore_from", args.restore_from)
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
        output_data = testing(sess, graph, model_name=args.model_name)

        with open(args.output_path + '/Data{}_class{}_model{}_test_return_data_acc_{:.3f}.txt'.format(
                    args.data_source, args.num_classes, args.model_name,
                    output_data["test_accuracy"]), 'wb') as ff:
            pickle.dump({"output_data": output_data}, ff)
        dataio.save_plots(sess, args, output_data, training=False)
        
