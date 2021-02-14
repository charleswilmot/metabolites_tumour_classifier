## @package procedure
#  Testing and training procedures.
import os
import numpy as np
import logging as log
import tensorflow as tf
from tqdm import tqdm
from dataio import save_my_model, load_model, save_plots
import plot as plot
import pickle
import dataio
from sklearn import metrics

logger = log.getLogger("classifier")
# initializer = tf.glorot_uniform_initializer()
initializer = tf.keras.initializers.he_normal(seed=845)


def reduce_mean_loss_accuracy(ret):
    """
        ONly use it in test_phase, where ret["batch_size"] is not summed up yet.
        :param ret: dictionary containing the keys "num_correct", "loss_sum" and "batch_size"
        :return:
        """
    N = sum([b["batch_size"] for b in ret])
    loss = sum([b["loss"] for b in ret]) / N
    accuracy = sum([b["num_correct"] for b in ret]) / N
    return loss, accuracy


def sum_confusion(ret):
    """
    ONly use it in test_phase, where ret["batch_size"] is not summed up yet.
    :param ret:
    :return:
    """
    N = sum([b["batch_size"] for b in ret])
    confusion = sum([b["confusion"] for b in ret])
    return confusion


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
            if_get_wrong=False, if_get_certain=False, theta=0.90,
            one_epoch_learning=False):
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
        exp_keys = ["labels", "features", "sample_ids", "ids",
                    "logits", "conv", "gap_w"]
        if if_get_certain:
            exp_keys += ["certain_labels", "certain_sample_ids", "certain_ids",
                         "certain_logits", "certain_conv"]
    else:
        exp_keys = ["labels", "features", "sample_ids", "ids",
                    "logits"]
        if if_get_certain:
            exp_keys += ["certain_labels", "certain_ids",
                         "certain_sample_ids", "certain_logits"]
    if if_get_wrong:
        exp_keys += ["wrong_labels", "wrong_sample_ids", "wrong_features"]
    if one_epoch_learning:
        exp_keys += ["one_ep_labels", "one_ep_sample_ids", "one_ep_ids", "one_ep_features"]


    for key in exp_keys:
        results[key] = []

    for i in tqdm(range(compute_batches)):
        if "train_op" in fetches.keys():  # if training
            run_all = sess.run(fetches, feed_dict={fetches["learning_rate_op"]: lr})
        else:  # when testing/validating
            run_all = sess.run(fetches)

        for _, key in enumerate(run_all.keys()):
            # Sum over all the sumable variables
            if key in sum_keys:
                results[key] = results[key] + run_all[key]
            elif key in exp_keys:
                results[key] = run_all[key]  # if np.isnan(run_all[key]).any()

        if if_get_wrong:
            results = get_wrong_examples(results)
        if if_get_certain:
            results = get_most_cer_uncertain_samples(results, cer_thr=theta, if_check_cam=if_check_cam)
        if one_epoch_learning:
            results = get_one_epoch_learning_stats(results)
    return results


def compute_test_only(sess, fetches, compute_batches=100,
                      if_get_wrong=False, if_get_certain=False,
                      if_check_cam=False):
    """
    Compute test only results
    :param sess:
    :param fetches:
    :param compute_batches:
    :param lr:
    :param if_get_wrong:
    :param if_get_certain:
    :return:
    """
    sum_keys = ["loss", "num_correct", "confusion", "batch_size"]

    run_all = []
    collections = {}
    if "conv" in fetches.keys():
        if_check_cam = True
        exp_keys = ["labels", "features", "sample_ids",
                    "logits", "conv", "gap_w"]
        if if_get_certain:
            exp_keys += ["certain_labels", "certain_sample_ids",
                         "certain_logits", "certain_conv"]
    else:
        exp_keys = ["labels", "features", "sample_ids",
                    "logits"]
        if if_get_certain:
            exp_keys += ["certain_labels",
                         "certain_sample_ids", "certain_logits"]
    if if_get_wrong:
        exp_keys += ["wrong_labels", "wrong_sample_ids", "wrong_features"]

    for key in exp_keys:
        collections[key] = []

    init_wrong = True
    init_certrin = True
    for i in tqdm(range(compute_batches)):
        ret = sess.run(fetches)
        run_all.append(ret)

        if if_get_wrong:
            if init_wrong:
                for key in ret.keys():
                    collections[key] = ret[key]
                collections = get_wrong_examples(collections)
                init_wrong = True if len(collections["wrong_labels"]) == 0 \
                    else False
            else:
                collections = get_wrong_examples(collections)
        if if_get_certain:
            if init_certrin:
                for key in ret.keys():
                    collections[key] = ret[key]
                collections = get_most_cer_uncertain_samples(collections, cer_thr=0.9999,
                                                             if_check_cam=if_check_cam)
                init_certrin = True if len(collections["certain_labels"]) == 0 \
                    else False
            else:
                collections = get_most_cer_uncertain_samples(collections, cer_thr=0.9999,
                                                             if_check_cam=if_check_cam)
    return run_all, collections


def get_most_cer_uncertain_samples(results, cer_thr=0.9, if_check_cam=True):
    """
    Get num2get wrong examples
    :param results: dict, with keys "wrong_BL", "wrong_EPG"
    :param cer_thr: how many examples to collect
    :return:
    """
    if len(np.array(results["labels"]).shape) > 1:
        labels_int = np.argmax(results["labels"], axis=1)
    else:
        labels_int = results["labels"]
    sample_ids = results["sample_ids"]
    pat_ids = results["ids"]
    certain_inds = np.where(np.sum((results["logits"] > cer_thr), axis=1) == 1)[0].astype(np.int)

    if certain_inds.size != 0:
        if len(results["certain_sample_ids"]) == 0:
            results["certain_sample_ids"] = np.empty(0)
            results["certain_ids"] = np.empty(0)
            results["certain_labels"] = np.empty(0)
            results["certain_logits"] = np.empty((0, results["logits"].shape[-1]))
            if if_check_cam:
                shape = [a for a in results["conv"].shape]
                shape[0] = 0
                results["certain_conv"] = np.empty(shape)
        if if_check_cam:
            results["certain_conv"] = np.vstack((results["certain_conv"], results["conv"][certain_inds]))

        results["certain_labels"] = np.append(results["certain_labels"], labels_int[certain_inds]).astype(np.int)
        results["certain_sample_ids"] = np.append(results["certain_sample_ids"], sample_ids[certain_inds]).astype(np.int)
        results["certain_ids"] = np.append(results["certain_ids"], pat_ids[certain_inds]).astype(np.int)
        results["certain_logits"] = np.vstack((results["certain_logits"], results["logits"][certain_inds]))

    return results

def get_one_epoch_learning_stats(results):
    """
    Get num2get wrong examples
    :param results: dict, with keys "wrong_BL", "wrong_EPG"
    :return:
    """
    if len(np.array(results["labels"]).shape) > 1:
        labels_int = np.argmax(results["labels"], axis=1)
    else:
        labels_int = results["labels"]
    sample_ids = results["sample_ids"]
    pat_ids = results["ids"]

    if len(results["one_ep_sample_ids"]) == 0:
        results["one_ep_sample_ids"] = np.empty(0)
        results["one_ep_ids"] = np.empty(0)
        results["one_ep_labels"] = np.empty(0)
        results["one_ep_logits"] = np.empty((0, results["logits"].shape[-1]))

    results["one_ep_labels"] = np.append(results["one_ep_labels"], labels_int).astype(np.int)
    results["one_ep_sample_ids"] = np.append(results["one_ep_sample_ids"], sample_ids).astype(np.int)
    results["one_ep_ids"] = np.append(results["one_ep_ids"], pat_ids).astype(
        np.int)
    results["one_ep_logits"] = np.vstack((results["one_ep_logits"], results["logits"]))

    return results


def get_wrong_examples(results):
    """
    Get num2get wrong examples
    :param results: dict, with keys "wrong_BL", "wrong_EPG"
    :return:
    """
    if len(np.array(results["labels"]).shape) > 1:
        labels_int = np.argmax(results["labels"], axis=1)
    else:
        labels_int = results["labels"]
    sample_ids = results["sample_ids"]

    pred_lbs = np.argmax(results["logits"], axis=1)
    wrong_inds = np.where(labels_int != pred_lbs)[0]
    if len(results["features"].shape) == 2:
        shape = (0, results["features"].shape[-1])
    elif len(results["features"].shape) == 3:
        shape = (0, results["features"].shape[1], results["features"].shape[-1])

    if len(wrong_inds) != 0:
        if len(results["wrong_sample_ids"]) == 0:
            results["wrong_sample_ids"] = np.empty(0)
            results["wrong_labels"] = np.empty(0)
            results["wrong_features"] = np.empty(shape)

        results["wrong_sample_ids"] = np.append(results["wrong_sample_ids"], sample_ids[wrong_inds]).astype(np.int)
        results["wrong_labels"] = np.append(results["wrong_labels"], labels_int[wrong_inds]).astype(np.int)
        results["wrong_features"] = np.vstack((results["wrong_features"], results["features"][wrong_inds]))
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
# @param ret dictionary containing the keys "num_correct", "loss_sum" and "batch_size"
# @param N number of examples computed by compute
# @see compute
def get_activity(ret):
    activities = np.empty((0, 200))
    for b in ret:
        activities = np.vstack((activities, b["test_activity"]))

    return activities


## Concat data in one training/test set
# @param ret dictionary containing the keys "num_correct", "loss_sum" and "batch_size"
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
            ret["{}_accuracy".format(train_or_test)] = results["num_correct"] / results["batch_size"]
        elif k == 'loss':
            ret["{}_loss".format(train_or_test)] = results["loss"] / results["batch_size"]
        elif k == "batch_size":
            ret["num_trained"] = results["batch_size"]  # it is only in training
        else:
            ret["{}_".format(train_or_test) + k] = results[k]
    return ret


########################################################################
def training(sess, args, graph, saver):
    """
    Complete training procedure
    :param sess: a tensorflow Session object
    :param args: the arguments passed to the software
    :param graph: graph the graph (cf See also)
    :param saver:
    :return:
    """
    print("Starting training procedure")
    best_saver = tf.compat.v1.train.Saver(max_to_keep=2, save_relative_paths=True)  # keep the top 3 best models

    output_data = {}
    output_data["train_loss"] = []
    output_data["train_accuracy"] = []
    output_data["train_confusion"] = 0
    output_data["test_loss"] = []
    output_data["test_accuracy"] = []
    output_data["test_confusion"] = 0
    output_data["test_logits"] = []
    output_data["current_step"] = 0
    end = False
    best_accuracy = 0
    epoch = 0
    num_trained = 0
    lr = args.learning_rate

    # while condition(end, output_data, epoch, args.number_of_epochs):
    for epoch in range(args.number_of_epochs):
        print("epoch: ", epoch)
        # train phase
        if len(output_data[
                   "test_accuracy"]) > 4:  # if the test_acc keeps dropping for 3 steps, reduce the learning rate
            lr = reduce_lr_on_plateu(
                lr,
                np.array(output_data["test_accuracy"][-3 - 1:]),
                factor=0.5, patience=3,
                epsilon=1e-04, min_lr=10e-8)

        ret_train = train_phase(sess, graph, args,
                                lr=lr, if_get_wrong=False,
                                if_get_certain=True, train_or_test="train")

        if ret_train is not None:
            output_data["train_loss"].append(ret_train["train_loss"])
            output_data["train_accuracy"].append(ret_train["train_accuracy"])
        else:
            end = True
        num_trained += ret_train["num_trained"]

        ## test phase
        if_check_cam = True if "CAM" in args.model_name else False
        ret_test = validation_phase(sess, graph,
                                    if_check_cam=if_check_cam,
                                    compute_batches=graph["test_num_batches"],
                                    train_or_test="test")

        # if args.if_save_certain and len(ret_test["test_certain_labels"]) > 0 and epoch < 21:
        #     certain_data = np.concatenate((ret_test["test_certain_sample_ids"].reshape(-1, 1),
        #                                    ret_test["test_certain_labels"].reshape(-1, 1),
        #                                    ret_test["test_certain_logits"]), axis=1)
        #
        #     np.savetxt(os.path.join(args.output_path, "certains",
        #                             "certain_data_{}_epoch_{}_num_{}_{}.csv".format("test", epoch, ret_test[
        #                                 "test_certain_sample_ids"].size, args.data_source)),
        #                certain_data, header="sample_ids,labels" + ",logits" * args.num_classes, delimiter=",")

        epoch = num_trained * 1.0 / graph["train_num_samples"]
        output_data["test_loss"].append(ret_test["test_loss"])
        output_data["test_accuracy"].append(ret_test["test_accuracy"])
        output_data["test_confusion"] = ret_test["test_confusion"]
        output_data["test_wrong_features"] = ret_test["test_wrong_features"]
        output_data["test_wrong_labels"] = ret_test["test_wrong_labels"]
        # output_data["test_activity"] = ret["test_activity"]
        output_data["test_labels"] = ret_test["test_labels"]  # collect all prediction
        output_data["test_logits"] = ret_test["test_logits"]
        output_data["test_ids"] = ret_test["test_ids"]
        # output_data["test_conv"] = ret["test_conv"]
        output_data["current_step"] += 1
        # TODO: how to simplify the collecting of data for future plot? Don't need to fetch labels every epoch
        logger.debug("Epoch {}, Testing phase done\t({})".format(epoch, ret_test["test_accuracy"]))

        if epoch > 50 and epoch % 50 == 0:
            print(
                "Epoch {:0.1f} Best test accuracy {}, test AUC {}\n Test Confusion:\n{}".format(epoch, output_data[
                    "test_accuracy"][-1], metrics.roc_auc_score(ret_test["test_labels"], ret_test["test_logits"]),
                                                                                                ret_test[
                                                                                                    "test_confusion"]))
            save_plots(sess, args, output_data, training=True, epoch=epoch)

        ## save model
        if output_data["test_accuracy"][-1] > best_accuracy:
            print(
                "Epoch {:0.1f} Best test accuracy {}, "
                "test AUC {}\n Test Confusion:\n{}\n"
                "saved_dir: {}".format(epoch,
                                       output_data["test_accuracy"][-1],
                                       metrics.roc_auc_score(ret_test["test_labels"], ret_test["test_logits"]),
                                       ret_test["test_confusion"],
                                       args.output_path))
            best_accuracy = output_data["test_accuracy"][-1]
            auc = metrics.roc_auc_score(output_data["test_labels"], output_data["test_logits"])
            save_my_model(best_saver, sess, args.model_save_dir, len(output_data["test_accuracy"]),
                          name=np.str("{:.4f}".format(best_accuracy)))
            save_plots(sess, args, output_data, training=True, epoch=epoch)

            if "CAM" in args.model_name:
                class_maps, rand_inds = plot.get_class_map(ret_test["test_labels"],
                                                           ret_test["test_conv"],
                                                           ret_test["test_gap_w"],
                                                           args.height, 1, number2use=200)
                plot.plot_class_activation_map(sess, class_maps,
                                               ret_test["test_features"][rand_inds],
                                               ret_test["test_labels"][rand_inds],
                                               ret_test["test_logits"][rand_inds],
                                               epoch, args)

    print("Training procedure done")
    save_plots(sess, args, output_data, training=True, epoch=epoch)
    return output_data


def single_epo_runs(sess, args, graph):
    """
    Complete training procedure
    :param sess: a tensorflow Session object
    :param args: the arguments passed to the software
    :param graph: graph the graph (cf See also)
    :param saver:
    :return:
    """
    from scipy.special import softmax
    print("Starting training procedure")

    output_data = {}
    output_data["train_loss"] = []
    output_data["train_accuracy"] = []
    output_data["train_confusion"] = 0
    output_data["test_loss"] = []
    output_data["test_accuracy"] = []
    output_data["test_confusion"] = 0
    output_data["test_logits"] = []
    output_data["current_step"] = 0
    num_trained = 0
    lr = args.learning_rate

    # while condition(end, output_data, epoch, args.number_of_epochs):
    for epoch in range(args.number_of_epochs):
        # single-epoch-learning to get correct clf rate of the whole dataset.
        # train phase, reinitialize the model in each epoch
        args.rand_seed = np.random.randint(0, 9999)
        tf.compat.v1.set_random_seed(np.int(args.rand_seed))
        initialize(sess, graph, test_only=False)
        sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
        
        if len(output_data[
                   "test_accuracy"]) > 4:  # if the test_acc keeps dropping for 3 steps, reduce the learning rate
            lr = reduce_lr_on_plateu(
                lr,
                np.array(output_data["test_accuracy"][-3 - 1:]),
                factor=0.5, patience=3,
                epsilon=1e-04, min_lr=10e-8)

        ret_train = train_phase(sess, graph, args,
                                lr=lr, if_get_wrong=False,
                                if_get_certain=True, train_or_test="train")
        

        one_ep_data = np.concatenate((np.array(ret_train["train_one_ep_sample_ids"]).reshape(-1, 1),
                                      np.array(ret_train["train_one_ep_ids"]).reshape(-1, 1),
                                    np.array(ret_train["train_one_ep_labels"]).reshape(-1, 1),
                                    ret_train["train_one_ep_logits"]), axis=1)

        np.savetxt(os.path.join(args.output_path, "certains",
            "one_{}_num_{}-{}_{}_theta_{}_s{}.csv".format(epoch, ret_train[
                "train_one_ep_sample_ids"].size,
                                                      ret_train[
                                                          "train_one_ep_sample_ids"].max(),
                                                      args.data_source,
                                                      args.theta_thr,
                                                      args.rand_seed)),
                   one_ep_data,
                   header="sample_id,pat_id,label" + ",logits" * args.num_classes,
                   delimiter=",")

        if ret_train is not None:
            output_data["train_loss"].append(ret_train["train_loss"])
            output_data["train_accuracy"].append(ret_train["train_accuracy"])
        else:
            end = True
        num_trained += ret_train["num_trained"]

        epoch = num_trained * 1.0 / graph["train_num_samples"]
        # output_data["test_conv"] = ret["test_conv"]
        output_data["current_step"] += 1
        # TODO: how to simplify the collecting of data for future plot? Don't need to fetch labels every epoch

    print("100 single-epoch Training procedure done")
    output_data["test_labels"] = np.eye(args.num_classes)
    
    output_data["test_logits"] = softmax(np.random.randn(args.num_classes, args.num_classes), axis=1)
    # save_plots(sess, args, output_data, training=True, epoch=epoch)
    return output_data



def train_phase(sess, graph, args, lr=0.001,
                if_get_wrong=False, if_get_certain=False,
                train_or_test="train"):  # change
    # sess, graph, args.test_every, lr=lr, if_get_wrong=False,
    #                                 if_get_certain=True
    """
    Train phase
    :param sess:
    :param graph:
    :param test_every:
    :param epoch:
    :param lr:
    :param if_get_wrong:
    :param if_get_certain:
    :param train_or_test:
    :return:
    """
    names = ['loss', 'num_correct', 'confusion', 'labels',
             'batch_size', 'train_op', 'logits',
             'learning_rate_op', "sample_ids", "ids"]
    fetches = get_fetches(graph, names, train_or_test=train_or_test)

    ret = compute(sess, fetches, compute_batches=args.test_every, lr=lr,
                  if_get_wrong=if_get_wrong, if_get_certain=if_get_certain,
                  one_epoch_learning=args.if_single_runs, theta=args.theta_thr)
    
    if args.if_single_runs:  #collect the ids and sample ids
        return_names = ["accuracy", "loss", "confusion", "batch_size",
                            "certain_labels", "certain_logits", "certain_sample_ids", "certain_ids",
                            "one_ep_labels", "one_ep_logits", "one_ep_sample_ids", "one_ep_ids"]
    else:
        return_names = ["accuracy", "loss", "confusion", "batch_size",
                    "certain_labels", "certain_logits", "certain_sample_ids", "certain_ids"]

    return get_returns(ret, return_names, train_or_test=train_or_test)


def validation_phase(sess, graph, compute_batches=100,
                     if_check_cam=False, train_or_test='test'):
    """
    Testing phase
    :param sess:
    :param graph:
    :param compute_batches:
    :param if_check_cam:
    :param train_or_test:
    :return:
    """
    # return loss / accuracy for the complete set
    initialize(sess, graph, test_only=True)
    if if_check_cam:
        names = ['loss', 'num_correct', 'confusion',
                 'batch_size', 'labels', 'ids', 'sample_ids',
                 'logits', 'features', 'conv', 'gap_w']
        fetches = get_fetches(graph, names, train_or_test=train_or_test)

        results = compute(sess, fetches,
                          compute_batches=compute_batches,
                          if_get_wrong=True,
                          if_get_certain=True)

        return_names = ["accuracy", "loss", "confusion",
                        "labels", "ids", "features", "sample_ids", "logits",
                        "wrong_features", "wrong_labels", "wrong_sample_ids",
                        "conv", "gap_w", "certain_labels", "certain_logits",
                        "certain_conv", "certain_sample_ids"]
        ret = get_returns(results, return_names, train_or_test=train_or_test)
    else:
        names = ['loss', 'num_correct', 'confusion',
                 'batch_size', 'labels', 'ids',
                 'logits', 'features', "sample_ids"]
        fetches = get_fetches(graph, names, train_or_test=train_or_test)
        results = compute(sess, fetches,
                          compute_batches=compute_batches,
                          if_get_wrong=True,
                          if_get_certain=True)
        return_names = ["accuracy", "loss", "confusion",
                        "labels", "ids", "features", "sample_ids", "logits",
                        "wrong_labels", "wrong_sample_ids", "wrong_features",
                        "certain_labels", "certain_logits", "certain_sample_ids"]

        ret = get_returns(results, return_names, train_or_test=train_or_test)

    return ret


def test_phase(sess, graph, compute_batches=100,
               if_check_cam=False, train_or_test='test'):
    """
    Testing phase
    :param sess:
    :param graph:
    :param compute_batches:
    :param if_check_cam:
    :param train_or_test:
    :return:
    """
    # return loss / accuracy for the complete set
    initialize(sess, graph, test_only=True)
    if if_check_cam:
        names = ['loss', 'num_correct', 'confusion',
                 'batch_size', 'labels', 'ids', 'sample_ids',
                 'logits', 'features', 'conv', 'gap_w']
        fetches = get_fetches(graph, names, train_or_test=train_or_test)

        results, collections = compute_test_only(sess, fetches,
                                                 compute_batches=compute_batches,
                                                 if_get_wrong=True,
                                                 if_get_certain=True,
                                                 if_check_cam=if_check_cam)

        loss, accuracy = reduce_mean_loss_accuracy(results)
        confusion = sum_confusion(results)
        labels = concat_data(results, key="labels")
        ids = concat_data(results, key="ids")
        features = concat_data(results, key="features")
        logits = concat_data(results, key="logits")

        conv = concat_data(results, key="conv")
        sample_ids = concat_data(results, key="sample_ids")

        ret = {"test_gap_w": results[0]["gap_w"],
               "test_loss": loss, "test_accuracy": accuracy, "test_confusion": confusion,
               "test_labels": labels, "test_ids": ids, "test_features": features,
               "test_sample_ids": sample_ids,
               "test_logits": logits,
               "test_conv": collections["conv"],
               "test_wrong_features": collections["wrong_features"],
               "test_wrong_labels": collections["wrong_labels"],
               "test_wrong_sample_ids": collections["wrong_sample_ids"],
               "test_certain_labels": collections["certain_labels"],
               "test_certain_logits": collections["certain_logits"],
               "test_certain_sample_ids": collections["certain_sample_ids"],
               "test_certain_conv": collections["certain_conv"]
               }

    else:
        names = ['loss', 'num_correct', 'confusion',
                 'batch_size', 'labels', 'ids',
                 'logits', 'features', "sample_ids"]
        fetches = get_fetches(graph, names, train_or_test=train_or_test)
        results, collections = compute_test_only(sess, fetches,
                                                 compute_batches=compute_batches,
                                                 if_get_wrong=True,
                                                 if_get_certain=True,
                                                 if_check_cam=if_check_cam)

        loss, accuracy = reduce_mean_loss_accuracy(results)
        confusion = sum_confusion(results)
        labels = concat_data(results, key="labels")
        ids = concat_data(results, key="ids")
        features = concat_data(results, key="features")
        sample_ids = concat_data(results, key="sample_ids")
        logits = concat_data(results, key="logits")

        # return_names = ["accuracy", "loss", "confusion",
        #                 "labels", "ids", "features", "sample_ids", "logits",
        #                 "wrong_labels", "wrong_sample_ids", "wrong_features",
        #                 "certain_labels", "certain_logits", "certain_sample_ids"]
        #
        # ret = get_returns(results, return_names, train_or_test=train_or_test)

        ret = {"test_loss": loss, "test_accuracy": accuracy, "test_confusion": confusion,
                    "test_labels": labels, "test_ids": ids, "test_features": features,
                    "test_sample_ids": sample_ids,
                    "test_logits": logits,
                    "test_wrong_features": collections["wrong_features"],
                    "test_wrong_labels": collections["wrong_labels"],
                    "test_wrong_sample_ids": collections["wrong_sample_ids"],
                    "test_certain_labels": collections["certain_labels"],
                    "test_certain_logits": collections["certain_logits"],
                    "test_certain_sample_ids": collections["certain_sample_ids"]
                    }

    return ret


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
            print("Termination condition fulfilled")
        return not c


## Dispatches the call between test and train
# @param sess a tensorflow Session object
# @param args the arguments passed to the software
# @param graph the graph (cf See also)
# @param input_data training and testing data
def main_train(sess, args, graph):
    saver = tf.compat.v1.train.Saver(save_relative_paths=True)

    if args.restore_from:
        print("restore_from", args.restore_from)
        global_step = load_model(saver, sess, args.restore_from)
        print("Restore model Done! Global step is {}".format(global_step))
    else:
        # raise(NotImplementedError("Initialize train iterator here..."))
        print("Initializing network with random weights")
        sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])

    if args.train_or_test == 'train' and not args.if_single_runs:
        print("Starting training")
        initialize(sess, graph, test_only=False)
        args.save(os.path.join(args.output_path, "network", "parameters.json"))
        output_data = training(sess, args, graph, saver)
        dataio.save_plots(sess, args, output_data, training=True)
    elif args.train_or_test == 'test':
        print("Starting testing")
        initialize(sess, graph, test_only=True)
        if_check_cam = True if "CAM" in args.model_name else False
        output_data = test_phase(sess, graph, compute_batches=graph["test_batches"],
                                 if_check_cam=if_check_cam, train_or_test='test')

        if len(output_data["test_certain_labels"]) > 0:
            certain_data = np.concatenate((output_data["test_certain_sample_ids"].reshape(-1, 1),
                                           output_data["test_certain_labels"].reshape(-1, 1),
                                           output_data["test_certain_logits"]), axis=1)
            np.savetxt(os.path.join(args.output_path, "certain_data_{}_epoch_{}.csv".format("test", "TEST")),
                       certain_data, header="sample_ids,labels" + ",logits" * args.num_classes, delimiter=",")

        with open(args.output_path + '/Data{}_class{}_model{}_test_return_data_acc_{:.3f}.txt'.format(
                args.data_source, args.num_classes, args.model_name,
                output_data["test_accuracy"]), 'wb') as ff:
            pickle.dump({"output_data": output_data}, ff)
        dataio.save_plots(sess, args, output_data, training=False)
        logger.debug("Starting training")
        args.save(os.path.join(args.output_path, "network", "parameters.json"))
    # to get the correct clf rate of the whole data
    elif args.train_or_test == 'train' and args.if_single_runs:
        print("Starting single epoch training")
        initialize(sess, graph, test_only=False)
        args.save(os.path.join(args.output_path, "network", "parameters.yaml"))
        output_data = single_epo_runs(sess, args, graph)


