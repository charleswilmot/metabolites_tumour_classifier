## @package dataio
#  This package handles the interface with the hard drive.
#
#  It can in particular read and write matlab or python matrices.
import numpy as np
import logging as log
import fnmatch
import sys
import os
import plot
import tensorflow as tf
import scipy.io
import random
from collections import Counter
from scipy.stats import zscore
import matplotlib.pyplot as plt
import ipdb
import plot as Plot

logger = log.getLogger("classifier")
from sklearn.model_selection import train_test_split
import pandas as pd


def find_files(directory, pattern='*.csv'):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

    return files


def get_val_data(labels, ids, num_val, spectra, train_test, validate, class_id=0, fold=0):
    """
    Get fixed ratio of val data from different numbers of each class
    :param labels: array
    :param ids: a
    :param num_val: int
    :param spectra: array
    :param train_test: dict, "features", "labels", "ids"
    :param validate: dict, with "features", "labels", "ids"
    :param class_id: the id of the class
    :return:
    """

    indices = np.where(labels == class_id)[0]
    random.shuffle(indices)
    val_inds = indices[fold * np.int(num_val): (fold + 1) * np.int(num_val)]
    train_test_inds = indices[np.int(num_val):]
    train_test["features"] = np.vstack((train_test["features"], spectra[train_test_inds, :]))
    train_test["labels"] = np.append(train_test["labels"], labels[train_test_inds])
    train_test["ids"] = np.append(train_test["ids"], ids[train_test_inds])

    validate["features"] = np.vstack((validate["features"], spectra[val_inds, :]))
    validate["labels"] = np.append(validate["labels"], labels[val_inds])
    validate["ids"] = np.append(validate["ids"], ids[val_inds])

    return train_test, validate


def pick_lout_ids(ids, count, num_lout=1, start=0):
    """
    Leave out several subjects for validation
    :param labels:
    :param ids:
    :param leave_out_id:
    :param spectra:
    :param train_test:
    :param validate:
    :return:
    Use it at the first time to get a overview of the data distribution. sorted_count = sorted(count.items(), key=lambda kv: kv[1])
    # np.savetxt("../data/20190325/20190325_count.csv", np.array(sorted_count), fmt="%.1f", delimiter=',')
    """
    # count the num of samples of each id
    if start == 9:
        lout_ids = list(count.keys())[num_lout * start:]
    else:
        lout_ids = list(count.keys())[num_lout * start: num_lout * (start + 1)]
    # np.savetxt("../data/lout_ids_{}.csv".format(start), np.array(lout_ids), fmt="%.1f", delimiter=',')
    return lout_ids


def split_data_for_lout_val(args):
    """
    Split the original data in leave several subjects
    :param args:
    :return: save two .mat files
    """
    mat = scipy.io.loadmat(args.input_data)["DATA"]
    spectra = mat[:, 2:]
    labels = mat[:, 1]  ##20190325-9243 samples, 428 patients
    ids = mat[:, 0]

    count = dict(Counter(list(ids)))

    for i in range(len(count) // args.num_lout):
        validate = {}
        validate["features"] = np.empty((0, 288))
        validate["labels"] = np.empty((0))
        validate["ids"] = np.empty((0))

        lout_ids = pick_lout_ids(ids, count, num_lout=args.num_lout, start=i)  # leave 10 subjects out

        all_inds = np.empty((0))
        for id in lout_ids:
            inds = np.where(ids == id)[0]
            all_inds = np.append(all_inds, inds)
            validate["features"] = np.vstack((validate["features"], spectra[inds, :]))
            validate["labels"] = np.append(validate["labels"], labels[inds])
            validate["ids"] = np.append(validate["ids"], ids[inds])
        train_test_data = np.delete(mat, all_inds, axis=0)  # delete all leaved-out subjects
        print("Leave out: \n", lout_ids, "\n num_lout\n", len(validate["labels"]))

        # ndData
        val_mat = {}
        train_test_mat = {}
        val_mat["DATA"] = np.zeros((validate["labels"].size, 290))
        val_mat["DATA"][:, 0] = validate["ids"]
        val_mat["DATA"][:, 1] = validate["labels"]
        val_mat["DATA"][:, 2:] = validate["features"]
        train_test_mat["DATA"] = np.zeros((len(train_test_data), 290))
        train_test_mat["DATA"][:, 0] = train_test_data[:, 0]
        train_test_mat["DATA"][:, 1] = train_test_data[:, 1]
        train_test_mat["DATA"][:, 2:] = train_test_data[:, 2:]
        print("num_train\n", len(train_test_mat["DATA"][:, 1]))
        scipy.io.savemat(
            os.path.dirname(args.input_data) + '/20190325-{}class_lout{}_val_data{}.mat'.format(args.num_classes,
                                                                                                args.num_lout, i),
            val_mat)
        scipy.io.savemat(
            os.path.dirname(args.input_data) + '/20190325-{}class_lout{}_train_test_data{}.mat'.format(args.num_classes,
                                                                                                       args.num_lout,
                                                                                                       i),
            train_test_mat)


def split_data_for_val(args):
    """
    Split the original data into train_test set and validate set
    :param args:
    :return: save two .mat files
    """
    mat = scipy.io.loadmat(args.input_data)["DATA"]
    np.random.shuffle(mat)  # shuffle the data
    spectra = mat[:, 2:]
    labels = mat[:, 1]
    ids = mat[:, 0]

    num_val = ids.size // 10  # leave 100 samples from each class out
    for fold in range(10):
        train_test = {}
        validate = {}
        validate["features"] = np.empty((0, 288))
        validate["labels"] = np.empty((0))
        validate["ids"] = np.empty((0))
        train_test["features"] = np.empty((0, 288))
        train_test["labels"] = np.empty((0))
        train_test["ids"] = np.empty((0))

        if args.num_classes == 2:
            for class_id in range(args.num_classes):
                train_test, validate = get_val_data(labels, ids, num_val, spectra, train_test, validate,
                                                    class_id=class_id, fold=fold)
        elif args.num_classes == 6:
            for class_id in range(args.num_classes):
                train_test, validate = get_val_data(labels, ids, num_val, spectra, train_test, validate,
                                                    class_id=class_id, fold=fold)
        elif args.num_classes == 3:  # ()
            for class_id in range(args.num_classes):
                train_test, validate = get_val_data(labels, ids, num_val, spectra, train_test, validate,
                                                    class_id=class_id, fold=fold)
        ### ndData
        val_mat = {}
        train_test_mat = {}
        val_mat["DATA"] = np.zeros((validate["labels"].size, 290))
        val_mat["DATA"][:, 0] = validate["ids"]
        val_mat["DATA"][:, 1] = validate["labels"]
        val_mat["DATA"][:, 2:] = validate["features"]
        train_test_mat["DATA"] = np.zeros((train_test["labels"].size, 290))
        train_test_mat["DATA"][:, 0] = train_test["ids"]
        train_test_mat["DATA"][:, 1] = train_test["labels"]
        train_test_mat["DATA"][:, 2:] = train_test["features"]
        scipy.io.savemat(
            os.path.dirname(args.input_data) + '/{}class_val_rand_data{}.mat'.format(args.num_classes, fold), val_mat)
        scipy.io.savemat(
            os.path.dirname(args.input_data) + '/{}class_train_test_rand_data{}.mat'.format(args.num_classes, fold),
            train_test_mat)


def get_data(args):
    """
    Load self_saved data. A dict, data["features"], data["labels"]. See the save function in split_data_for_val()
    # First time preprocess data functions are needed: split_data_for_val(args),split_data_for_lout_val(args)
    :param args: Param object with path to the data
    :return:
    """
    mat = scipy.io.loadmat(args.input_data)["DATA"]
    labels = mat[:, 1]

    whole_set = np.zeros((mat.shape[0], mat.shape[1] + 1))
    whole_set[:, 0] = np.arange(mat.shape[0])  # tag every sample
    whole_set[:, 1:] = mat
    train_data = {}
    test_data = {}

    ## following code is to get only label 0 and 1 data from the file. TODO: to make this more easy and clear
    if args.num_classes - 1 < np.max(labels):
        sub_inds = np.empty((0))
        for class_id in range(args.num_classes):
            sub_inds = np.append(sub_inds, np.where(labels == class_id)[0])
        sub_inds = sub_inds.astype(np.int32)
        sub_mat = whole_set[sub_inds]
    else:
        sub_mat = whole_set

    np.random.shuffle(sub_mat)
    print("data labels: ", sub_mat[:, 2])

    if args.test_ratio == 1:
        X_train, X_test, Y_train, Y_test = [], sub_mat, [], sub_mat[:, 2]
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(sub_mat, sub_mat[:, 2], test_size=args.test_ratio)
        # X_train, X_test, Y_train, Y_test = sub_mat, sub_mat[0], sub_mat[:, 2], sub_mat[0, 2]  #only for 100-single-epoch run

    test_data["spectra"] = zscore(X_test[:, 3:], axis=1).astype(np.float32)
    test_data["labels"] = Y_test.astype(np.int32)
    assert np.sum(Y_test.astype(np.int32) == X_test[:, 2].astype(np.int32)) == len(
        X_test), "train_test_split messed up the data!"
    test_data["ids"] = X_test[:, 1].astype(np.int32)
    test_data["sample_ids"] = X_test[:, 0].astype(np.int32)
    test_data["num_samples"] = len(test_data["labels"])
    print("num_samples: ", test_data["num_samples"])
    #
    test_count = dict(Counter(list(test_data["ids"])))  # count the num of samples of each id
    sorted_count = sorted(test_count.items(), key=lambda kv: kv[1])
    np.savetxt(os.path.join(args.output_path, "test_ids_count_{}.csv".format(args.data_source)), np.array(sorted_count),
               fmt='%d',
               delimiter=',')
    np.savetxt(os.path.join(args.output_path, "original_labels_{}.csv".format(args.data_source)),
               np.array(test_data["labels"]), fmt='%d',
               delimiter=',')

    ## oversample the minority samples ONLY in training data
    if args.train_or_test == 'train':
        if args.aug_folds != 0:
            train_data = augment_data(X_train, X_train, args)  #If not from certain, then random pick from train to aug train
            args.num_train = train_data["spectra"].shape[0]
            print("Use X_train aug. X_train \n After augmentation--num of train class 0: ", len(np.where(train_data["labels"] == 0)[0]),
                  "num of train class 1: ",
                  len(np.where(train_data["labels"] == 1)[0]))
        else:
            train_data["spectra"] = X_train[:, 3:]
            train_data["labels"] = Y_train
            true_lables = X_train[:, 2]
            train_data["ids"] = X_train[:, 1]
            train_data["sample_ids"] = X_train[:, 0]

        train_data["sample_ids"], train_data["ids"], train_data["labels"], train_data["spectra"] = \
            oversample_train(train_data["sample_ids"], train_data["ids"],
                             train_data["labels"], train_data["spectra"])
        print("After oversampling--num of train class 0: ", len(np.where(train_data["labels"] == 0)[0]),
              "\n num of train class 1: ", len(np.where(train_data["labels"] == 1)[0]))
        train_data["num_samples"] = len(Y_train)
        train_data["spectra"] = zscore(train_data["spectra"], axis=1).astype(np.float32)
        train_data["labels"] = train_data["labels"].astype(np.int32)
        # assert np.sum(train_data["labels"].astype(np.int32) == true_lables.astype(np.int32)) == len(
        #     train_data["labels"]), "train_test_split messed up the data!"
        train_data["ids"] = train_data["ids"].astype(np.int32)
        train_data["sample_ids"] = train_data["sample_ids"].astype(np.int32)

        train_count = dict(Counter(list(train_data["ids"])))  # count the num of samples of each id
        sorted_count = sorted(train_count.items(), key=lambda kv: kv[1])
        np.savetxt(os.path.join(args.output_path, "train_ids_count_{}.csv".format(args.data_source)),
                   np.array(sorted_count), fmt='%d', delimiter=',')

    return train_data, test_data


def get_single_ep_data(args):
    """
    Load the data all as training for single-epoch training
    # First time preprocess data functions are needed: split_data_for_val(args),split_data_for_lout_val(args)
    :param args: Param object with path to the data
    :return:
    """
    mat = scipy.io.loadmat(args.input_data)["DATA"]
    labels = mat[:, 1]

    whole_set = np.zeros((mat.shape[0], mat.shape[1] + 1))
    whole_set[:, 0] = np.arange(mat.shape[0])  # tag every sample
    whole_set[:, 1:] = mat
    train_data = {}
    test_data = {}

    ## following code is to get only label 0 and 1 data from the file. TODO: to make this more easy and clear
    if args.num_classes - 1 < np.max(labels):
        sub_inds = np.empty((0))
        for class_id in range(args.num_classes):
            sub_inds = np.append(sub_inds, np.where(labels == class_id)[0])
        sub_inds = sub_inds.astype(np.int32)
        sub_mat = whole_set[sub_inds]
    else:
        sub_mat = whole_set

    np.random.shuffle(sub_mat)
    print("data labels: ", sub_mat[:, 2])

    # use all data for single-epoch training
    X_train, X_test, Y_train, Y_test = sub_mat, sub_mat[0:5,:], sub_mat[:, 2], sub_mat[0:5,2]

    test_data["spectra"] = zscore(X_test[:, 3:], axis=1).astype(np.float32)
    test_data["labels"] = Y_test.astype(np.int32)
    assert np.sum(Y_test.astype(np.int32) == X_test[:, 2].astype(np.int32)) == len(
        X_test), "train_test_split messed up the data!"
    test_data["ids"] = X_test[:, 1].astype(np.int32)
    test_data["sample_ids"] = X_test[:, 0].astype(np.int32)
    test_data["num_samples"] = len(test_data["labels"])
    print("num_samples: ", test_data["num_samples"])
    #
    test_count = dict(Counter(list(test_data["ids"])))  # count the num of samples of each id
    sorted_count = sorted(test_count.items(), key=lambda kv: kv[1])
    np.savetxt(os.path.join(args.output_path, "test_ids_count_{}.csv".format(args.data_source)), np.array(sorted_count),
               fmt='%d',
               delimiter=',')
    np.savetxt(os.path.join(args.output_path, "original_labels_{}.csv".format(args.data_source)),
               np.array(test_data["labels"]), fmt='%d',
               delimiter=',')

    ## oversample the minority samples ONLY in training data
    if args.train_or_test == 'train':
        if args.aug_folds != 0:
            train_data = augment_data(X_train, X_train, args)  #If not from certain, then random pick from train to aug train
            args.num_train = train_data["spectra"].shape[0]
            print("Use X_train aug. X_train \n After augmentation--num of train class 0: ", len(np.where(train_data["labels"] == 0)[0]),
                  "num of train class 1: ",
                  len(np.where(train_data["labels"] == 1)[0]))
        else:
            train_data["spectra"] = X_train[:, 3:]
            train_data["labels"] = Y_train
            true_lables = X_train[:, 2]
            train_data["ids"] = X_train[:, 1]
            train_data["sample_ids"] = X_train[:, 0]

        train_data["sample_ids"], train_data["ids"], train_data["labels"], train_data["spectra"] = \
            oversample_train(train_data["sample_ids"], train_data["ids"],
                             train_data["labels"], train_data["spectra"])
        print("After oversampling--num of train class 0: ", len(np.where(train_data["labels"] == 0)[0]),
              "\n num of train class 1: ", len(np.where(train_data["labels"] == 1)[0]))
        train_data["num_samples"] = len(Y_train)
        train_data["spectra"] = zscore(train_data["spectra"], axis=1).astype(np.float32)
        train_data["labels"] = train_data["labels"].astype(np.int32)
        # assert np.sum(train_data["labels"].astype(np.int32) == true_lables.astype(np.int32)) == len(
        #     train_data["labels"]), "train_test_split messed up the data!"
        train_data["ids"] = train_data["ids"].astype(np.int32)
        train_data["sample_ids"] = train_data["sample_ids"].astype(np.int32)

        train_count = dict(Counter(list(train_data["ids"])))  # count the num of samples of each id
        sorted_count = sorted(train_count.items(), key=lambda kv: kv[1])
        np.savetxt(os.path.join(args.output_path, "train_ids_count_{}_tot_num_{}.csv".format(args.data_source, len(mat))),
                   np.array(sorted_count), fmt='%d', delimiter=',')

    return train_data, test_data


def get_data_from_certain_ids(args, certain_fns="f1"):
    """
    Load data from previous certain examples
    :param args:
    :param certain_fns: list of filenames, from train and validation
    :return:
    """
    mat = scipy.io.loadmat(args.input_data)["DATA"]
    labels = mat[:, 1]

    whole_set = np.zeros((mat.shape[0], mat.shape[1] + 1))
    whole_set[:, 0] = np.arange(mat.shape[0])  # tag every sample
    whole_set[:, 1:] = mat
    train_data = {}
    test_data = {}

    # certain_mat = np.empty((0, new_mat.shape[1]))
    sort_data = pd.read_csv(certain_fns, header=0).values
    sort_samp_ids = sort_data[:, 0].astype(np.int)
    sort_rate= sort_data[:, 1].astype(np.float32)
    picked_ids = sort_samp_ids[-np.int(args.theta_thr*len(sort_data)):]
    print(os.path.basename(certain_fns), len(picked_ids), "samples\n")
    certain_mat = whole_set[picked_ids]

    np.savetxt(os.path.join(args.output_path, "selected_top_{}percent_total_{}_samples.csv".format(args.theta_thr*100, len(picked_ids))), np.concatenate((picked_ids.reshape(-1,1),whole_set[picked_ids,1:3],sort_rate[picked_ids].reshape(-1,1)), axis=1), fmt='%.4f', header="samp_id,pat_id,lb,clf_rate", delimiter=',')

    ## following code is to get only label 0 and 1 data from the file. TODO: to make this more easy and clear
    if args.num_classes - 1 < np.max(labels):
        sub_inds = np.empty((0))
        for class_id in range(args.num_classes):
            sub_inds = np.append(sub_inds, np.where(labels == class_id)[0])
        sub_inds = sub_inds.astype(np.int32)
        sub_mat = whole_set[sub_inds]
    else:
        sub_mat = whole_set

    np.random.shuffle(sub_mat)
    print("data labels: ", sub_mat[:, 2])

    print("top", args.theta_thr*100, "% as distill, certain samples 0: ", len(np.where(certain_mat[:, 2] == 0)[0]),
          "\ncertain samples 1: ", len(np.where(certain_mat[:, 2] == 1)[0]))

    if args.train_or_test == 'train':
        temp_rand = np.arange(len(sub_mat))
        np.random.shuffle(temp_rand)
        sub_mat_shuflle = sub_mat[temp_rand]
    elif args.train_or_test == 'test':  # In test, don't shuffle
        sub_mat_shuflle = sub_mat
        print("data labels: ", sub_mat_shuflle[:, 2])

    X_train, X_test, Y_train, Y_test = train_test_split(sub_mat_shuflle, sub_mat_shuflle[:, 2],
                                                        test_size=args.test_ratio)

    test_data["spectra"] = zscore(X_test[:, 3:], axis=1).astype(np.float32)
    test_data["labels"] = Y_test.astype(np.int32)
    test_data["ids"] = X_test[:, 1].astype(np.int32)
    test_data["sample_ids"] = X_test[:, 0].astype(np.int32)
    test_data["num_samples"] = len(test_data["labels"])
    assert np.sum(Y_test.astype(np.int32) == X_test[:, 2].astype(np.int32)) == len(
        X_test), "train_test_split messed up the data!"
    print("Test num of class 0: ", len(np.where(test_data["labels"] == 0)[0]), "num of class 1: ",
          len(np.where(test_data["labels"] == 1)[0]))
    #
    test_count = dict(Counter(list(test_data["ids"])))  # count the num of samples of each id
    sorted_count = sorted(test_count.items(), key=lambda kv: kv[1])
    np.savetxt(os.path.join(args.output_path, "test_ids_count.csv"), np.array(sorted_count), fmt='%d',
               delimiter=',')
    np.savetxt(os.path.join(args.output_path, "original_labels.csv"), np.array(test_data["labels"]), fmt='%d', delimiter=',')

    ## oversample the minority samples ONLY in training data
    if args.train_or_test == 'train':
        # augment the training data
        if args.aug_folds != 0:
            train_data = augment_data(X_train, certain_mat, args)
            args.num_train = train_data["spectra"].shape[0]
            print("Use Certain aug. X_train, After augmentation--class 0: ", len(np.where(train_data["labels"] == 0)[0]), "class 1: ",
                  len(np.where(train_data["labels"] == 1)[0]))
        else:
            train_data["spectra"] = X_train[:, 3:]
            train_data["labels"] = Y_train
            true_lables = X_train[:, 2]
            train_data["ids"] = X_train[:, 1]
            train_data["sample_ids"] = X_train[:, 0]

        # change the oversampling after the augmentation
        train_data["sample_ids"], train_data["ids"], train_data["labels"], train_data["spectra"] = \
            oversample_train(train_data["sample_ids"], train_data["ids"],
                             train_data["labels"], train_data["spectra"])
        print("After oversampling--class 0: ",
              len(np.where(train_data["labels"] == 0)[0]), "class 1: ",
              len(np.where(train_data["labels"] == 1)[0]))

        train_data["num_samples"] = len(train_data["labels"])
        train_data["spectra"] = zscore(train_data["spectra"], axis=1).astype(np.float32)
        train_data["labels"] = train_data["labels"].astype(np.int32)
        # assert np.sum(train_data["labels"].astype(np.int32) == true_lables.astype(np.int32)) == len(
        #     train_data["labels"]), "train_test_split messed up the data!"
        train_data["ids"] = train_data["ids"].astype(np.int32)
        train_data["sample_ids"] = train_data["sample_ids"].astype(np.int32)

        args.num_train = train_data["spectra"].shape[0]

        train_count = dict(Counter(list(train_data["ids"])))  # count the num of samples of each id
        sorted_count = sorted(train_count.items(), key=lambda kv: kv[1])
        np.savetxt(os.path.join(args.output_path, "train_ids_count.csv"), np.array(sorted_count), fmt='%d',
                   delimiter=',')

    return train_data, test_data


def oversample_train(samp_ids, pat_ids, labels, features):
    """
    Oversample the minority samples
    :param train_data:"spectra", 2d array, "labels", 1d array
    :return:
    """
    # from imblearn.over_sampling import RandomOverSampler
    # ros = RandomOverSampler(random_state=34)
    # X_resampled, y_resampled = ros.fit_resample(features, labels)
    if np.sum(labels == 0) > np.sum(labels == 1):
        mino = 1
        majo = 0
    else:
        mino = 0
        majo = 1
    logger.info("minority is class ", mino, np.sum(labels == mino), "majority has ", np.sum(labels == majo))
    over_samp_inds = np.random.choice(np.where(labels == mino)[0], np.sum(labels == majo), replace=True)
    total_inds = list(over_samp_inds) + list(np.where(labels == majo)[0])
    samp_ids_oversamp = samp_ids[total_inds]
    pat_ids_oversamp = pat_ids[total_inds]
    lbs_oversamp = labels[total_inds]
    fea_oversamp = features[total_inds]
    return samp_ids_oversamp, pat_ids_oversamp, lbs_oversamp, fea_oversamp


def augment_data(aug_target, augs, args):
    """
    Get the augmentation based on mean of subset. ONly do it on train spectra
    :param aug_target: 2d array, n_sample * 291 [sample_id, patient_id, label, features*288], the target sample to be augmented (certain or whole set)
    :param augs: 2d array, samples used to augment others (certain)
    :param args
    :return: train_aug: dict
    """
    train_data_aug = {}
    # X_train_aug = aug_target
    if "mean" in args.aug_method:
        augmented_whole = augment_with_batch_mean(args, aug_target, augs)
    elif "noise" == args.aug_method:
        augmented_whole = augment_with_random_noise(args, aug_target)
    #
    # print("Augmentation number of class 0", np.where(X_train_aug[:, 2] == 0)[0].size, "number of class 1", np.where(X_train_aug[:, 2] == 1)[0].size)
    train_data_aug["spectra"] = augmented_whole[:, 3:].astype(np.float32)
    train_data_aug["labels"] = augmented_whole[:, 2].astype(np.int32)
    train_data_aug["ids"] = augmented_whole[:, 1].astype(np.int32)
    train_data_aug["sample_ids"] = augmented_whole[:, 0].astype(np.int32)

    return train_data_aug


def augment_with_batch_mean(args, aug_target, augs):
    """ aug_target, augs
    Augment the original spectra with the mini-mini-same-class-batch mean
    :param aug_target: 2d array, samples need to be augmented (certain or whole)
    :param augs: 2d array, samples used to augment (certain)
    :return:
    train_data_aug: 2d array
    """
    num2average = 1
    X_train_aug = np.empty((0, aug_target.shape[1]))
    X_train_aug = np.vstack((X_train_aug, aug_target))  # the first fold is the original data
    for class_id in range(args.num_classes):
        # find all the samples from this class from the samples that used to augment other samples
        if args.aug_method == "ops-mean" or args.aug_method == "ops_mean":
            inds = np.where(augs[:, 2] == args.num_classes - 1 - class_id)[0]
        elif args.aug_method == "same-mean" or args.aug_method == "same_mean":
            inds = np.where(augs[:, 2] == class_id)[0]
        elif args.aug_method == "both-mean" or args.aug_method == "both_mean":
            inds = np.arange(len(augs[:, 2]))  # use all labels to augment

        # randomly select 100 groups of 100 samples each and get mean
        aug_inds = np.random.choice(inds, args.aug_folds * len(np.where(aug_target[:, 2] == class_id)[0]) * num2average,
                                    replace=True).reshape(-1, num2average)
        target_inds = np.random.choice(np.where(aug_target[:, 2] == class_id)[0], args.aug_folds * len(
            np.where(aug_target[:, 2] == class_id)[0]) * num2average).reshape(-1, num2average)
        mean_batch = np.mean(augs[:, 3:][aug_inds], axis=1)  # get a batch of spectra to get the mean
        noise_aug_scale = np.random.uniform(args.aug_scale-0.05, args.aug_scale+0.05, size=[len(mean_batch), 1])

        aug_zspec = (1 - args.aug_scale) * aug_target[:, 3:][np.squeeze(target_inds)] + mean_batch * noise_aug_scale
        combine = np.concatenate((aug_target[:, 0][target_inds].reshape(-1, 1),
                                  aug_target[:, 1][target_inds].reshape(-1, 1),
                                  aug_target[:, 2][target_inds].reshape(-1, 1), aug_zspec), axis=1)
        X_train_aug = np.vstack((X_train_aug, combine))

        Plot.plot_train_samples(aug_zspec, aug_target[:, 2][target_inds], args, postfix="samples", data_dim=args.data_dim)

    print("original spec total shape", class_id, aug_target[:, 3:].shape, "augment spec shape: ",
          X_train_aug[:, 3:].shape)
    return X_train_aug


def augment_with_random_noise(args, target):
    """
    Add random noise on the original spectra
    :param target: 2d array, target samples to be augmented
    :return: train_data_aug: dict
    """

    noise = args.aug_scale * \
            np.random.uniform(low=0.0, high=1.0, size=[args.aug_folds, target[:, 2].size, target[:, 3:].shape[-1]])
    combine = np.empty((0, args.data_len))
    for fold in range(args.aug_folds):
        aug_zspec = target[:, 3:] + noise[fold]
        combine = np.vstack((combine, aug_zspec))

    sample_ids = np.tile(target[:, 0].reshape(-1, 1), [args.aug_folds, 1])
    patient_ids = np.tile(target[:, 1].reshape(-1, 1), [args.aug_folds, 1])
    labels = np.tile(target[:, 2].reshape(-1, 1), [args.aug_folds, 1])
    train_aug = np.concatenate((sample_ids, patient_ids, labels, combine), axis=1)

    Plot.plot_train_samples(train_aug[:, 3:], train_aug[:, 2], args, postfix="samples", data_dim=args.data_dim)

    return train_aug
#
#
# def augment_with_random_samples(args, target):
#     """
#     augment with random selected samples without distillation
#     :param target: 2d array, target samples to be augmented
#     :return: train_data_aug: dict
#     """
#     num2average = 1
#     noise = args.aug_scale * \
#             np.random.uniform(low=0.0, high=1.0, size=[args.aug_folds, target[:, 2].size, target[:, 3:].shape[-1]])
#     combine = np.empty((0, args.data_len))
#
#     # randomly select 100 groups of 100 samples each and get mean
#     aug_inds = np.random.choice(target[:, 2].size, args.aug_folds * len(np.where(aug_target[:, 2] == class_id)[0]) * num2average, replace=True).reshape(-1, num2average)
#     target_inds = np.random.choice(np.where(aug_target[:, 2] == class_id)[0], args.aug_folds * len(
#         np.where(aug_target[:, 2] == class_id)[0]) * num2average).reshape(-1, num2average)
#     mean_batch = np.mean(augs[:, 3:][aug_inds], axis=1)  # get a batch of spectra to get the mean
#     aug_zspec = (1 - args.aug_scale) * aug_target[:, 3:][np.squeeze(target_inds)] + mean_batch * args.aug_scale
#     combine = np.concatenate((aug_target[:, 0][target_inds].reshape(-1, 1),
#                               aug_target[:, 1][target_inds].reshape(-1, 1),
#                               aug_target[:, 2][target_inds].reshape(-1, 1), aug_zspec), axis=1)
#     X_train_aug = np.vstack((X_train_aug, combine))
#
#     for fold in range(args.aug_folds):
#         aug_zspec = target[:, 3:] + noise[fold]
#         combine = np.vstack((combine, aug_zspec))
#
#     sample_ids = np.tile(target[:, 0].reshape(-1, 1), [args.aug_folds, 1])
#     patient_ids = np.tile(target[:, 1].reshape(-1, 1), [args.aug_folds, 1])
#     labels = np.tile(target[:, 2].reshape(-1, 1), [args.aug_folds, 1])
#     train_aug = np.concatenate((sample_ids, patient_ids, labels, combine), axis=1)
#
#     Plot.plot_train_samples(train_aug[:, 3:], train_aug[:, 2], args, postfix="samples", data_dim=args.data_dim)
#
#     return train_aug


def get_data_tensors(args, certain_fns=None):
    """
    Get batches of data in tf.dataset

    :param args:
    :param certain_fns:
    :param mix_ori: whether use the original noisy samples as
    :return:
    """
    data = {}
    if certain_fns is None:  # get data from origal array
        train_data, test_data = get_data(args)
    else:  # Get certain AND mix original un-distilled samples
        train_data, test_data = get_data_from_certain_ids(args,
                                                          certain_fns=certain_fns)

    test_spectra, test_labels, test_ids, test_sample_ids = tf.constant(test_data["spectra"]), tf.constant(
        test_data["labels"]), tf.constant(test_data["ids"]), tf.constant(test_data["sample_ids"])
    test_ds = tf.compat.v1.data.Dataset.from_tensor_slices(
        (test_spectra, test_labels, test_ids, test_sample_ids)).batch(args.test_bs)
    if test_data["num_samples"] < args.test_bs:
        args.test_bs = test_data["num_samples"]

    iter_test = tf.compat.v1.data.make_initializable_iterator(test_ds)
    data["test_initializer"] = iter_test.initializer
    batch_test = iter_test.get_next()
    data["test_features"] = batch_test[0]
    data["test_labels"] = tf.one_hot(batch_test[1], args.num_classes)
    data["test_ids"] = batch_test[2]
    data["test_sample_ids"] = batch_test[3]
    data["test_num_samples"] = test_data["num_samples"]
    data["test_batches"] = test_data["num_samples"] // args.test_bs
    print("test samples: ", test_data["num_samples"], "num_batches: ", data["test_batches"])
    if args.train_or_test == 'train':
        train_spectra, train_labels, train_ids, train_sample_ids = tf.constant(train_data["spectra"]), tf.constant(
            train_data["labels"]), tf.constant(train_data["ids"]), tf.constant(train_data["sample_ids"])
        train_ds = tf.compat.v1.data.Dataset.from_tensor_slices(
            (train_spectra, train_labels, train_ids, train_sample_ids)).shuffle(buffer_size=8000).repeat().batch(
            args.batch_size)
        iter_train = train_ds.make_initializable_iterator()
        batch_train = iter_train.get_next()
        data["train_features"] = batch_train[0]
        data["train_labels"] = tf.one_hot(batch_train[1], args.num_classes)
        data["train_ids"] = batch_train[2]  # in training, we don't consider patient-ids
        data["train_sample_ids"] = batch_train[3]  # in training, we don't consider patient-ids
        data["train_initializer"] = iter_train.initializer
        data["train_num_samples"] = train_data["num_samples"]
        data["train_batches"] = train_data["num_samples"] // args.batch_size
        args.test_every = train_data["num_samples"] // (args.test_freq * args.batch_size)
        # test_freq: how many times to test in one training epoch

    return data, args


def get_single_ep_training_data_tensors(args, certain_fns=None):
    """
    Get batches of data in tf.dataset

    :param args:
    :param certain_fns:
    :param mix_ori: whether use the original noisy samples as
    :return:
    """
    data = {}
    train_data, test_data = get_single_ep_data(args)

    test_spectra, test_labels, test_ids, test_sample_ids = tf.constant(test_data["spectra"]), tf.constant(
        test_data["labels"]), tf.constant(test_data["ids"]), tf.constant(test_data["sample_ids"])
    test_ds = tf.compat.v1.data.Dataset.from_tensor_slices(
        (test_spectra, test_labels, test_ids, test_sample_ids)).batch(args.test_bs)
    if test_data["num_samples"] < args.test_bs:
        args.test_bs = test_data["num_samples"]

    iter_test = tf.compat.v1.data.make_initializable_iterator(test_ds)
    data["test_initializer"] = iter_test.initializer
    batch_test = iter_test.get_next()
    data["test_features"] = batch_test[0]
    data["test_labels"] = tf.one_hot(batch_test[1], args.num_classes)
    data["test_ids"] = batch_test[2]
    data["test_sample_ids"] = batch_test[3]
    data["test_num_samples"] = test_data["num_samples"]
    data["test_batches"] = test_data["num_samples"] // args.test_bs
    print("test samples: ", test_data["num_samples"], "num_batches: ", data["test_batches"])
    if args.train_or_test == 'train':
        train_spectra, train_labels, train_ids, train_sample_ids = tf.constant(train_data["spectra"]), tf.constant(
            train_data["labels"]), tf.constant(train_data["ids"]), tf.constant(train_data["sample_ids"])
        train_ds = tf.compat.v1.data.Dataset.from_tensor_slices(
            (train_spectra, train_labels, train_ids, train_sample_ids)).shuffle(buffer_size=8000).repeat().batch(
            args.batch_size)
        iter_train = train_ds.make_initializable_iterator()
        batch_train = iter_train.get_next()
        data["train_features"] = batch_train[0]
        data["train_labels"] = tf.one_hot(batch_train[1], args.num_classes)
        data["train_ids"] = batch_train[2]  # in training, we don't consider patient-ids
        data["train_sample_ids"] = batch_train[3]  # in training, we don't consider patient-ids
        data["train_initializer"] = iter_train.initializer
        data["train_num_samples"] = train_data["num_samples"]
        data["train_batches"] = train_data["num_samples"] // args.batch_size
        args.test_every = train_data["num_samples"] // (args.test_freq * args.batch_size)
        # test_freq: how many times to test in one training epoch

    return data, args


def get_noisy_mnist_data(args):
    """
    Get batches of data in tf.dataset

    :param args:
    :param certain_fns:
    :param mix_ori: whether use the original noisy samples as
    :return:
    """
    data = {}
    train_data, test_data = load_mnist_with_noise(args)

    test_spectra, test_labels, test_ids, test_sample_ids = tf.constant(test_data["spectra"]), tf.constant(
        test_data["labels"]), tf.constant(test_data["ids"]), tf.constant(test_data["sample_ids"])
    test_ds = tf.compat.v1.data.Dataset.from_tensor_slices(
        (test_spectra, test_labels, test_ids, test_sample_ids)).batch(args.test_bs)
    if test_data["num_samples"] < args.test_bs:
        args.test_bs = test_data["num_samples"]

    iter_test = tf.compat.v1.data.make_initializable_iterator(test_ds)
    data["test_initializer"] = iter_test.initializer
    batch_test = iter_test.get_next()
    data["test_features"] = batch_test[0]
    data["test_labels"] = tf.one_hot(batch_test[1], args.num_classes)
    data["test_ids"] = batch_test[2]
    data["test_sample_ids"] = batch_test[3]
    data["test_num_samples"] = test_data["num_samples"]
    data["test_batches"] = test_data["num_samples"] // args.test_bs
    print("test samples: ", test_data["num_samples"], "num_batches: ", data["test_batches"])
    if args.train_or_test == 'train':
        train_spectra, train_labels, train_ids, train_sample_ids = tf.constant(train_data["spectra"]), tf.constant(
            train_data["labels"]), tf.constant(train_data["ids"]), tf.constant(train_data["sample_ids"])
        train_ds = tf.compat.v1.data.Dataset.from_tensor_slices(
            (train_spectra, train_labels, train_ids, train_sample_ids)).shuffle(buffer_size=8000).repeat().batch(
            args.batch_size)
        iter_train = train_ds.make_initializable_iterator()
        batch_train = iter_train.get_next()
        data["train_features"] = batch_train[0]
        data["train_labels"] = tf.one_hot(batch_train[1], args.num_classes)
        data["train_ids"] = batch_train[2]  # in training, we don't consider patient-ids
        data["train_sample_ids"] = batch_train[3]  # in training, we don't consider patient-ids
        data["train_initializer"] = iter_train.initializer
        data["train_num_samples"] = train_data["num_samples"]
        data["train_batches"] = train_data["num_samples"] // args.batch_size + 1
        args.test_every = train_data["num_samples"] // (args.test_freq * args.batch_size)
        # test_freq: how many times to test in one training epoch

    return data, args


def load_mnist_with_noise(args):
    """
    Load self_saved data. A dict, data["features"], data["labels"]. See the save function in split_data_for_val()
    # First time preprocess data functions are needed: split_data_for_val(args),split_data_for_lout_val(args)
    :param args: Param object with path to the data
    :return:
    """
    from tensorflow.keras.datasets import mnist

    args.num_classes = 10
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    whole_set = np.concatenate((np.append(Y_train, Y_test).reshape(-1, 1), np.vstack(
        (X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)))), axis=1)
    new_mat = np.zeros((whole_set.shape[0], whole_set.shape[1] + 2))
    new_mat[:, 0] = np.arange(whole_set.shape[0])  # tag every sample
    new_mat[:, 2:] = whole_set
    new_mat = new_mat.astype(np.float32)
    train_data = {}
    test_data = {}

    Y_tot_noisy = introduce_label_noisy(whole_set[:, 0], noisy_ratio=args.noise_ratio, num_classes=args.num_classes, save_dir=args.output_path)
    new_mat[:, 1] = Y_tot_noisy  # noisy labels

    np.random.shuffle(new_mat)
    print("data labels: ", new_mat[:, 2])

    if args.test_ratio == 1:
        X_train, X_test, Y_train, Y_test = [], new_mat, [], new_mat[:, 2]
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(new_mat, new_mat[:, 2], test_size=args.test_ratio)

    test_data["spectra"] = X_test[:, 3:]
    test_data["labels"] = Y_test.astype(np.int32)
    assert np.sum(Y_test.astype(np.int32) == X_test[:, 2].astype(np.int32)) == len(
        X_test), "train_test_split messed up the data!"
    test_data["ids"] = X_test[:, 1].astype(np.int32)
    test_data["sample_ids"] = X_test[:, 0].astype(np.int32)
    test_data["num_samples"] = len(test_data["labels"])
    print("num_samples: ", test_data["num_samples"])
    #
    test_count = dict(Counter(list(test_data["ids"])))  # count the num of samples of each id
    sorted_count = sorted(test_count.items(), key=lambda kv: kv[1])
    np.savetxt(os.path.join(args.output_path, "test_ids_count_{}.csv".format(args.data_source)), np.array(sorted_count), fmt='%d', delimiter=',')
    np.savetxt(os.path.join(args.output_path, "original_labels_{}.csv".format(args.data_source)),
               np.array(test_data["labels"]), fmt='%d',
               delimiter=',')

    ## oversample the minority samples ONLY in training data
    if args.train_or_test == 'train':
        if args.aug_folds != 0:
            train_data = augment_data(X_train, X_train, args)
            args.num_train = train_data["spectra"].shape[0]
            print("After augmentation--num of train class 0: ", len(np.where(train_data["labels"] == 0)[0]),
                  "num of train class 1: ",
                  len(np.where(train_data["labels"] == 1)[0]))
        else:
            train_data["spectra"] = X_train[:, 3:]
            train_data["labels"] = Y_train
            true_lables = X_train[:, 2]
            train_data["ids"] = X_train[:, 1]
            train_data["sample_ids"] = X_train[:, 0]

        train_data["num_samples"] = len(Y_train)
        train_data["spectra"] = zscore(train_data["spectra"], axis=1).astype(np.float32)
        train_data["labels"] = train_data["labels"].astype(np.int32)

        train_data["ids"] = train_data["ids"].astype(np.int32)
        train_data["sample_ids"] = train_data["sample_ids"].astype(np.int32)

        train_count = dict(Counter(list(train_data["ids"])))  # count the num of samples of each id
        sorted_count = sorted(train_count.items(), key=lambda kv: kv[1])
        np.savetxt(os.path.join(args.output_path, "train_ids_count_{}.csv".format(args.data_source)),
                   np.array(sorted_count), fmt='%d', delimiter=',')

    return train_data, test_data


## Make the output dir
# @param args the arguments passed to the software
def make_output_dir(args, sub_folders=["CAMs"]):
    os.makedirs(args.output_path)
    args.model_save_dir = os.path.join(args.output_path, "network")
    os.makedirs(args.model_save_dir )
    for sub in sub_folders:
        os.makedirs(os.path.join(args.output_path, sub))
    # copy and save all the files
    copy_save_all_files(args)
    print(args.input_data)
    print(args.output_path)


def save_command_line(save_dir):
    cmd = " ".join(sys.argv[:])
    with open(save_dir + "/command_line.txt", 'w') as f:
        f.write(cmd)


def save_plots(sess, args, output_data, training=False, epoch=0, auc=0.5):
    logger.info("Saving output data")
    plot.all_figures(sess, args, output_data, training=training, epoch=epoch)
    logger.info("Output data saved to {}".format("TODO"))


def load_model(saver, sess, save_dir):
    ckpt = tf.train.get_checkpoint_state(save_dir)
    if ckpt:
        logger.info('Checkpoint found: {}'.format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        logger.info('  Global step was: {}'.format(global_step))
        logger.info('  Restoring...')
        saver.restore(sess, ckpt.model_checkpoint_path)
        logger.info(' Done.')
        return global_step
    else:
        logger.info(' No checkpoint found.')
        return None


def save_my_model(saver, sess, save_dir, step, name=None):
    """
    Save the model under current step into save_dir
    :param saver: tf.Saver
    :param sess: tf.Session
    :param save_dir: str, directory to save the model
    :param step: int, current training step
    :param name: if specify a name, then save with this name
    :return:
    """
    model_name = 'model.ckpt'
    if not name:
        checkpoint_path = os.path.join(save_dir, model_name)
    else:
        checkpoint_path = os.path.join(save_dir, name + model_name)
    logger.info('Saving checkpoint to {} ...'.format(save_dir))
    sys.stdout.flush()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saver.save(sess, checkpoint_path, global_step=step)
    logger.info('Done.')


def copy_save_all_files(args):
    """
    Copy and save all files related to model directory
    :param args:
    :return:
    """
    src_dir = '../src'
    save_dir = os.path.join(args.model_save_dir, 'src')
    if not os.path.exists(save_dir):  # if subfolder doesn't exist, should make the directory and then save file.
        os.makedirs(save_dir)
    req_extentions = ['py', 'json']
    for filename in os.listdir(src_dir):
        exten = filename.split('.')[-1]
        if exten in req_extentions:
            src_file_name = os.path.join(src_dir, filename)
            target_file_name = os.path.join(save_dir, filename)
            with open(src_file_name, 'r') as file_src:
                with open(target_file_name, 'w') as file_dst:
                    for line in file_src:
                        file_dst.write(line)
    print('Done WithCopy File!')

def introduce_label_noisy(original_lbs, noisy_ratio=0.2, num_classes=10, save_dir="save"):
    """
    Randomly introduce noisy to the labels
    :param noisy_ratio:
    :param num_classes:
    :return:
    """
    count_noise = []
    noisy_lbs = original_lbs.copy()

    all_classes = np.arange(10)
    for c in range(num_classes):
        c_inds = np.where(original_lbs == c)[0]
        rest_lbs = all_classes[all_classes != c]
        # first round random selection and random flipping
        noisy_c_inds = np.random.choice(c_inds, np.int(c_inds.size * noisy_ratio), replace=False)   # indices that we want to random flip classes
        rest_lbs_inds = np.random.uniform(0, len(rest_lbs), len(noisy_c_inds)).astype(np.int)
        flipped_lbs = rest_lbs[rest_lbs_inds]
        noisy_lbs[noisy_c_inds] = flipped_lbs

        count_noise.append([c, len(c_inds), np.sum(original_lbs[c_inds] != noisy_lbs[c_inds])])
    print(np.array(count_noise))

    plt.figure()
    plt.bar(np.array(count_noise)[:, 0], np.array(count_noise)[:, 1], 0.4, label="total count"),
    plt.bar(np.array(count_noise)[:, 0], np.array(count_noise)[:, 2], 0.4, label="noise label count")
    plt.xlabel("digits")
    plt.ylabel("count")
    plt.legend(),
    plt.title("Noisy labeling ratio {}%".format(noisy_ratio*100)),
    plt.savefig(save_dir + '/distribution of noisy labels in mnist.png', format='png')
    plt.close()

    return noisy_lbs
