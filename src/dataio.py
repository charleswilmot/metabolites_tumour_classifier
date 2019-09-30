## @package dataio
#  This package handles the interface with the hard drive.
#
#  It can in particular read and write matlab or python matrices.
import numpy as np
import logging as log
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
from plot import plot_aug_examples
logger = log.getLogger("classifier")


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
    val_inds = indices[fold*np.int(num_val): (fold+1)*np.int(num_val)]
    train_test_inds = indices[np.int(num_val):]
    train_test["features"] = np.vstack((train_test["features"], spectra[train_test_inds, :]))
    train_test["labels"] = np.append(train_test["labels"], labels[train_test_inds])
    train_test["ids"] = np.append(train_test["ids"], ids[train_test_inds])

    validate["features"] = np.vstack((validate["features"], spectra[val_inds, :]))
    validate["labels"] = np.append(validate["labels"],labels[val_inds])
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
        lout_ids = list(count.keys())[num_lout*start :]
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
        scipy.io.savemat(os.path.dirname(args.input_data) + '/20190325-{}class_lout{}_val_data{}.mat'.format(args.num_classes, args.num_lout, i), val_mat)
        scipy.io.savemat(
            os.path.dirname(args.input_data) + '/20190325-{}class_lout{}_train_test_data{}.mat'.format(args.num_classes, args.num_lout, i), train_test_mat)


def split_data_for_val(args):
    """
    Split the original data into train_test set and validate set
    :param args:
    :return: save two .mat files
    """
    mat = scipy.io.loadmat(args.input_data)["DATA"]
    np.random.shuffle(mat)   # shuffle the data
    spectra = mat[:, 2:]
    labels = mat[:, 1]
    ids = mat[:, 0]

    num_val = ids.size // 10   # leave 100 samples from each class out
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
                train_test, validate = get_val_data(labels, ids, num_val, spectra, train_test, validate, class_id=class_id, fold=fold)
        elif args.num_classes == 6:
            for class_id in range(args.num_classes):
                train_test, validate = get_val_data(labels, ids, num_val, spectra, train_test, validate, class_id=class_id, fold=fold)
        elif args.num_classes == 3: # ()
            for class_id in range(args.num_classes):
                train_test, validate = get_val_data(labels, ids, num_val, spectra, train_test, validate, class_id=class_id, fold=fold)
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
        scipy.io.savemat(os.path.dirname(args.input_data) + '/{}class_val_rand_data{}.mat'.format(args.num_classes, fold), val_mat)
        scipy.io.savemat(os.path.dirname(args.input_data) + '/{}class_train_test_rand_data{}.mat'.format(args.num_classes, fold), train_test_mat)


def get_data(args):
    """
    Load self_saved data. A dict, data["features"], data["labels"]. See the save function in split_data_for_val()
    # First time preprocess data functions are needed: split_data_for_val(args),split_data_for_lout_val(args)
    :param args: Param object with path to the data
    :return:
    """
    mat = scipy.io.loadmat(args.input_data)["DATA"]
    spectra = mat[:, 2:]
    labels = mat[:, 1]
    ids = mat[:, 0]
    train_data = {}
    test_data = {}
    spectra = zscore(spectra, axis=1)
    ## following code is to get only label 0 and 1 data from the file. TODO: to make this more easy and clear
    if args.num_classes-1 < np.max(labels):
        need_inds = np.empty((0))
        for class_id in range(args.num_classes):
            need_inds = np.append(need_inds, np.where(labels==class_id)[0])
        need_inds = need_inds.astype(np.int32)
        spectra = spectra[need_inds]
        labels = labels[need_inds]
        ids = ids[need_inds]

    if args.test_or_train == 'train':
        temp_rand = np.arange(len(labels))
        np.random.shuffle(temp_rand)
        spectra_rand = spectra[temp_rand]
        labels_rand = labels[temp_rand]
        ids_rand = ids[temp_rand]
    elif args.test_or_train == 'test':   # In test, don't shuffle
        spectra_rand = spectra
        labels_rand = labels
        ids_rand = ids
        print("data labels: ", labels_rand)
    assert args.num_classes != np.max(labels), "The number of class doesn't match the data!"
    args.num_train = int(((100 - args.test_ratio) * spectra_rand.shape[0]) // 100)
    train_data["spectra"] = spectra_rand[0:args.num_train].astype(np.float32)
    train_data["labels"] = np.squeeze(labels_rand[0:args.num_train]).astype(np.int32)
    train_data["ids"] = np.squeeze(ids_rand[0:args.num_train]).astype(np.int32)

    test_data["spectra"] = spectra_rand[args.num_train:].astype(np.float32)
    test_data["labels"] = np.squeeze(labels_rand[args.num_train:]).astype(np.int32)
    test_data["ids"] = np.squeeze(ids_rand[args.num_train:]).astype(np.int32)
    test_data["num_samples"] = len(test_data["labels"])
    print("num_samples: ", test_data["num_samples"])

    test_count = dict(Counter(list(test_data["ids"])))  # count the num of samples of each id
    sorted_count = sorted(test_count.items(), key=lambda kv: kv[1])
    np.savetxt(os.path.join(args.output_path, "test_ids_count.csv"), np.array(sorted_count), fmt='%d', delimiter=',')
    np.savetxt(os.path.join(args.output_path, "original_labels.csv"), np.array(test_data["labels"]), fmt='%d', delimiter=',')



    ## oversample the minority samples ONLY in training data
    if args.test_or_train == 'train':
        train_data = augment_data(train_data, args)
        args.num_train = int(((100 - args.test_ratio) * train_data["spectra"].shape[0]) // 100)
        print("After augmentation--num of train class 0: ", len(np.where(train_data["labels"] == 0)[0]), "num of train class 1: ",
              len(np.where(train_data["labels"] == 1)[0]))
        train_data = oversample_train(train_data, args.num_classes)
        print("After oversampling--num of train class 0: ", len(np.where(train_data["labels"] == 0)[0]), "num of train class 1: ", len(np.where(train_data["labels"] == 1)[0]))
        train_data["num_samples"] = len(train_data["labels"])

        train_count = dict(Counter(list(train_data["ids"])))  # count the num of samples of each id
        sorted_count = sorted(train_count.items(), key=lambda kv: kv[1])
        np.savetxt(os.path.join(args.output_path, "train_ids_count.csv"), np.array(sorted_count), fmt='%d', delimiter=',')
    
    return train_data, test_data


def oversample_train(train_data, num_classes):
    """
    Oversample the minority samples
    :param train_data:"spectra", 2d array, "labels", 1d array
    :return:
    """
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=34)
    X_resampled, y_resampled = ros.fit_resample(train_data["spectra"], train_data["labels"])
    train_data["spectra"] = X_resampled
    train_data["labels"] = y_resampled
    
    return train_data
    

def augment_data(train_data, args):
    """
    Get the augmentation based on mean of subset. ONly do it on train spectra

    :param train_data:
    :param args
    :return:
    """
    spec = train_data["spectra"]
    true_labels = train_data["labels"]
    true_ids = train_data["ids"]

    spec_aug = spec
    labels_aug = true_labels
    ids_aug = true_ids
    if "mean" in args.aug_method:
        ids_aug, labels_aug, spec_aug = augment_with_batch_mean(args,
                                                                ids_aug,   # need this for placeholder
                                                                labels_aug,
                                                                spec, spec_aug, # need this for placeholder
                                                                true_ids,
                                                                true_labels)
    elif args.aug_method == "noise":
        ids_aug, labels_aug, spec_aug = augment_with_random_noise(args,
                                                                  ids_aug,
                                                                  labels_aug,
                                                                  spec, spec_aug,
                                                                  true_ids,
                                                                  true_labels)

    print("Augmentation number of class 0", np.where(labels_aug == 0)[0].size, "number of class 1", np.where(labels_aug == 1)[0].size)
    train_data["spectra"] = spec_aug.astype(np.float32)
    train_data["labels"] = labels_aug.astype(np.int32)
    train_data["ids"] = ids_aug.astype(np.int32)

    return train_data


def augment_with_batch_mean(args,
                            ids_aug, labels_aug,
                            spec, spec_aug, true_ids,
                            true_labels):
    """
    Augment the original spectra with the mini-mini-same-class-batch mean
    :param aug_folds: how many folds to augment the data
    :param aug_scale: what is the scale of the mean/noise to add for augmentation
    :param ids_aug: also perserve the patient ids. Augmented ids
    :param labels_aug:
    :param num_classes:
    :param spec:
    :param spec_aug:
    :param true_ids: true patient ids
    :param true_labels:
    :return:
    """
    num2average = 1
    for class_id in range(args.num_classes):
        # find all the samples from this class
        if args.aug_method == "ops_mean":
            inds = np.where(true_labels == args.num_classes-1-class_id)[0]
        elif args.aug_method == "mean":
            inds = np.where(true_labels == class_id)[0]
        elif args.aug_method == "compose":
            inds = np.where(true_labels == class_id)[0]

        # randomly select 100 groups of 100 samples each and get mean
        aug_inds = np.random.choice(inds, inds.size*num2average, replace=True).reshape(-1, num2average)  # random pick 10000 samples and take mean every 2 samples
        mean_batch = np.mean(spec[aug_inds], axis=1)   # get a batch of spectra to get the mean for aug

        new_mean = (mean_batch - np.mean(mean_batch, axis=1)[:, np.newaxis]) / np.std(mean_batch, axis=1)[:, np.newaxis]
        # new_mean = (mean_batch - np.max(mean_batch, axis=1)[:, np.newaxis]) / (np.max(mean_batch, axis=1) - np.min(mean_batch, axis=1))[:, np.newaxis]
        # rec_inds = np.random.choice(inds, aug_folds * 100, replace=True).reshape(aug_folds, 100)  # aug receiver inds
        plot_aug_examples(new_mean, num2average, spec, true_labels, args)

        for fold in range(args.aug_folds):
            aug_zspec = (1 - args.aug_scale) * spec[inds] + new_mean[np.random.choice(new_mean.shape[0], inds.size)] * args.aug_scale
            spec_aug = np.vstack((spec_aug, aug_zspec))
            labels_aug = np.append(labels_aug, true_labels[inds])
            ids_aug = np.append(ids_aug, true_ids[inds])
        print("original spec shape class : ", class_id, spec.shape, "augment spec shape: ", spec_aug.shape)
    return ids_aug, labels_aug, spec_aug



def augment_with_random_noise(aug_folds, aug_scale,
                              ids_aug, labels_aug,
                              num_classes, spec,
                              spec_aug, true_ids,
                              true_labels):
    """
    Add random noise on the original spectra
    :param aug_folds:
    :param aug_scale:
    :param ids_aug:
    :param labels_aug:
    :param num_classes:
    :param spec:
    :param spec_aug:
    :param true_ids:
    :param true_labels:
    :return:
    """
    noise = aug_scale * np.random.uniform(size=[aug_folds, true_labels.size, spec.shape[-1]])

    for fold in range(aug_folds):
        aug_zspec = spec + noise[fold]
        spec_aug = np.vstack((spec_aug, aug_zspec))
        labels_aug = np.append(labels_aug, true_labels)
        ids_aug = np.append(ids_aug, true_ids)
    return ids_aug, labels_aug, spec_aug


## Get batches of data in tf.dataset
# @param args the arguments passed to the software
# @param train_data dict, "features", "labels"
# @param test_data dict, "features", "labels"
def get_data_tensors(args):
    data = {}
    train_data, test_data = get_data(args)
    
    test_spectra, test_labels, test_ids = tf.constant(test_data["spectra"]), tf.constant(test_data["labels"]), tf.constant(test_data["ids"])
    test_ds = tf.data.Dataset.from_tensor_slices((test_spectra, test_labels, test_ids)).batch(args.test_bs)
    
    iter_test = tf.compat.v1.data.make_initializable_iterator(test_ds)
    data["test_initializer"] = iter_test.initializer
    batch_test = iter_test.get_next()
    data["test_features"] = batch_test[0]
    data["test_labels"] = tf.one_hot(batch_test[1], args.num_classes)
    data["test_ids"] = batch_test[2]
    data["test_num_samples"] = test_data["num_samples"]
    data["test_batches"] = test_data["num_samples"] // args.test_bs
    print("test samples: ", test_data["num_samples"], "num_batches: ", data["test_batches"])
    if args.test_or_train == 'train':
        train_spectra, train_labels = tf.constant(train_data["spectra"]), tf.constant(train_data["labels"])
        train_ds = tf.data.Dataset.from_tensor_slices((train_spectra, train_labels)).shuffle(buffer_size=10000).repeat().batch(
            args.batch_size)
        iter_train = train_ds.make_initializable_iterator()
        batch_train = iter_train.get_next()
        data["train_features"] = batch_train[0]
        data["train_labels"] = tf.one_hot(batch_train[1], args.num_classes)
        data["train_initializer"] = iter_train.initializer
        data["train_num_samples"] = train_data["num_samples"]
        data["train_batches"] = train_data["num_samples"] // args.batch_size
        args.test_every = train_labels.get_shape().as_list()[0] // (args.test_freq * args.batch_size)
        # test_freq: how many times to test in one training epoch

    return data, args


## Make the output dir
# @param args the arguments passed to the software
def make_output_dir(args, sub_folders=["CAMs"]):
    if os.path.isdir(args.output_path):
        logger.critical("Output path already exists. Please use an other path.")
        raise FileExistsError("Output path already exists.")
    else:
        os.makedirs(args.output_path)
        os.makedirs(args.model_save_dir)
        for sub in sub_folders:
            os.makedirs(os.path.join(args.output_path, sub))
        # copy and save all the files
        copy_save_all_files(args)
        print(args.input_data)
        print(args.output_path)



def save_command_line(args):
    cmd = " ".join(sys.argv[:])
    with open(args.output_path + "/command_line.txt", 'w') as f:
        f.write(cmd)


def save_plots(sess, args, output_data, training=False, epoch=0):
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

