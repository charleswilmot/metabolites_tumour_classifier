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
# from sklearn.model_selection import train_test_split
from collections import Counter
import ipdb
logger = log.getLogger("classifier")


def get_val_data(labels, ids, num_val, spectra, train_test, validate, class_id=0):
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
    val_inds = indices[0: np.int(num_val)]
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
    spectra = mat[:, 2:]
    labels = mat[:, 1]
    ids = mat[:, 0]
    train_test = {}
    validate = {}
    validate["features"] = np.empty((0, 288))
    validate["labels"] = np.empty((0))
    validate["ids"] = np.empty((0))
    train_test["features"]= np.empty((0, 288))
    train_test["labels"] = np.empty((0))
    train_test["ids"] = np.empty((0))

    num_val = 100   # leave 100 samples from each class out
    if args.num_classes == 2:
        for class_id in range(args.num_classes):
            train_test, validate = get_val_data(labels, ids, num_val, spectra, train_test, validate, class_id=class_id)

    elif args.num_classes == 6:
        for class_id in range(args.num_classes):
            train_test, validate = get_val_data(labels, ids, num_val, spectra, train_test, validate, class_id=class_id)
    elif args.num_classes == 3: # ()
        for class_id in range(args.num_classes):
            train_test, validate = get_val_data(labels, ids, num_val, spectra, train_test, validate, class_id=class_id)
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
    scipy.io.savemat(os.path.dirname(args.input_data) + '/{}class_val_data.mat'.format(args.num_classes, num_val), val_mat)
    scipy.io.savemat(os.path.dirname(args.input_data) + '/{}class_train_test_data.mat'.format(args.num_classes, num_val), train_test_mat)



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
    ## following code is to get only label 0 and 1 data from the file. TODO: to make this more easy and clear
    if args.num_classes-1 < np.max(labels):
        need_inds = np.empty((0))
        for class_id in range(args.num_classes):
            need_inds = np.append(need_inds, np.where(labels==class_id)[0])
        need_inds = need_inds.astype(np.int32)
        spectra = spectra[need_inds]
        labels = labels[need_inds]
        ids = ids[need_inds]

    temp_rand = np.arange(len(labels))
    np.random.shuffle(temp_rand)
    spectra_rand = spectra[temp_rand]
    labels_rand = labels[temp_rand]
    ids_rand = ids[temp_rand]
    assert args.num_classes != np.max(labels), "The number of class doesn't match the data!"
    args.num_train = int(((100 - args.test_ratio) * spectra_rand.shape[0]) // 100)
    train_data["spectra"] = spectra_rand[0:args.num_train].astype(np.float32)
    train_data["labels"] = np.squeeze(labels_rand[0:args.num_train]).astype(np.int32)
    train_data["ids"] = np.squeeze(ids_rand[0:args.num_train]).astype(np.int32)
    test_data["spectra"] = spectra_rand[args.num_train:].astype(np.float32)
    test_data["labels"] = np.squeeze(labels_rand[args.num_train:]).astype(np.int32)
    test_data["ids"] = np.squeeze(ids_rand[args.num_train:]).astype(np.int32)

    test_count = dict(Counter(list(test_data["ids"])))  # count the num of samples of each id
    sorted_count = sorted(test_count.items(), key=lambda kv: kv[1])
    np.savetxt(os.path.join(args.output_path, "test_ids_count.csv"), np.array(sorted_count), fmt='%d', delimiter=',')
    ## oversample the minority samples ONLY in training data
    if args.test_or_train == 'train':
        train_data = oversample_train(train_data, args.num_classes)
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
    
    
## Get batches of data in tf.dataset
# @param args the arguments passed to the software
# @param train_data dict, "features", "labels"
# @param test_data dict, "features", "labels"
def get_data_tensors(args):
    data = {}
    train_data, test_data = get_data(args)
    
    test_spectra, test_labels, test_ids = tf.constant(test_data["spectra"]), tf.constant(test_data["labels"]), tf.constant(test_data["ids"])
    test_ds = tf.data.Dataset.from_tensor_slices((test_spectra, test_labels, test_ids)).batch(args.batch_size)
    
    iter_test = test_ds.make_initializable_iterator()
    data["test_initializer"] = iter_test.initializer
    batch_test = iter_test.get_next()
    data["test_features"] = batch_test[0]
    data["test_labels"] = tf.one_hot(batch_test[1], args.num_classes)
    data["test_ids"] = batch_test[2]
    if args.test_or_train == 'train':
        train_spectra, train_labels = tf.constant(train_data["spectra"]), tf.constant(train_data["labels"])
        train_ds = tf.data.Dataset.from_tensor_slices((train_spectra, train_labels)).shuffle(buffer_size=10000).repeat().batch(
            args.batch_size)
        iter_train = train_ds.make_initializable_iterator()
        batch_train = iter_train.get_next()
        data["train_features"] = batch_train[0]
        data["train_labels"] = tf.one_hot(batch_train[1], args.num_classes)
        data["train_initializer"] = iter_train.initializer
        args.test_every = train_labels.get_shape().as_list()[0] // (args.test_freq * args.batch_size)
        # test_freq: how many times to test in one training epoch

    return data, args


## Make the output dir
# @param args the arguments passed to the software
def make_output_dir(args):
    if os.path.isdir(args.output_path):
        logger.critical("Output path already exists. Please use an other path.")
        raise FileExistsError("Output path already exists.")
    else:
        os.makedirs(args.output_path)
        os.makedirs(args.model_save_dir)
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
    src_dir = '../src'  ## the dir of original files
    save_dir = os.path.join(args.model_save_dir, 'src')
    if not os.path.exists(save_dir):  # if subfolder doesn't exist, should make the directory and then save file.
        os.makedirs(save_dir)

    for filename in os.listdir(src_dir):
        src_file_name = os.path.join(src_dir, filename)
        target_file_name = os.path.join(save_dir, filename)
        try:
            with open(src_file_name, 'r') as file_src:
                with open(target_file_name, 'w') as file_dst:
                    for line in file_src:
                        file_dst.write(line)
        except:
            print('WithCopy Failed!')
        finally:
            print('Done WithCopy File!')
            

