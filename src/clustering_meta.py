
#### read EEG recording and plot it
# import matplotlib
# matplotlib.use('Agg')
# import pqkmeans
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import itertools
import pandas as pd
import pickle
import time
import fnmatch
import scipy.io as scipyio
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import datetime
from scipy.stats import zscore
import ipdb
from collections import Counter
from sklearn import svm
from sklearn import metrics

import matplotlib.pylab as pylab
base_size = 24
params = {'legend.fontsize': base_size - 2,
          'figure.figsize': (12, 10),
          'axes.labelsize': base_size - 2,
          # 'weight' : 'bold',
          'axes.titlesize': base_size,
          'xtick.labelsize': base_size - 3,
          'ytick.labelsize': base_size - 3}
pylab.rcParams.update(params)


def find_files(directory, pattern='Data*.csv', withlabel=True, rand_label=True):
    '''fine all the files in one directory and assign '1'/'0' to F or N files'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            if withlabel:
                if 'BL' in filename:
                    label = 0
                elif 'EPG' in filename:
                    label = 1
                elif rand_label:
                    label = 9
                files.append((os.path.join(root, filename), label))
            else:  # only get names
                files.append(os.path.join(root, filename))

    return files


def random_select_files(data_dir, class_mode='1EPG',
                        pattern='*.csv', num_hours_per_class=20,
                        test_ratio=0, save_dir='KMeans/'):
    """
    Find all the files in one directory with pattern in the filenames and perform train_test_split, and save file names seperately.
    :param args.data_dir: str, the directory of the files
    :param args.class_mode: str, "1EPG (BL-EPG)", "3EPG (BL-earlyEPG-middleEPG-lateEPG)"
    :param args.test_ratio: the ratio of whole data used for testing
    :param args.num_hours_per_class: how many hours to choose for training. If it is 9999, then get all
    :param save_dir:
    :return: test_files, list all the testing files
    :return: train_files, list all the training files
    """
    train_files_labels = []
    test_files_labels = []
    ## get the number of files in folders
    for root, dirnames, fnames in os.walk(data_dir):
        if class_mode == "1EPG":
            if os.path.basename(root) == "BL":
                print("file is found in", root)
                fnames = fnmatch.filter(fnames, pattern)
                train_files_labels, test_files_labels = get_train_test_files_split(root, fnames, test_ratio, train_files_labels, test_files_labels, label=0, num2use=num_hours_per_class)
            elif os.path.basename(root) == "EPG":
                print("file is found in", root)
                fnames = fnmatch.filter(fnames, pattern)
                train_files_labels, test_files_labels = get_train_test_files_split(root, fnames, test_ratio, train_files_labels, test_files_labels, label=1, num2use=num_hours_per_class)


    np.savetxt(os.path.join(save_dir, "test_files-{}.txt".format(os.path.basename(data_dir))), np.array(test_files_labels), fmt="%s", delimiter=",")
    if test_ratio != 1:   # when it is not in test_only case, there are training files
        np.savetxt(os.path.join(save_dir, "train_files.txt"), np.array(train_files_labels), fmt="%s", delimiter=",")
    return train_files_labels, test_files_labels


def get_train_test_files_split(root, fns, ratio, train_list, test_list, label=0, num2use=100):
    """
    Get equal number of files for testing from each folder
    :param fns: list, all file names from the folder
    :param ratio: float, the test file ratio.
    :param train_list: the list for training files
    :param test_list: the list for testing files
    :param label: int, the label need to be assigned to the file
    :param num2use: int, the number of files that you want to use(randomize file selection)
    :return: lists, editted train and test file lists
    """
    np.random.shuffle(fns)
    num_files = min(len(fns), num2use)

    num_test_files = np.ceil(ratio * num_files).astype(np.int)

    train_within_folder = []
    for ind, f in enumerate(fns[0:num_files]):
        if ind < num_test_files:
            test_list.append((os.path.join(root, f), label))
        else:
            train_list.append((os.path.join(root, f), label))
            train_within_folder.append((os.path.join(root, f), label))

    # if ratio != 1.0:
    #     if num2use > len(fns):  # only oversampling training data
    #         theo_num_train = np.int(num2use * (1 - ratio))
    #         repeat_times = theo_num_train // len(train_within_folder) - 1 # already have one round of those files
    #         train_within_folder = train_within_folder * repeat_times + train_within_folder[0: (theo_num_train - (len(train_within_folder)) * repeat_times )]
    #         for ind, fn in enumerate(train_within_folder):
    #             train_list.append((fn[0], label) )

    return train_list, test_list


def oversample_train(features, labels):
    """
    Oversample the minority samples
    :param features:2d array, , [num_samples, num_features]if there is only one feature then use features.reshape(-1, 1)
    :param labels:2d array, [num_samples, 1], use reshape(-1, 1)
    :return:
    """
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=34)
    X_resampled, y_resampled = ros.fit_resample(features, labels)
    features = X_resampled
    labels = y_resampled

    return features, labels


def slide_and_segment(data, win=256, nonoverlap=64, if_flat=True):
    """
    flatten the data and get the sliding windowed data
    :param data: 2d arraym, [batch_size, 512] or [seq_len, width]
    :param win: int, the window length
    :param nonoverlap: the nonoverlap length
    :param if_flat: the nonoverlap length
    :return:
    dd = np.random.randn(1200).reshape(300, 4)
        pe = (num_seg, window, dd.shape[-1])
        des = (dd.itemsize*stride*dd.shape[-1], dd.itemsize*dd.shape[-1], data.itemsize)
        d_x = np.lib.stride_tricks.as_strided(dd, shape=e, strides=des)
        d_x[0] = [[  0   1   2   3]
                 [  4   5   6   7]
                 [  8   9  10  11]
                 [ 12  13  14  15]...]
         expand_x[0] =
         [[ 80  81  82  83]
         [ 84  85  86  87]
         [ 88  89  90  91]
         [ 92  93  94  95]
         [ 96  97  98  99]]
    """
    if if_flat:
        flat = data.reshape(-1)
        num_seg = (flat.size - np.int(win)) // nonoverlap + 1

        # Option 1
        shape = (num_seg, win)  ## done change the num_seq
        strides = (flat.itemsize * nonoverlap, flat.itemsize)
        expand_x = np.lib.stride_tricks.as_strided(flat, shape=shape, strides=strides)
    else:
        # new = data[:, 0:3]
        # num_seg = (new.shape[0] - np.int(window)) // stride + 1
        # shape = (num_seg, window, new.shape[-1])
        # strides = (new.itemsize * stride * new.shape[-1], new.itemsize * new.shape[-1], new.itemsize)
        # ex_new = np.lib.stride_tricks.as_strided(new, shape=shape, strides=strides)
        num_seg = (data.shape[0] - np.int(win)) // nonoverlap + 1
        shape = (num_seg, win, data.shape[-1])
        strides = (data.itemsize * nonoverlap * data.shape[-1], data.itemsize * data.shape[-1], data.itemsize)
        expand_x = np.lib.stride_tricks.as_strided(np.array(data), shape=shape, strides=strides)

    return expand_x, num_seg

    # Option 2
    # indexer = np.arange(window)[None, :] + stride * np.arange(window - stride)[:, None]


def normal_kmeans_fit(features, num_clusters=5, seed=589):
    """
    Apply unsupervised clustering method to get clusters
    :param features:
    :param num_clusters:
    :param save_folder:
    :param seed:
    :return:

    # unsup_model = KMeans(n_clusters=n_clusters)
    # unsup_model.fit(train_features)

    # centroids = unsup_model.cluster_centers_
    # centroids_x = centroids[:, 0]
    # centroids_y = centroids[:, 1]
    # plt.scatter(centroids_x, centroids_y, marker='D', s=50)
    # plt.show()"""
    t1 = time.time()
    print("Cluster {}".format(num_clusters))
    model = KMeans(n_clusters=num_clusters, random_state=seed, max_iter=5000)
    model.fit(features)
    print("{} cluster fitting, {} time".format(num_clusters, time.time() - t1))

    return model


def plot_inertias(inert, plot_range, num_clusters=3, save_folder="KMeans/"):
    """

    :param inert:
    :param plot_range:
    :param num_clusters:
    :param save_folder:
    :return:
    """
    plt.plot(plot_range, inert, '-o')
    plt.savefig(save_folder + "cluster_{}_inertias.png".format(num_clusters))
    plt.close()


def get_crosstab(features, labels, mod,
                 num_clusters=3, save_folder='KMeans/',
                 postfix='test', sorted_index=None,
                 train_bg=None, if_sort_clusters=False):
    """

    :param features:
    :param labels:
    :param mod:
    :param num_clusters:
    :param save_folder:
    :param postfix:
    :param sorted_index: 1d array, the order to plot cluster IDs
    :param train_bg: the histogram of train data as background
    :return:
    """
    labels = labels.astype(np.int)
    names = ["Healthy", "Tumor"]
    colors = ['royalblue', 'violet']
    # fea = np.clip(features, 1e-22, 1)
    pred = mod.predict(features)

    df = pd.DataFrame({"clusters": pred, "true": np.array(names)[labels]})
    ct = pd.crosstab(df["clusters"], df["true"], normalize='columns')
    t2h_ratio = ct['Tumor'].values / ct['Healthy'].values
    sorted_t2h_ratio = np.argsort(t2h_ratio)

    with open(save_folder + '/pred_lbs-true_lbs-for-crosstabs-cluster{}-{}-{}.txt'.format(num_clusters, postfix, sorted_index),
              'wb') as f:
        pickle.dump({"pred_lbs": pred,
                     "true_lbs": labels,
                     "true_names": np.array(names)[labels]
                     }, f)

    # print("{} cluster cross_tabular \n{}".format(num_clusters, ct))
    # ct.plot.bar(stacked=False, title="Crosstabular on {} set".format(postfix))
    if "train" in postfix:
        if if_sort_clusters:
            # sort the cluster by the tumor/healthy ratio from left->right
            sorted_cluster = sorted(zip(ct['Healthy'].values, ct['Tumor'].values), key=lambda x: x[1]/x[0])
            ratio = ct['Tumor'].values / ct['Healthy'].values
            sorted_index = np.argsort(ratio)
            sort_postfix = "sorted"
        else:
            sorted_index = np.arange(num_clusters)
            sort_postfix = "not_sorted"
            sorted_cluster = np.vstack((ct['Healthy'][sorted_index].values, ct['Tumor'][sorted_index].values)).T

        np.savetxt(os.path.join(save_folder, "indices_{}.txt".format(sort_postfix)), sorted_index, fmt='%d', delimiter=',')
        np.savetxt(os.path.join(save_folder, "crosstab_values_{}_in_the_cluster.txt".format(sort_postfix)), sorted_cluster, fmt='%.6f', delimiter=',')
    else:
        assert len(sorted_index) != 0, "Need to pass in sorted index!"
        # assert len(train_bg) != 0, "Need to pass the train background histogram!"
        if len(ct.keys()) < 2:
            if 'Healthy' in ct.keys():
                # there will be nan when where is no segments assigned to a specific cluster
                classification = ct['Healthy'][sorted_index].values
                classification = np.nan_to_num(classification)
                # one file can only from either 'BL', or 'EPG'. fill the other as zeros
                sorted_cluster = np.vstack((classification, np.zeros((classification.size)))).T
            elif 'Tumor' in ct.keys():
                classification = ct['Tumor'][sorted_index].values
                classification = np.nan_to_num(classification)
                sorted_cluster = np.vstack((np.zeros((classification.size)), classification)).T
        else:
            sorted_cluster = np.vstack((ct['Healthy'][sorted_index].values, ct['Tumor'][sorted_index].values)).T

    plt.figure(figsize=[15, 11])
    plt.bar(np.arange(len(sorted_cluster)), np.array(sorted_cluster)[:, 0],
            facecolor=colors[0], edgecolor='k', label='Healthy', width=0.4)
    plt.bar(np.arange(len(sorted_cluster)), -np.array(sorted_cluster)[:, 1],
            facecolor=colors[1], edgecolor='k', label='Tumor', width=0.4)
    # if 'train' not in postfix:
    #     plt.bar(np.arange(len(sorted_cluster)) - 0.2, np.array(train_bg)[:, 0],
    #             facecolor=colors[0], edgecolor='k', alpha=0.12,
    #             linestyle='--', label='train-BL', width=0.4)
    #     plt.bar(np.arange(len(sorted_cluster)) - 0.2, -np.array(train_bg)[:, 1],
    #             facecolor=colors[1], edgecolor='k', alpha=0.12,
    #             linestyle='--', label='train-EPG', width=0.4)
    plt.hlines(0, 0, len(sorted_cluster), 'k')
    for x, y in zip(np.arange(len(sorted_cluster)), np.array(sorted_cluster)[:, 0]):
        if y > 0.0:
            plt.text(x, y , '%.2f' % (y*10), ha='center', va='bottom', fontsize=base_size-4)

    for x, y in zip(np.arange(len(sorted_cluster)), np.array(sorted_cluster)[:, 1]):
        if y > 0.0:
            plt.text(x, -y, '%.2f' % (y*10), ha='center', va='top', fontsize=base_size-4)

    plt.yticks(()),
    plt.ylabel("frequency [%]", fontsize=base_size-2)
    plt.xlabel("cluster IDs", fontsize=base_size-2)
    plt.title("Frequency count. Healthy to tumor {}".format(sorted_t2h_ratio))
    plt.xlim(-.5, len(sorted_cluster))
    plt.legend(fontsize=base_size-1, loc="best")
    plt.xticks(np.arange(len(sorted_cluster)), np.array(sorted_index), fontsize=base_size-2)
    plt.savefig(save_folder+"/cluster_{}_crosstab_{}_{}.png".format(num_clusters, postfix, sorted_index), format="png")
    plt.savefig(save_folder+"/cluster_{}_crosstab_{}_{}.pdf".format(num_clusters, postfix, sorted_index), format="pdf")
    # plt.savefig(os.path.join(save_folder, "cluster_{}_crosstab_{}_EPG{}.pdf".format(num_clusters, postfix, sorted_index)), format="pdf")
    plt.close()

    return pred, ct, sorted_index


def plot_one_cluster(features, labels, pred,
                     count=[100, 10], num_clusters=3,
                     cluster_id=0,
                     save_folder="KMeans/",
                     postfix='test',
                     num_figs=1):
    """

    :param features:
    :param labels:
    :param pred:
    :param count:
    :param num_clusters:
    :param cluster_id:
    :param win:
    :param nonoverlap:
    :param save_folder:
    :param postfix:
    :param semilogx: if True, then indicates FFT clustering, then plot log
    :param freq:
    :return:
    """
    inds = np.where(pred == cluster_id)[0]
    fea = features[inds]
    row = 8
    col = 3

    num_figs = min(num_figs, fea.shape[0] // (row * col))

    rand_ind = np.random.choice(fea.shape[0], (min(row * col * num_figs, fea.shape[0])), replace=False)
    fea_plot = fea[rand_ind]

    # if np.sum(labels[inds] == 0) > (row * col // 2):   #here the data is not shuffled
    #     rand_inds = np.random.choice(np.sum(labels[inds] == 0), (min(row*col//2, np.sum(labels[inds] == 0))))
    for j in range(num_figs):
        fig, axs = plt.subplots(row, col, 'col', 'row', figsize=[20, 13])
        for i in range(j * row * col, (j + 1) * row * col):
            axs[(i - j * row * col) // col, np.mod(i, col)].plot(np.arange(fea_plot[i].size) / 512, fea_plot[i])

            fig.text(0.5, 0.92,"{}-clusters No. {} cluster , count-({})".format(
                num_clusters, cluster_id, (count*10000).astype(int)/10000.),
                     horizontalalignment="center",
                     color="k", size=22, verticalalignment='top')

        fig.text(0.5, 0.08, "index [a.u.]", horizontalalignment="center", color="k", size=20, verticalalignment='top')
        fig.text(0.08, 0.5, "normalized value [a.u.]", horizontalalignment="center", color="k", size=20, rotation='vertical')
        fig.subplots_adjust(wspace=0)
        fig.subplots_adjust(hspace=0)
        plt.savefig(os.path.join(save_folder, "{}_clusters_-label_{}-{}-fig-{}.png" .format(num_clusters, cluster_id, postfix, j)))
        # plt.savefig(os.path.join(save_folder, "{}_clusters_-label_{}-{}-fig-{}.pdf" .format(num_clusters, cluster_id, postfix, j)), format='pdf')
        plt.close()

        print("cluster-{}-fig-{}finished!".format(cluster_id, j), "foler: ", os.path.join(save_folder, "{}_clusters_-label_{}-{}-fig-{}.png" .format(num_clusters, cluster_id, postfix, j)))

        # Save mean pattern in each cluster
        crosstab_count = (count * 10000).astype(int) / 10000.
        plot_mean_spec_in_cluster(fea, cluster_id, num_clusters, postfix, crosstab_count=crosstab_count, save_folder=save_folder)


def plot_mean_spec_in_cluster(fea, cluster_id, num_clusters, postfix, crosstab_count=[10, 100], save_folder="/results"):
    """

    :param fea:
    :param cluster_id:
    :param num_clusters:
    :param postfix:
    :param save_folder:
    :return:
    """

    spec_mean = np.mean(fea, axis=0)
    spec_std = np.std(fea, axis=0)
    data = np.hstack((spec_mean.reshape(-1, 1), spec_std.reshape(-1, 1)))
    plt.figure()
    plt.plot(np.arange(spec_mean.size), spec_mean, 'royalblue')
    # plt.errorbar(np.arange(spec_mean.size), spec_mean, spec_std, alpha=0.25)
    plt.fill_between(np.arange(spec_mean.size), spec_mean - spec_std, spec_mean + spec_std, alpha=0.25, facecolor='royalblue')
    plt.xlabel("index")
    plt.title("{}-clusters No. {} cluster, count {}".format(num_clusters, cluster_id, crosstab_count))
    ylabel = "normalized value [a.u.]"
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(save_folder, "{}_clusters_label_{}-spec-{}-mean.png".format(num_clusters, cluster_id, postfix)), format="png")
    plt.savefig(os.path.join(save_folder, "{}_clusters_label_{}-spec-{}-mean.pdf".format(num_clusters, cluster_id, postfix)), format="pdf")
    plt.close()
    np.savetxt(os.path.join(save_folder, "{}_clusters_label_{}-spec-{}-mean.csv".format(num_clusters, cluster_id, postfix)), data, header='mean,std', delimiter=',')



def get_spectrogram(data, fs=512, nperseg=2048, nonoverlap=256, dim2use=200):
    _, _, stfts = signal.stft(data, fs=fs, nperseg=nperseg, noverlap=nonoverlap)
    power_spec = np.real(stfts * np.conj(stfts))  # A float32 Tensor of shape [batch_size, time_bins, fft_bins].
    power_spec = np.log(power_spec + 1e-13)
    # plt.imshow(power_spec, interpolation='nearest', cmap='viridis', aspect='auto', origin='lower')
    return power_spec[0: dim2use]


def get_slide_seg_from_files(fn_with_l, win=2560, nonoverlap=2560):
    """
    :param fn_with_l: list of tuple (filename, label), if only one fn_with_l then use [fn_with_l] as input
    :param win: int
    :param nonoverlap: int
    :param num_segments: int
    :return:
    """
    features = np.empty((0, win))
    labels = np.empty((0))
    for j in range(len(fn_with_l)):
        ret, label = get_slide_seg_from_one_file(fn_with_l[j][0], fn_with_l[j][1],
                                                 win=win,
                                                 nonoverlap=nonoverlap)

        features = np.vstack((features, ret))
        labels = np.append(labels, label).astype(np.int)
    features = features.reshape([-1, win])
    labels = labels.reshape(-1).astype(np.int)

    return features, labels


def get_slide_seg_from_one_file(filename, label, win=2056, nonoverlap=2056):
    """
    Get the sliding segments from one file
    :param filename: str
    :param label: int
    :param win:
    :param nonoverlap:
    :return:
    """
    data = pd.read_csv(filename, header=None).values
    features = data[:, 2:].reshape(-1).astype(np.float)
    slide_features, num_segments = slide_and_segment(features, win=win, nonoverlap=nonoverlap)
    slide_labels = np.repeat(label, num_segments)

    return slide_features, slide_labels


def k_mean_distance(data, center, i_centroid, cluster_labels):
    """
    Get the distance of data towards a cluster center
    :param data:
    :param center:
    :param i_centroid:
    :param cluster_labels:
    :return:
    """
    # distances = [np.sqrt((x - cx) ** 2 + (y - cy) ** 2) for (x, y) in data[cluster_labels == i_centroid]]
    distances = np.linalg.norm(data-center)
    return distances


def evaluate_clusters(features, labels, mod, num_clusters=4,
                      save_folder='/results',
                      postfix='test', if_save_data=False,
                      sorted_index=None,
                      train_bg=None,
                      if_sort_clusters=False):
    """
    :param features: array
    :param labels: array,
    :param mod: model
    :param num_clusters: ind
    :param seed: ind
    :param save_folder: str
    :param postfix: str
    :param freq: 1d array, the freq space to plot
    :param if_save_data: bool
    :param sorted_index: 1d array, how to arrange the clusters in the crosstab
    :param if_sort_clusters: bool, whether to order the cluster by some creteriion
    :return:
    """

    # # get the cross tabular
    sorted_index = None if "train" in postfix else sorted_index
    pred, crosstab, sorted_index = get_crosstab(features, labels, mod,
                                                num_clusters=num_clusters,
                                                save_folder=save_folder,
                                                postfix=postfix,
                                                sorted_index=sorted_index,
                                                train_bg=train_bg,
                                                if_sort_clusters=if_sort_clusters)


    # Save data in the cluster
    if if_save_data:
        for j in range(num_clusters):
            data_in_cluster = np.array(features[pred == j])
            np.savetxt(os.path.join(save_folder, 'data',
                                    "{}_clusters_No.{}_{}_data"
                                    .format(num_clusters, j, postfix)),
                       data_in_cluster,
                       fmt='%.2f', delimiter=',')


    return pred, crosstab, sorted_index


def load_pickle_model(model_dir):
    # Load from file
    with open(model_dir, 'rb') as file:
        pickle_model = pickle.load(file)

    return pickle_model


def check_if_artifacts(data, thr_arti=0.12, thr_deriv=0.1):
    """
    CHeck whether a signal is artifacts or not.
    :param data: 2d array
    :param thr_arti: float, threshold for artifacts: what is the tolerance threshold of the artifacts percentage
    :param threshold: float, threshold for sec_derivitive: how close the second derivitive need to be in order to say that is a flat line.
    :return: arti_anno: the annotation of whether is an artifacts or not
    """
    dx = ((data[:, 1:] - data[:, 0:-1]) * 1000).astype(np.int) / 1000.
    ddx = (np.abs(dx[:, 1:] - dx[:, 0:-1]) < thr_deriv).astype(int)
    arti_percent = np.sum(ddx, axis=1) * 1.0 / ddx.shape[1]
    arti_anno = arti_percent >= thr_arti

    return arti_anno, arti_percent


def get_artf_percentage_in_dir(dirs, thr_arti=0.12, thr_deriv=0.5, win=2056, nonoverlap=2056, num_segments=None):
    """

    :param dirs:
    :param thr_arti: float, threshold for artifacts: what is the tolerance threshold of the artifacts percentage
    :param thr_deriv: float, threshold for sec_derivitive: how close the second derivitive need to be in order to say that is a flat line.
    :return:
    """
    for data_dir in dirs:
        print(data_dir)
        files = find_files(data_dir, pattern='*filter.csv', withlabel=True, rand_label=False)
        file_wise_percent = []
        rat_wise_percent = 0.
        print(files)
        for fn in files:
            label = 0 if 'BL' in os.path.basename(fn[0]) else 1
            # label = 0 if 'BL' in os.path.basename(fn[0]) else 1
            anno, percent, features, labels = get_artf_percentage_in_one_file(fn[0], label=label, thr_arti=thr_arti, thr_deriv=thr_deriv, win=win, nonoverlap=nonoverlap)
            file_wise_percent.append((fn[0], np.sum(anno) / anno.size))
            rat_wise_percent += np.sum(anno) / anno.size

            arti_free_inds = np.where(anno == False)[0]

            arti_free_data = features[arti_free_inds]
            arti_free_labels = labels[arti_free_inds].astype(int).reshape([-1, 1])

            # new_data = np.hstack((arti_free_labels, arti_free_data))

            print("Modified: ", fn[0][0:-4])
            np.savetxt(fn[0][0:-4] + '-5s-{}.csv'.format(len(arti_free_inds)), arti_free_data, fmt="%.3f", delimiter=',')

    sorted_percent = sorted(list(file_wise_percent), key=lambda x: x[1])
    np.savetxt(
        os.path.join(data_dir,
                     "{}_artifacts_statistics_{:.4f}_sorted.txt"
                     .format(os.path.basename(data_dir),
                             rat_wise_percent / len(files))),
        np.array(sorted_percent), fmt="%s", delimiter=",")
    print("Sorted folder: ", data_dir)


def get_artf_percentage_in_one_file(filename, label=None, thr_arti=0.12, thr_deriv=0.5, win=2056, nonoverlap=2056):
    """
    Get the artifacts percentage in one hour file
    :param filename:
    :param thr_arti: float, threshold for artifacts: what is the tolerance threshold of the artifacts percentage
    :param thr_deriv: float, threshold for sec_derivitive: how close the second derivitive need to be in order to say that is a flat line.
    :return:
    """
    if label is not None:
        features, labels = get_slide_seg_from_one_file(filename, label, win=win, nonoverlap=nonoverlap)
    else:
        features, labels = get_slide_seg_from_one_file(filename, 'None', win=win, nonoverlap=nonoverlap)
    anno, percent = check_if_artifacts(features, thr_arti=thr_arti, thr_deriv=thr_deriv)

    return anno, percent, features, labels


def exclude_arti_from_files_in_dirs(dirs, threshold=0.0):
    """
    exclude the artifacts segments (5s) from the file and save the rest
    :param dirs:
    :param threshold:
    :return:
    """
    for dir in dirs:
        txts = find_files(dir, pattern="*sorted.txt", withlabel=False)
        print("Load txt: ", txts)
        fns_with_percent = list(pd.read_csv(txts[0]).values)
        exc_inds = np.where(np.array(fns_with_percent)[:, 1] > threshold)[0]
        np.savetxt(
            os.path.join(dir,
                         "{}_modified_files_{}_threshold.txt"
                         .format(os.path.basename(dir), threshold)),
            np.array(fns_with_percent)[exc_inds], fmt="%s", delimiter=",")
        for ind in exc_inds:
            if os.path.isfile(fns_with_percent[ind][0]):

                features, labels = get_slide_seg_from_one_file(fns_with_percent[ind][0], fns_with_percent[ind][1], win=2560, nonoverlap=2560)
                anno, percent = check_if_artifacts(features, thr_arti=0.10, thr_deriv=0.5)
                new_features = features[np.where(anno == False)[0]]
                np.savetxt(fns_with_percent[ind][0][0:-4] + '-arti-free.csv',
                           np.array(new_features.reshape(-1, 512)),
                           fmt="%.2f", delimiter=",")
                print("Modified file: ", fns_with_percent[ind][0])

def exclude_arti_from_data(features, labels, thr_arti=0.12, thr_deriv=0.5):
    """

    :param data:
    :param thr_arti:
    :param thr_deriv:
    :return:
    """
    anno, percent = check_if_artifacts(features, thr_arti=thr_arti, thr_deriv=thr_deriv)
    arti_inds = np.where(anno == False)[0]
    clean_features = features[arti_inds]
    clean_labels = labels[arti_inds]

    return clean_features, clean_labels


def get_anno_for_one_file(filename, seg_anno, save_dir='results/'):
    """
    Generate annotation file for the data file
    :param filename: str
    :param seg_anno: array of True or false
    :param save_dir:
    :return:
    """
    start_y = np.int(os.path.basename(filename).split('-')[0])
    start_mon = np.int(os.path.basename(filename).split('-')[1])
    start_d = np.int(os.path.basename(filename).split('-')[2].split("T")[0])
    start_h = np.int(os.path.basename(filename).split('-')[2].split("T")[1])
    start_m = np.int(os.path.basename(filename).split('-')[3])
    start_s = np.int(os.path.basename(filename).split('-')[4])

    annotations = []

    inds = np.where(seg_anno == True)[0]
    if len(inds) > 0:
        xcoords = list(np.where((inds[1:] - inds[0:-1]) > 1)[0] + 1)
        for ind in np.arange(len(xcoords)) * 2:
            xcoords.insert(ind, xcoords[ind] - 1)
        start_end_inds = [0] + xcoords + [len(inds) - 1]

        for j in range(0, len(start_end_inds), 2):  #
            print("xc: ", j)  # [0] + xcoords + [len(vis_h)-1]
            incre_sec = np.mod(inds[start_end_inds[j]], 60)
            incre_min = inds[start_end_inds[j]] // 60
            if start_s + incre_sec < 60:
                s = start_s + incre_sec
            else:
                s = np.mod(start_s + incre_sec, 60)
                incre_min += 1
            if start_m + incre_min < 60:
                m = start_m + incre_min
                incre_h = 0
            else:
                m = np.mod(start_m + incre_min, 60)
                incre_h = 1

            h = start_h + incre_h

            d = start_d if h < 24 else (start_d + 1)
            mon = start_mon + 1 if (start_mon in [1, 3, 5, 7, 8, 10, 12] and d > 31) \
                                   or (start_mon in [4, 6, 9, 11] and d > 30) \
                                   or (start_mon == 2 and d > 28) \
                else start_mon
            y = start_y
            duration = inds[start_end_inds[j + 1]] - inds[start_end_inds[j]] + 1
            anno_txt = 'poor reception'
            annotation = "{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:03f} {:03f} {}"\
                .format(y, mon, d, h, m, s, duration, anno_txt)
            annotations.append(annotation)
        np.savetxt(os.path.join(save_dir, os.path.basename(filename)[0:-4])+ "_artifacts_annotation.txt",
            np.array(annotations), header="onset duration annotation", fmt="%s", delimiter=",", newline='\n')
    else:
        print("There is no artifacts in file {}".format(filename[0]))


def train_clustering(features, labels, start=10, end=11,
                     seed=589, save_folder='KMeans/',
                     if_sort_clusters=False):
    """
    The whole training process in clustering
    :param train_f_with_l:
    :param train_f_with_l:
    :param window:
    :param stride:
    :param kmeans_mode: str, ['normal', 'pq']
    :param encoder: if kmeans_mode is pq, then pass in the pqencoder
    :param data_mode: str, ['raw', 'spectra']
    :param start:
    :param end:
    :return:predictions,
    :return:crosstabs,

    """
    inertias = []
    # Start training given number of clusters to use
    for n_clusters in range(start, end):
        model = normal_kmeans_fit(features, num_clusters=n_clusters, seed=seed)

        # save the model
        pkl_filename = "{}_cluster_model-random_seed{}.pkl".format( n_clusters, seed)
        k_cluster_folder = os.path.join(save_folder, "{}-cluster".format(n_clusters))
        check_make_dir(k_cluster_folder, ['data', 'plots'])
        with open(os.path.join(k_cluster_folder, pkl_filename), 'wb') as file:
            pickle.dump(model, file)
        print("pkl file saved: ", os.path.join(save_folder, pkl_filename))


        inertias.append(model.inertia_)
        plt.plot(np.arange(start, n_clusters+1), np.array(inertias), 'o-')
        plt.xlabel("number of clusterse")
        plt.ylabel("within cluster distance")
        plt.savefig(os.path.join(k_cluster_folder, "inertials-cluster-{}.png".format(n_clusters)), format='png')
        plt.close()
        np.savetxt(os.path.join(save_folder, "inertials-cluster-{}.csv".format( n_clusters)), np.array(inertias), delimiter=',')

        predictions, crosstabs, sorted_inds = \
            evaluate_clusters(features, labels, model,
                              num_clusters=n_clusters,
                              save_folder=k_cluster_folder,
                              postfix="train",
                              if_save_data=True,
                              sorted_index=None,
                              if_sort_clusters=if_sort_clusters)

        plot_cluster_examples(features, labels, crosstabs,
                              predictions,
                              num_clusters=n_clusters,
                              postfix="train",
                              save_folder=k_cluster_folder,
                              num_figs=3)


def plot_cluster_examples(features, labels, crosstabs,
                          predictions,
                          num_clusters=None,
                          postfix='filename',
                          save_folder='KMeans/',
                          num_figs=1):
    """
    Plot examples in the cluster both in raw data and extracted features
    :param features: 2d array, whatever used for clustering
    :param labels: 1d array,
    :param crosstabs: 2d array
    :param predictions: 1d array, predicted cluster assignment for each sample
    :param raw_features:
    :param num_clusters:
    :param data_mode:
    :param freq:
    :return:
    """
    # Save examples from different clusters
    for ind, cluster_id in zip(np.arange(crosstabs.shape[0]), np.array(crosstabs.index)):
        plot_one_cluster(features, labels, predictions,
                      count=crosstabs.values[ind],
                      num_clusters=num_clusters,
                      cluster_id=cluster_id,
                      save_folder=os.path.join(save_folder, 'plots'),
                      postfix=postfix,num_figs=num_figs)


    print("{} cluster fitting finished!".format(num_clusters))
    print("All finished!")


def check_make_dir(root, subfolders):
    """

    :param root: str, root dir
    :param subfolders: list, subfolders you want to creat
    :return:
    """
    if not os.path.exists(root):
        os.makedirs(root)
    for ind, sub in enumerate(subfolders):
        if not os.path.exists(os.path.join(root, sub)):
            os.makedirs(os.path.join(root, sub))



def load_and_test(data_dir, pkl_dir="/results"):
    """

    :param pkl_dir:
    :param test_dirs:
    :param win:
    :param nonoverlap:
    :param data_mode:
    :return:
    """
    n_clusters = os.path.basename(pkl_dir).split("_")[0]
    model = load_pickle_model(pkl_dir)
    sorted_index = np.loadtxt(os.path.join(os.path.dirname(pkl_dir), "indices_not_sorted.txt")).astype(np.int)
    sorted_cluster = np.loadtxt(os.path.join(os.path.dirname(pkl_dir), "crosstab_values_not_sorted_in_the_cluster.txt"), delimiter=',')

    save_dir = os.path.dirname(pkl_dir) + '/{0:%Y-%m-%dT%H-%M-%S}-test'.format(datetime.datetime.now())
    check_make_dir(save_dir, ['data', 'plots'])

    # evaluate files ordered with time and save the predicted
    spec, lbs, ids = load_data(data_dir, num_classes=2)

    ## Get distances of data to cluster centeres
    centroids = model.cluster_centers_
    distances = []
    sub_dist = []
    for i, center in enumerate(centroids):
        mean_distance = np.linalg.norm(spec-center, axis=1)
        sub_dist.append(mean_distance)
        distances = np.array(sub_dist).T
    np.savetxt(os.path.join(save_dir, "distances_to_centroids.csv"), distances, delimiter=",", header="dis2cl0,dis2cl1,dis2cl2,dis2cl3,dis2cl4,dis2cl5,dis2cl6", fmt="%.5f")
    np.savetxt(os.path.join(save_dir, "distances_to_healthy_tumore_centers.csv"), dist2, delimiter=",", header="dis2cl0_healthy,dis2cl3_tumor", fmt="%.5f")

    # tsne vis. the distance to two typical clusters
    dist2 = np.vstack((distances[0], distances[3])).T
    ind0 = np.where(lbs == 0)[0]
    ind1 = np.where(lbs == 1)[0]
    plt.figure(figsize=[12, 10])
    plt.scatter(distances[0][ind0], distances[3][ind0], color="royalblue", label="healthy")
    plt.scatter(distances[0][ind1], distances[3][ind1], color="violet", label="tumor")
    plt.legend()
    plt.savefig(save_dir + '/distances_healthy_tumor.png', format="png")
    plt.close()


    # tsne vis the distances
    X_embedded2 = TSNE(n_components=2).fit_transform(spec)
    fig = plt.figure(figsize=[12, 10])
    ax = fig.add_subplot(111)
    ax.scatter(X_embedded2[:, 0][ind0], X_embedded2[:, 1][ind0], color="royalblue", alpha=0.5, marker="o"),
    ax.scatter(X_embedded2[:, 0][ind1], X_embedded2[:, 1][ind1], color="violet", alpha=0.5, marker="^"),
    plt.savefig(save_dir + '/tsne_spec_2d.png', format="png"),
    plt.close()

    predictions, crosstabs, _ = \
        evaluate_clusters(spec,
                          lbs,
                          model,
                          num_clusters=n_clusters,
                          save_folder=save_dir,
                          postfix='test',
                          if_save_data=False,
                          sorted_index=sorted_index,
                          train_bg=sorted_cluster)

    # plot_cluster_examples(spec, lbs, crosstabs,
    #                       predictions,
    #                       num_clusters=n_clusters,
    #                       save_folder=save_dir,
    #                       postfix="test-lout40-5",
    #                       num_figs=3)

    count = dict(Counter(list(ids)))
    for id in count.keys():
        plt.figure(figsize=[12, 8])
        id_inds = np.where(ids == 514)[0]
        vote_label = np.sum(lbs[id_inds]) * 1.0 / id_inds.size
        vote_pred = np.sum(predictions[id_inds][:, 1]) / id_inds.size

        label_of_id = 0 if vote_label < 0.5 else 1
        pred_of_id = 0 if vote_pred < 0.5 else 1

        color = "slateblue" if label_of_id == pred_of_id else "r"
        ax = plt.subplot(1, 1, 1)
        pred_hist = plt.hist(np.array(predictions[id_inds][:, 1]), align='mid', bins=10, range=(0.0, 1.0), color=color, label="predicted")
        ymin, ymax = ax.get_ylim()
        ymax += 1
        plt.vlines(0.5, ymin, ymax, colors='k', linestyles='--')
        plt.text(0.25, ymax, str(np.int(np.sum(predictions[id_inds][:, 1] < 0.5))), fontsize=16)
        plt.text(0.75, ymax, str(np.int(np.sum(predictions[id_inds][:, 1] >= 0.5))), fontsize=16)
        plt.legend()
        plt.ylabel("frequency")
        plt.xlabel("probability of classified as class 1")
        plt.title("True label {} - pred as {} / (in total {} voxels for id {})".format(label_of_id, pred_of_id, id_inds.size, id))
        plt.tight_layout()
        plt.savefig(save_dir + '/prob_distri_of_id_{}.png'.format(id), format="png")
        plt.close()


    # collect the labels for each patient and make the classification
    print("Done")



def load_data(data_dir, num_classes=2):
    """

    :param data_dir:
    :param num_classes:
    :return:
    """
    mat = scipyio.loadmat(data_dir)["DATA"]
    spectra = mat[:, 2:]
    labels = mat[:, 1]
    ids = mat[:, 0]
    sample_ids = np.arange(len(mat))
    spectra = zscore(spectra, axis=1)
    ## following code is to get only label 0 and 1 data from the file. TODO: to make this more easy and clear
    if num_classes - 1 < np.max(labels):
        need_inds = np.empty((0))
        for class_id in range(num_classes):
            need_inds = np.append(need_inds, np.where(labels == class_id)[0])
        need_inds = need_inds.astype(np.int32)
        spectra = spectra[need_inds]
        labels = labels[need_inds]
        ids = ids[need_inds]

    return spectra, labels, ids


def plot_from_certain_ids(data_dir, fn):
    """
    from the sample_ids provided by fn, plot the mean of the samples
    :param fn:
    :param lbs:
    :param spec:
    :return:
    """
    mat = scipyio.loadmat(data_dir)["DATA"]
    spec = mat[:, 2:]
    lbs = mat[:, 1]
    ids = mat[:, 0]

    data = pd.read_csv(fn, header=0).values
    data_ids = data[:, 0].astype(np.int)
    certain_data = spec[data_ids]
    certain_lb = lbs[data_ids]
    epoch = os.path.basename(fn).split("_")[-2]
    ind0 = np.where(certain_lb == 0)[0]
    ind1 = np.where(certain_lb == 1)[0]
    assert (certain_lb == lbs[data_ids]).all(), "labels do not match!"
    spec0 = certain_data[ind0]
    spec1 = certain_data[ind1]
    mean0 = np.mean(spec0, axis=0)
    std0 = np.std(spec0, axis=0)
    mean1 = np.mean(spec1, axis=0)
    std1 = np.std(spec1, axis=0)
    fig, axs = plt.subplots(2, 1, 'col', figsize=[12, 10])
    axs[0].plot(spec0.T, 'lightskyblue', alpha=0.5)
    axs[0].plot(spec0[0], 'lightskyblue', alpha=0.5, label="individual healthy samples")
    axs[0].plot(mean0, 'royalblue', linewidth=2.0, label="mean of healthy samples")
    axs[1].plot(spec1.T, 'violet', alpha=0.5)
    axs[1].plot(spec1[0], 'violet', alpha=0.5, label="individual tumor samples")
    axs[1].plot(mean1, 'magenta', linewidth=2.0, label="mean of tumor samples")
    axs[0].legend()
    axs[1].legend()
    plt.xlabel("metabolite indices"),
    plt.ylabel("normalized amplitude"),
    plt.savefig(fn[0:-4] + ".png", format="png"),
    plt.close()


def cluster_spectra(data_dir):
    """
    CLuster the presaved certain examples from all the rats together.
    :param win:
    :param nonoverlap:
    :return:
    """
    # fn = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/saved_certain/2019-10-07T17-02-18-data-20190325-3class_lout40_train_test_data5-1d-class-2-Res_ECG_CAM-relu-aug_ops_meanx5-0.7-train--auc0.715/certains/certain_data_train_epoch_20_num4407.csv"
    #
    # plot_from_certain_ids(data_dir, fn)
    spec, lbs, ids = load_data(data_dir, num_classes=2)
    data_mode = 'spec'
    postfix = 'whole-20190325'
    root_folder = '../KMeans/metabolites_clustering-{}-{}/{}' \
        .format(data_mode, postfix, '{0:%Y-%m-%dT%H-%M-%S}'.format(datetime.datetime.now()))
    # Get data
    print("Got all data", spec.shape, 'label shape:', lbs.shape, '\n save_foler: ', root_folder)

    train_clustering(spec, lbs,
                     start=7, end=8, seed=589,
                     save_folder=root_folder,
                     if_sort_clusters=False)
    print("clustering is Done!")


def SVM_classifier(train_data_dir, test_data_dir):
    spec, lbs, ids = load_data(train_data_dir, num_classes=2)
    data_mode = 'spec'
    postfix = 'whole-20190325'
    root_folder = '../SVM/metabolites_SVM-{}-{}/{}' \
        .format(data_mode, postfix, '{0:%Y-%m-%dT%H-%M-%S}'.format(datetime.datetime.now()))
    # Get data
    print("Got all data", spec.shape, 'label shape:', lbs.shape, '\n save_foler: ', root_folder)
    check_make_dir(root_folder, [])

    clf = svm.SVC()
    clf.fit(spec, lbs)

    test_spec, test_lbs, test_ids = load_data(test_data_dir, num_classes=2)
    pred = clf.predict(test_spec)

    auc = metrics.roc_auc_score(test_lbs, pred)
    np.savetxt(os.path.join(root_folder, "svm_pred_labels_auc_{:.3f}.txt".format(auc)), np.array(pred), fmt="%d", delimiter=",")


#######################################################################################
if __name__ == "__main__":

    process = "train"



    if process == "train":
        data_dir = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data5.mat"
        cluster_spectra(data_dir)

    elif process == "test":
        data_dir = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_val_data5.mat"
        pkl_dir = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/src/KMeans/metabolites_clustering-spec-lout40-5/2019-10-01T11-58-13/7-cluster/7_cluster_model-random_seed589.pkl"
        load_and_test(data_dir, pkl_dir=pkl_dir)

    elif process == "svm":
        train_data_dir = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data5.mat"
        test_data_dir = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_val_data5-2class_human_performance844_with_labels.mat"

        SVM_classifier(train_data_dir, test_data_dir)

