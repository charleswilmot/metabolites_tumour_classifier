import datetime
import os
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from scipy.stats import zscore
import scipy.io as scipyio
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
from sklearn import metrics
from sklearn.manifold import TSNE

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


def cluster_spectra(data_dir):
    """
    CLuster the presaved certain examples from all the rats together.
    :param win:
    :param nonoverlap:
    :return:
    """
    spec, lbs, ids = load_data(data_dir, num_classes=2)
    data_mode = 'spec'
    postfix = 'lout40-5'
    root_folder = 'KMeans/metabolites_clustering-{}-{}/{}' \
        .format(data_mode, postfix, '{0:%Y-%m-%dT%H-%M-%S}'.format(datetime.datetime.now()))
    # Get data
    print("Got all data", spec.shape, 'label shape:', lbs.shape, '\n save_foler: ', root_folder)


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
    pred = mod.predict_proba(features)
    pred_lbs = np.argmax(pred, axis=1)

    df = pd.DataFrame({"clusters": pred_lbs, "true": np.array(names)[labels]})
    ct = pd.crosstab(df["clusters"], df["true"], normalize='columns')
    t2h_ratio = ct['Tumor'].values / ct['Healthy'].values
    sorted_t2h_ratio = np.argsort(t2h_ratio)

    save_data = np.vstack((pred_lbs, labels)).T
    np.savetxt(os.path.join(save_dir, "pred_cluster+labels.csv"), save_data, header="pred_cluster,labels", delimiter=",", fmt="%d")

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
    plt.hlines(0, 0, len(sorted_cluster), 'k')
    for x, y in zip(np.arange(len(sorted_cluster)), np.array(sorted_cluster)[:, 0]):
        if y > 0.0:
            plt.text(x, y , '%.2f' % (y*100), ha='center', va='bottom', fontsize=base_size-4)

    for x, y in zip(np.arange(len(sorted_cluster)), np.array(sorted_cluster)[:, 1]):
        if y > 0.0:
            plt.text(x, -y, '%.2f' % (y*100), ha='center', va='top', fontsize=base_size-4)

    plt.yticks(()),
    plt.ylabel("frequency [%]", fontsize=base_size-2)
    plt.xlabel("cluster IDs", fontsize=base_size-2)
    plt.title("Frequency count. Healthy to tumor {}".format(sorted_t2h_ratio))
    plt.xlim(-.5, len(sorted_cluster))
    plt.legend(fontsize=base_size-1, loc="best")
    plt.xticks(np.arange(len(sorted_cluster)), np.array(sorted_index), fontsize=base_size-2)
    plt.savefig(save_folder+"/cluster_{}_crosstab_{}_{}.png".format(num_clusters, postfix, sorted_index))
    # plt.savefig(os.path.join(save_folder, "cluster_{}_crosstab_{}_EPG{}.pdf".format(num_clusters, postfix, sorted_index)), format="pdf")
    plt.close()

    return pred_lbs, pred, ct, sorted_index


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
    plt.savefig(os.path.join(save_folder,
                             "{}_clusters_label_{}-spec-{}-mean.png".format(num_clusters, cluster_id, postfix)))
    plt.close()
    np.savetxt(os.path.join(save_folder, "{}_clusters_label_{}-spec-{}-mean.csv".format(num_clusters, cluster_id, postfix)), data, header='mean,std', delimiter=',')


def load_and_split_data(data_dir, num_classes=2):
    """

    :param data_dir:
    :return:
    """
    spec, lbs, ids = load_data(data_dir, num_classes=num_classes)
    X_train, X_val, Y_train, Y_val = train_test_split(spec, lbs, test_size=0.25, random_state=132)
    return X_train, X_val, Y_train, Y_val


def train_GMM(fit_data, num_clusters=5, save_dir="restuls/"):
    GMM = GaussianMixture(n_components=num_clusters, random_state=589).fit(fit_data)  # Instantiate and fit the model
    print('Converged:', GMM.converged_)  # Check if the model has converged
    means = GMM.means_
    covariances = GMM.covariances_
    # plot covariance matrix
    for ind in range(num_clusters):
        plt.figure()
        plt.imshow(covariances[ind], cmap="viridis", interpolation="nearest", aspect="auto")
        plt.xlabel("metabolite indices")
        plt.ylabel("metabolite indices")
        plt.title("Covariance matrix of cluster {}".format(ind))
        plt.savefig(os.path.join(save_dir, "covariance{}.png".format(ind)))
        plt.close()
    return GMM


def plot_auc_curve(y_true, y_pred, postfix="test", save_dir="results/"):
    """
    Plot AUC curve
    :param args:
    :param data:
    :return:
    """
    f = plt.figure()
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)  # input the positive label's prob distribution
    auc = metrics.roc_auc_score(y_true, y_pred)
    plt.plot(fpr, tpr, label="auc={0:.4f}".format(auc))
    plt.legend(loc=4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    f.savefig(save_dir + '/AUC_curve-auc_{:.4}-{}.png'.format(auc, postfix))
    np.savetxt(save_dir + '/AUC_curve-auc_{:.4}-{}.csv'.format(auc, postfix),
               np.hstack((y_true.reshape(-1,1), y_pred.reshape(-1,1))), fmt="%.8f", delimiter=',', header="labels,tumor_prob")
    plt.close()
#########################################################
if __name__ == "__main__":
    num_clusters = 6

    save_dir = "EM_results" + '/cluster_{0}/{1:%Y-%m-%dT%H-%M-%S}'.format(num_clusters, datetime.datetime.now())
    check_make_dir(save_dir, ["data", "plots"])
    data_dir = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data5.mat"
    test_data_dir = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_val_data5.mat"

    # 0. load and split data

    X_train, X_val, Y_train, Y_val = load_and_split_data(data_dir, num_classes=2)

    test_spec, test_lbs, test_ids = load_data(test_data_dir, num_classes=2)

    # 1. train a GMM
    GMM = train_GMM(X_train, num_clusters=num_clusters, save_dir=save_dir)

    # get cross tab
    pred_cluster, pred_prob, ct, sorted_index = \
        get_crosstab(X_train, Y_train, GMM,
                     num_clusters=num_clusters, save_folder=save_dir,
                     postfix='train', sorted_index=None,
                     train_bg=None, if_sort_clusters=False)

    # plot examples in the clusters
    # plot_cluster_examples(X_train, Y_train, ct,
    #                       pred_cluster,
    #                       num_clusters=num_clusters,
    #                       postfix="train",
    #                       save_folder=save_dir,
    #                       num_figs=3)

    test_pred_cluster, test_pred_prob, test_ct, test_sorted_index \
        = get_crosstab(test_spec, test_lbs, GMM,
                       num_clusters=num_clusters, save_folder=save_dir,
                       postfix='test', sorted_index=sorted_index,
                       train_bg=None, if_sort_clusters=False)

    # visualize with tsne
    for ii in range(num_clusters):
        for jj in range(ii, num_clusters):
            comb_data = np.vstack((test_pred_prob[:, ii], test_pred_prob[:, jj])).T
            X_embedded2 = TSNE(n_components=2).fit_transform(comb_data)
            fig = plt.figure(figsize=[12, 10])
            ax = fig.add_subplot(111)
            ind0 = np.where(test_lbs == 0)[0]
            ind1 = np.where(test_lbs == 1)[0]
            ax.scatter(X_embedded2[:, 0][ind0], X_embedded2[:, 1][ind0], color="royalblue", alpha=0.5, marker="o"),
            ax.scatter(X_embedded2[:, 0][ind1], X_embedded2[:, 1][ind1], color="violet", alpha=0.85, marker="^"),
            plt.savefig(save_dir + '/tsne_GMM_prob_2d_{}with{}.png'.format(ii, jj), format="png"),
            plt.close()

    count = dict(Counter(list(test_ids)))
    tumor_ratios = ct["Tumor"].values / ct["Healthy"].values
    norm_tumor_ratios =  tumor_ratios/ np.sum(tumor_ratios)
    sorted_t2h_ratio = np.argsort(tumor_ratios)
    health_ind = sorted_t2h_ratio[0]
    tumor_ind = sorted_t2h_ratio[-1]
    probs = test_pred_prob.dot(norm_tumor_ratios.reshape(-1, 1))
    # plot auc
    plot_auc_curve(test_lbs, probs, postfix="test", save_dir=save_dir)

    colors = ["royalblue", "m"]
    for id in count.keys():
        plt.figure(figsize=[12, 8])
        id_inds = np.where(test_ids == id)[0]
        vote_label = np.sum(test_lbs[id_inds]) * 1.0 / id_inds.size
        tumor_score = test_pred_prob[id_inds].dot(norm_tumor_ratios)
        vote_pred = np.sum(tumor_score) / id_inds.size

        label_of_id = 0 if vote_label < 0.5 else 1

        ax = plt.subplot(1, 1, 1)
        pred_hist = plt.hist(tumor_score, align='mid', bins=10, range=(0.0, 0.2), color=colors[label_of_id], label="tumor score")
        # ymin, ymax = ax.get_ylim()
        # ymax += 1
        # plt.vlines(0.5, ymin, ymax, colors='k', linestyles='--')
        # plt.text(0.25, ymax, str(np.int(np.sum(predictions[id_inds][:, 1] < 0.5))), fontsize=16)
        # plt.text(0.75, ymax, str(np.int(np.sum(predictions[id_inds][:, 1] >= 0.5))), fontsize=16)
        plt.legend()
        plt.ylabel("frequency")
        plt.xlabel("tumor score")
        plt.title(
            "True label {}, tumor score {:.3f} / (in total {} voxels for id {})".format(label_of_id, vote_pred, id_inds.size,
                                                                                 id))
        plt.tight_layout()
        plt.savefig(save_dir + '/prob_distri_of_id_{}.png'.format(id), format="png")
        plt.close()

    # Predict
    sample = np.vstack((test_spec[:, 171], test_spec[:, 62])).T
    prediction = GMM.predict_proba(sample[0:5])
    print(prediction)
    # Plot
    fig = plt.figure(figsize=(10, 10))
    ax0 = fig.add_subplot(111)
    ind0 = np.where(lbs == 0)[0]
    ind1 = np.where(lbs == 1)[0]
    ax0.scatter(features[:, 0][ind0], features[:, 1][ind0], c="royalblue")
    ax0.scatter(features[:, 0][ind1], features[:, 1][ind1], c="m")
    plt.show()
    print("ok")

    test_ind0 = np.where(test_lbs == 0)[0]
    test_ind1 = np.where(test_lbs == 1)[0]
    ax0.scatter(sample[:, 0][test_ind0], sample[:, 1][test_ind0], c='deepskyblue')
    ax0.scatter(sample[:, 0][test_ind1], sample[:, 1][test_ind1], c='orchid')
    for m, c in zip(means, covariances):
        multi_normal = multivariate_normal(mean=m, cov=c)
        ax0.contour(np.sort(features[:, 0]), np.sort(features[:, 1]), multi_normal.pdf(XY).reshape(len(features), len(features)), colors='black', alpha=0.3)
        ax0.scatter(m[0], m[1], c='grey', zorder=10, s=100)

    plt.show()
    print("ok")

