## @package plot
#  This package contains the code for producing plots about the training procedure.
import itertools
import matplotlib.pyplot as plt
import logging as log
import numpy as np
import matplotlib.pylab as pylab
from collections import Counter
# import tsne
import ipdb
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn import metrics

base = 22
params = {'legend.fontsize': base-8,
          'figure.figsize': (10, 8),
         'axes.labelsize': base-4,
         #'weight' : 'bold',
         'axes.titlesize':base,
         'xtick.labelsize':base-6,
         'ytick.labelsize':base-6}
pylab.rcParams.update(params)

logger = log.getLogger("classifier")


def loss_plot(ax, data, training=False):
    if training:
        train_loss = data["train_loss"]
        ax.plot(range(1, len(train_loss) + 1), train_loss, color='darkviolet', linestyle='--', marker='o', label='Train', alpha=0.8)
        test_loss = data["test_loss"]
        ax.plot(range(1, len(test_loss) + 1), test_loss, color='g', linestyle='--', marker='*', label='Validation', alpha=0.8)
    else:
        test_loss = [data["test_loss"]]
        ax.plot(range(1, len(test_loss) + 1), np.array(test_loss), color='g', linestyle='--', marker='*', label='Validation', alpha=0.8)
    ax.set_title("Loss")
    ax.legend(loc="best")


def accuracy_plot(ax, data, training=False):
    if training:
        train_accuracy = data["train_accuracy"]
        ax.plot(range(1, len(train_accuracy) + 1), train_accuracy, color='darkviolet', linestyle='--', marker='o', label='Train', alpha=0.8)
        test_accuracy = data["test_accuracy"]
        ax.plot(range(1, len(test_accuracy) + 1), test_accuracy, color='g', linestyle='--', marker='*',label='Validation', alpha=0.8)
        highest = max(test_accuracy)
    else:
        test_accuracy = data["test_accuracy"]
        ax.plot(range(1, test_accuracy.size + 1), test_accuracy, color='g', linestyle='--', marker='*', label='Validation',
                alpha=0.8)
        highest = test_accuracy
    ax.plot([np.argmax(test_accuracy) + 1, 0], [highest, highest], '-.k')
    plt.text(np.argmax(test_accuracy) + 1, highest, "%.4f" % (highest), horizontalalignment="center", color="k", size=18)
    ax.set_title("Accuracy")
    ax.legend(loc="lower right")


def loss_figure(args, data, training=False):
    f = plt.figure()
    ax = f.add_subplot(111)
    loss_plot(ax, data, training=training)
    f.savefig(args.output_path + '/loss_step_{}.png'.format(data["current_step"]))
    logger.info("Loss plot saved")


def accuracy_figure(args, data, training=False):
    f = plt.figure()
    ax = f.add_subplot(111)
    accuracy_plot(ax, data, training=training)
    f.savefig(args.output_path + '/accuracy_step_{}.png'.format(data["current_step"]))
    logger.info("Accuracy plot saved")


def plot_auc_curve(args, data, epoch=0):
    """
    Plot AUC curve
    :param args:
    :param data:
    :return:
    """
    f = plt.figure()
    fpr, tpr, _ = metrics.roc_curve(np.argmax(data["test_labels"], 1), data["test_pred"][:, 1])  # input the positive label's prob distribution
    auc = metrics.roc_auc_score(data["test_labels"], data["test_pred"])
    plt.plot(fpr, tpr, label="auc=" + str(auc))
    plt.legend(loc=4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    f.savefig(args.output_path + '/AUC_curve_step_{}.png'.format(epoch))
    plt.close()

def accuracy_loss_figure(args, data, training=False, epoch=0):
    f = plt.figure()
    ax = f.add_subplot(121)
    accuracy_plot(ax, data, training=training)
    ax = f.add_subplot(122)
    loss_plot(ax, data, training=training)
    plt.tight_layout()
    if training:
        f.savefig(args.output_path + '/accuracy_loss_step_{}-epoch-{}.png'.format(data["current_step"], epoch))
    else:
        f.savefig(args.output_path + '/accuracy_loss_step_{}.png'.format("VALID"))
    plt.close()
    logger.info("Accuracy + Loss plot saved")


def all_figures(args, data, training=False, epoch=0):
    # loss_figure(args, data, training=training)
    # accuracy_figure(args, data, training=training)
    accuracy_loss_figure(args, data, training=training, epoch=epoch)
    plot_confusion_matrix(args, data, ifnormalize=True, training=training)
    # plot_wrong_examples(args, data, training=training)
    plot_auc_curve(args, data, epoch=epoch)
    # plot_prob_distr_on_ids(data, args.output_path)
    # plot_tsne(args, data)
    # plot_hierarchy_cluster(args, data)


def plot_confusion_matrix(args, data, ifnormalize=False, training=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    :param confm: confusion matrix
    :param num_classes: int, the number of classes
    :param normalize: boolean, whether normalize to (0,1)
    :return:
    """
    if ifnormalize:
        cm = (data["test_confusion"] * 1.0 / data["test_confusion"].sum(axis=1)[:, np.newaxis])*1.0
        logger.info("Normalized confusion matrix")
    else:
        cm = data["test_confusion"]
        logger.info('Confusion matrix, without normalization')
    f = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
    tick_marks = np.arange(args.num_classes)
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)
    plt.colorbar()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] != 0:
            plt.text(j, i, np.int(cm[i, j]*100)/100.0, horizontalalignment="center", color="darkorange", size=20)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if training:
        f.savefig(args.output_path + '/confusion_matrix_step_{}.png'.format(data["current_step"]))
    else:
        f.savefig(args.output_path + '/confusion_matrix_step_{}.png'.format("VALID"))
    plt.close()
    logger.info("Confusion matrix saved")

def plot_wrong_examples(args, data, training=False):
    """
    Plot the wrongly classified examples
    :param args: contains hyperparams
    :param data: dict,
    :return:
    """
    f = plt.figure()
    labels = np.argmax(data["test_wrong_labels"], axis=1)
    colors = ["orchid", "deepskyblue", "plum", "darkturquoise", "m", "darkcyan"]
    num_classes = data["test_wrong_labels"].shape[-1]
    plt.title("Mistakes")
    for i in range(num_classes):
        ax = f.add_subplot(num_classes, 1, i+1)
        if len(data["test_wrong_features"][labels==i, :]) != 0:
            ax.plot(data["test_wrong_features"][labels==i, :].T, color=colors[i])
            ax.plot(data["test_wrong_features"][labels==i, :][0], color=colors[i], label="label {}".format(i))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.legend(loc="best")
        f.subplots_adjust(hspace=0)
    if training:
        f.savefig(args.output_path + '/{}-class_wrong_examples_{}.png'.format(num_classes, data["current_step"]))
    else:
        f.savefig(args.output_path + '/{}-class_wrong_examples_{}.png'.format(num_classes, "VALID"))
    plt.close()
    logger.info("Mistakes plot saved")

def plot_tsne(args, data):
    """
    Plot the wrongly classified examples
    :param args: contains hyperparams
    :param acti: dict,
    :return:
    """
    f = plt.figure()
    acti = data["test_activity"].astype(np.float64)
    labels = np.argmax(data["test_labels"], axis=1).astype(np.int)
    tsne_results = tsne.bh_sne(acti, d=2)
    #
    # colors = ["orchid", "deepskyblue", "plum", "darkturquoise", "m", "darkcyan"]
    # markers = np.random.choice(['o', '*', '^', 'D', 's', 'p'], args.num_classes)
    # target_names = ["label {}".format(i) for i in range(args.num_classes)]
    colors = ["orchid", "deepskyblue"]
    markers = ['o', '^']
    target_names = ["label {}".format(i) for i in [1, 3]] # np.arange(args.num_classes)

    ax = f.add_subplot(111)
    for color, marker, i, target_name in zip(colors, markers, [1, 3], target_names):
        ax.scatter(tsne_results[labels == i, 0], tsne_results[labels == i, 1], color=color, alpha=.8, linewidth=2, marker=marker, label=target_name)###lw=2,
    plt.setp(ax.get_xticklabels(), visible = False)
    plt.setp(ax.get_yticklabels(), visible = False)
    plt.legend(loc='best', shadow=False, scatterpoints=3)
    f.savefig(args.output_path + '/{}-tsne-on-activity-{}.png'.format(args.num_classes, data["current_step"]))
    plt.close()
    logger.info("TSNE saved")


def plot_hierarchy_cluster(args, data):
    """
    Plot hierarchy tree cluster to see subclusters
    :param args:
    :param data:
    :return:
    """
    f = plt.figure()
    inds = 0
    acti = data["test_activity"].astype(np.float64)
    labels = np.argmax(data["test_labels"], axis=1).astype(np.int)
    inds = np.append(inds, np.where(labels == 1)[0])
    inds = np.append(inds, np.where(labels == 3)[0])

    data_dist = pdist(acti[inds])
    data_link = linkage(data_dist, method='ward')
    dendrogram(data_link, labels=labels, leaf_font_size=8, leaf_rotation=90)  ##
    plt.xlabel("samples")
    plt.ylabel("distance")
    f.savefig(args.output_path + '/{}-hierarchy-on-activity-{}.png'.format(args.num_classes, data["current_step"]))
    plt.close()
    


def plot_class_activation_map(sess, class_activation_map, top_conv,
                                 images_test, labels_test, pred_labels, global_step,
                                 num_images, args):
    """TODO, the labels are all stacked not like test_conv is only the first batch
    Plot the class activation
    :param sess:
    :param class_activation_map:
    :param top_conv:
    :param images_test:
    :param labels_test:
    :param pred_labels:
    :param global_step:
    :param num_images:
    :param args:
    :return:
    """
    labels_int = np.argmax(labels_test, axis=1)
    classmap_answer = sess.run(class_activation_map)

    classmap_high = list(map(lambda x: (np.where(np.squeeze(x) > np.percentile(np.squeeze(x), 80))[0]), classmap_answer))
    classmap_low = list(map(lambda x: (np.where(np.squeeze(x) < np.percentile(np.squeeze(x), 18))[0]), classmap_answer))
    plots_per_fig = 5
    counts = np.arange(plots_per_fig)

    rand_inds = np.random.choice(np.arange(len(classmap_answer)), num_images)
    classmap_high = np.array(classmap_high)[rand_inds]
    classmap_low = np.array(classmap_low)[rand_inds]
    images_plot = images_test[rand_inds]
    labels_plot = labels_int[rand_inds]
    pred_plot = pred_labels[rand_inds]
    for j in range(num_images // plots_per_fig):
        fig = plt.figure()
        for count, vis_h, vis_l, ori in zip(counts, classmap_high[j*plots_per_fig : (j+1)*plots_per_fig], classmap_low[j*plots_per_fig : (j+1)*plots_per_fig], images_plot[j*plots_per_fig : (j+1)*plots_per_fig]):

            plt.subplot(plots_per_fig, 1, count+1)
            plt.plot(np.arange(ori.size), ori, 'darkorchid', label='original', linewidth=0.8)
            plt.plot(np.array(vis_h), np.repeat(ori.max(), vis_h.size), '.', color='deepskyblue', label='attention')

            att_indices = collect_and_plot_atten(vis_h, 1)  # collect the attention part indices

            plt.xlim([0, ori.size])
            plt.xlabel("label: {} - pred: {}".format(np.str(labels_plot[j*plots_per_fig+count]), np.str(pred_plot[j*plots_per_fig+count])))
            if count == 0:
                plt.legend(bbox_to_anchor=(0.15, 1.05, 0.7, 0.1), loc="lower left", mode="expand", borderaxespad=0, ncol=2, scatterpoints=3)
            if count == plots_per_fig - 1:
                plt.xlabel("time / s, label: {} - pred: {}".format(np.str(labels_plot[j*plots_per_fig+count]), np.str(pred_plot[j*plots_per_fig+count])))

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig(args.output_path + '/class_activity_map-step-test{}-count{}.pdf'.format(global_step, j), format='pdf')
        plt.close()#


def collect_and_plot_atten(indices, scale):
    """
    PLot vertical lines on the attention area
    :param indices:
    :return:
    """
    xcoords = list(np.where((indices[1:] - indices[0:-1]) > 1)[0] + 1)
    for ind in np.arange(len(xcoords)) * 2:
        xcoords.insert(ind, xcoords[ind] - 1)

    start_end_inds = [0] + xcoords + [len(indices) - 1]
    for xc in ([0] + xcoords + [len(indices) - 1]):  # [0] + xcoords + [len(vis_h)-1]
        plt.axvline(x=indices[xc] / np.float(scale), color='b', linewidth=0.35)

    return start_end_inds


def plot_prob_distr_on_ids(test_data, output_dir, num_classes=2):
    """
    Get the prob distribution histogram of samples of each id
    :param test_data:
    :return:
    """
    predictions = test_data["test_pred"]
    pred_int = np.argmax(test_data["test_pred"], axis=1)
    ids = test_data["test_ids"]
    labels_int = np.argmax(test_data["test_labels"], axis=1)
    count = dict(Counter(list(ids)))  # c
    
    for id in count.keys():
        if id == 443.0:
            print("421")
        plt.figure()
        id_inds = np.where(ids == id)[0]
        vote_label = np.sum(labels_int[id_inds]) * 1.0 / id_inds.size
        vote_pred = np.sum(predictions[id_inds][:, 1]) / id_inds.size
        # vote_pred = np.sum(pred_int[id_inds]) * 1.0 / id_inds.size
        
        label_of_id = 0 if vote_label < 0.5 else 1
        pred_of_id = 0 if vote_pred < 0.5 else 1
        
        ax = plt.subplot(1, 1, 1)
        pred_hist = plt.hist(np.array(predictions[id_inds][:, 1]), align='mid', bins=10, range=(0.0, 1.0), color='royalblue', label="predicted")
        ymin, ymax = ax.get_ylim()
        plt.vlines(0.5, ymin, ymax, colors='k', linestyles='--')
        plt.text(0.25, ymax, str(np.int(np.sum(predictions[id_inds][:, 1] < 0.5))), fontsize=base-4)
        plt.text(0.75, ymax, str(np.int(np.sum(predictions[id_inds][:, 1] >= 0.5))), fontsize=base-4)
        plt.legend()
        plt.ylabel("frequency")
        plt.xlabel("probability of classified as class 1")
        plt.title("True label {} - pred as {} / (in total {} voxels for id {})".format(label_of_id, pred_of_id, id_inds.size, id))
        plt.tight_layout()
        plt.savefig(output_dir + '/prob_distri_of_id_{}.png'.format(id), format="png")
        plt.close()