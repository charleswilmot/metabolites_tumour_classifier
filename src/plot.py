## @package plot
#  This package contains the code for producing plots about the training procedure.
import itertools
import matplotlib.pyplot as plt
import logging as log
import numpy as np
import matplotlib.pylab as pylab
# import tsne
import ipdb
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

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
    plot_confusion_matrix(args, data, ifnormalize=False, training=training)
    plot_wrong_examples(args, data, training=training)
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