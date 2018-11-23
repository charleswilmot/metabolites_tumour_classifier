## @package plot
#  This package contains the code for producing plots about the training procedure.
import itertools
import matplotlib.pyplot as plt
import logging as log
import numpy as np
import matplotlib.pylab as pylab
base = 22
params = {'legend.fontsize': base-4,
          'figure.figsize': (10, 8),
         'axes.labelsize': base-4,
         #'weight' : 'bold',
         'axes.titlesize':base,
         'xtick.labelsize':base-6,
         'ytick.labelsize':base-6}
pylab.rcParams.update(params)

logger = log.getLogger("classifier")


def loss_plot(ax, data):
    train_loss = data["train_loss"]
    ax.plot(range(1, len(train_loss) + 1), train_loss, color='darkviolet', linestyle='--', marker='o', label='Train')
    test_loss = data["test_loss"]
    ax.plot(range(1, len(test_loss) + 1), test_loss, color='g', linestyle='--', marker='o', label='Test')
    ax.set_title("Loss")
    ax.legend()


def accuracy_plot(ax, data):
    train_accuracy = data["train_accuracy"]
    ax.plot(range(1, len(train_accuracy) + 1), train_accuracy, color='darkviolet', linestyle='--', marker='o', label='Train')
    test_accuracy = data["test_accuracy"]
    ax.plot(range(1, len(test_accuracy) + 1), test_accuracy, color='g', linestyle='--', marker='o',label='Test')
    ax.plot([np.argmax(test_accuracy) + 1, 0], [max(test_accuracy), max(test_accuracy)], '-.k')
    ax.set_title("Accuracy")
    ax.legend()


def loss_figure(args, data):
    f = plt.figure()
    ax = f.add_subplot(111)
    loss_plot(ax, data)
    f.savefig(args.output_path + '/loss_step_{}.png'.format(data["current_step"]))
    plt.close()
    logger.info("Loss plot saved")


def accuracy_figure(args, data):
    f = plt.figure()
    ax = f.add_subplot(111)
    accuracy_plot(ax, data)
    f.savefig(args.output_path + '/accuracy_step_{}.png'.format(data["current_step"]))
    plt.close()
    logger.info("Accuracy plot saved")


def accuracy_loss_figure(args, data):
    f = plt.figure()
    ax = f.add_subplot(121)
    accuracy_plot(ax, data)
    ax = f.add_subplot(122)
    plt.tight_layout()
    loss_plot(ax, data)
    f.savefig(args.output_path + '/accuracy_loss_step_{}.png'.format(data["current_step"]))
    plt.close()
    logger.info("Accuracy + Loss plot saved")


def all_figures(args, data):
    loss_figure(args, data)
    accuracy_figure(args, data)
    accuracy_loss_figure(args, data)
    plot_confusion_matrix(args, data, ifnormalize=False)
    plot_wrong_examples(args, data)


def plot_confusion_matrix(args, data, ifnormalize=False):
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
    f.savefig(args.output_path + '/confusion_matrix_step_{}.png'.format(data["current_step"]))
    plt.close()
    logger.info("Confusion matrix saved")

def plot_wrong_examples(args, data):
    """
    Plot the wrongly classified examples
    :param args:
    :param data:
    :return:
    """
    colors = ["darkorchid", "royalblue"]
    f = plt.figure()
    ax1 = f.add_subplot(211)
    ax1.plot(data["test_wrong_features"][data["test_wrong_labels"]==0, :].T, color=colors[0])
    ax1.plot(data["test_wrong_features"][data["test_wrong_labels"]==0, :][0], color=colors[0], label="label 0")
    ax2 = f.add_subplot(212)
    ax2.plot(data["test_wrong_features"][data["test_wrong_labels"]==1, :].T, color=colors[1])
    ax2.plot(data["test_wrong_features"][data["test_wrong_labels"]==1, :][0], color=colors[1], label="label 1")
    plt.legend(loc="best")
    f.savefig(args.output_path + '/wrong_examples_{}.png'.format(data["current_step"]))
    plt.close()
    logger.info("Accuracy plot saved")