## @package plot
#  This package contains the code for producing plots about the training procedure.
import itertools
import matplotlib.pyplot as plt
import logging as log
import numpy as np
import matplotlib.pylab as pylab
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


def loss_plot(ax, data):
    train_loss = data["train_loss"]
    ax.plot(range(1, len(train_loss) + 1), train_loss, color='darkviolet', linestyle='--', marker='o', label='Train', alpha=0.8)
    test_loss = data["test_loss"]
    ax.plot(range(1, len(test_loss) + 1), test_loss, color='g', linestyle='--', marker='o', label='Test', alpha=0.8)
    ax.set_title("Loss")
    ax.legend()


def accuracy_plot(ax, data):
    train_accuracy = data["train_accuracy"]
    ax.plot(range(1, len(train_accuracy) + 1), train_accuracy, color='darkviolet', linestyle='--', marker='o', label='Train', alpha=0.8)
    test_accuracy = data["test_accuracy"]
    ax.plot(range(1, len(test_accuracy) + 1), test_accuracy, color='g', linestyle='--', marker='o',label='Test', alpha=0.8)
    ax.plot([np.argmax(test_accuracy) + 1, 0], [max(test_accuracy), max(test_accuracy)], '-.k')
    plt.text(np.argmax(test_accuracy) + 1, max(test_accuracy), "%.3f" % (max(test_accuracy)), horizontalalignment="center", color="k", size=18)
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
    loss_plot(ax, data)
    plt.tight_layout()
    f.savefig(args.output_path + '/accuracy_loss_step_{}.png'.format(data["current_step"]))
    plt.close()
    logger.info("Accuracy + Loss plot saved")


def all_figures(args, data):
    loss_figure(args, data)
    accuracy_figure(args, data)
    accuracy_loss_figure(args, data)
    plot_confusion_matrix(args, data, ifnormalize=True)
    plot_wrong_examples(args, data)


def plot_confusion_matrix(args, data, ifnormalize=True):
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
    f = plt.figure()
    # f.title("Wrongly classified examples")
    labels = np.argmax(data["test_wrong_labels"], axis=1)
    colors = ["darkorchid", "royalblue", "slateblue", "darkorange", "mediumseagreen", "plum"]
    num_classes = data["test_wrong_labels"].shape[-1]
    for i in range(num_classes):
        ax = f.add_subplot(num_classes, 1, i+1)
        ax.plot(data["test_wrong_features"][labels==i, :].T, color=colors[i])
        ax.plot(data["test_wrong_features"][labels==i, :][0], color=colors[i], label="label {}".format(i))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.legend(loc="best")
        f.subplots_adjust(hspace=0)
    f.savefig(args.output_path + '/{}-class_wrong_examples_{}.png'.format(num_classes, data["current_step"]))
    plt.close()
    logger.info("Mistakes plot saved")