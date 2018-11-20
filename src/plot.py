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
    ax.plot(range(1, len(train_loss) + 1), train_loss, '--bo', label='Train')
    test_loss = data["test_loss"]
    ax.plot(range(1, len(test_loss) + 1), test_loss, '--ro', label='Test')
    ax.set_title("Loss")
    ax.legend()


def accuracy_plot(ax, data):
    train_accuracy = data["train_accuracy"]
    ax.plot(range(1, len(train_accuracy) + 1), train_accuracy, '--bo', label='Train')
    test_accuracy = data["test_accuracy"]
    ax.plot(range(1, len(test_accuracy) + 1), test_accuracy, '--ro', label='Test')
    ax.plot([np.argmax(test_accuracy) + 1, 0], [max(test_accuracy), max(test_accuracy)], '-.k')
    ax.set_title("Accuracy")
    ax.legend()


def loss_figure(args, data):
    f = plt.figure()
    ax = f.add_subplot(111)
    loss_plot(ax, data)
    f.savefig(args.output_path + '/loss.png')
    logger.info("Loss plot saved")


def accuracy_figure(args, data):
    f = plt.figure()
    ax = f.add_subplot(111)
    accuracy_plot(ax, data)
    f.savefig(args.output_path + '/accuracy.png')
    logger.info("Accuracy plot saved")


def accuracy_loss_figure(args, data):
    f = plt.figure()
    ax = f.add_subplot(121)
    accuracy_plot(ax, data)
    ax = f.add_subplot(122)
    loss_plot(ax, data)
    f.savefig(args.output_path + '/accuracy_loss.png')
    logger.info("Accuracy + Loss plot saved")


def all_figures(args, data):
    loss_figure(args, data)
    accuracy_figure(args, data)
    accuracy_loss_figure(args, data)


def plot_confusion_matrix(confm, num_classes, save_dir, ifnormalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    :param confm: confusion matrix
    :param num_classes: int, the number of classes
    :param normalize: boolean, whether normalize to (0,1)
    :return:
    """
    if ifnormalize:
        cm = (confm * 1.0 / confm.sum(axis=1)[:, np.newaxis])*1.0
        # cm = cm.astype('float16') / cm.sum(axis=1)[:, np.newaxis]
        logger.info("Normalized confusion matrix")
    else:
        cm = confm
        logger.info('Confusion matrix, without normalization')
    f = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)
    plt.colorbar()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] != 0:
            plt.text(j, i, np.int(cm[i, j]*100)/100.0, horizontalalignment="center")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    f.savefig(save_dir + '/confusion_matrix.png')
    f.close()
    logger.info("Confusion matrix saved")