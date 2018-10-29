## @package plot
#  This package contains the code for producing plots about the training procedure.
import matplotlib.pyplot as plt
import logging as log
import numpy as np


logger = log.getLogger("classifier")


def loss_plot(ax, data):
    train_loss = data["train_loss"]
    ax.plot(range(1, len(train_loss) + 1), train_loss, '--bo', label='Train')
    test_loss = data["test_loss"]
    ax.plot(range(1, len(test_loss) + 1), test_loss, '--ro', label='Test')
    ax.legend()


def accuracy_plot(ax, data):
    train_accuracy = data["train_accuracy"]
    ax.plot(range(1, len(train_accuracy) + 1), train_accuracy, '--bo', label='Train')
    test_accuracy = data["test_accuracy"]
    ax.plot(range(1, len(test_accuracy) + 1), test_accuracy, '--ro', label='Test')
    ax.plot([np.argmin(test_accuracy), len(test_accuracy)], [min(test_accuracy), min(test_accuracy)], '-.k')
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
    loss_figure(args, data)
    accuracy_loss_figure(args, data)
