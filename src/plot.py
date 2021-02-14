## @package plot
#  This package contains the code for producing plots about the training procedure.
import itertools
import matplotlib.pyplot as plt
import logging as log
import numpy as np
from collections import Counter
from sklearn.manifold import TSNE
import ipdb
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn import metrics
import tensorflow as tf
import os
import matplotlib.pylab as pylab
base = 22
args = {'legend.fontsize': base - 8,
          'figure.figsize': (13, 8.6),
         'axes.labelsize': base-4,
        #'weight' : 'bold',
         'axes.titlesize':base,
         'xtick.labelsize':base-8,
         'ytick.labelsize':base-8}
pylab.rcParams.update(args)

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
        ax.plot(range(1, len(train_accuracy) + 1), train_accuracy, color='darkviolet', linestyle='--', marker='o', label='Train', alpha=0.7)
        test_accuracy = data["test_accuracy"]
        ax.plot(range(1, len(test_accuracy) + 1), test_accuracy, color='g', linestyle='--', marker='*',label='Validation', alpha=0.7)
        highest = max(test_accuracy)
    else:
        test_accuracy = data["test_accuracy"]
        ax.plot(range(1, test_accuracy.size + 1), test_accuracy, color='g', linestyle='--', marker='*', label='Validation',
                alpha=0.7)
        highest = test_accuracy
    ax.plot([np.argmax(test_accuracy) + 1, 0], [highest, highest], '-.k')
    plt.text(np.argmax(test_accuracy) + 1, highest, "%.4f" % (highest), horizontalalignment="center", color="k", size=18)
    ax.set_title("Accuracy")
    ax.legend(loc="lower right")
    return highest


def loss_figure(args, data, training=False):
    f = plt.figure()
    ax = f.add_subplot(111)
    loss_plot(ax, data, training=training)
    f.savefig(args.output_path + '/loss_step_{}.png'.format(data["current_step"]))
    logger.info("Loss plot saved")


def accuracy_figure(args, data, training=False, epoch=0):
    f = plt.figure()
    ax = f.add_subplot(111)
    if args.num_classes == 2:
        auc = metrics.roc_auc_score(data["test_labels"], data["test_logits"])
    else:
        auc = 0.500
    max_acc = accuracy_plot(ax, data, training=training)

    f.savefig(args.output_path + '/accuracy_step_{:.1f}_acc_{:.4f}_auc_{:.3f}_{}-{}.png'.format(epoch, max_acc, auc, args.data_source, args.train_or_test))
    plt.close()
    logger.info("Accuracy plot saved")


def plot_auc_curve(args, data, epoch=0):
    """
    Plot AUC curve
    :param args:
    :param data:
    :return:
    """
    f = plt.figure()
    fpr, tpr, _ = metrics.roc_curve(np.argmax(data["test_labels"], 1), data["test_logits"][:, 1])  # input the positive label's prob distribution
    auc = metrics.roc_auc_score(data["test_labels"], data["test_logits"])
    plt.plot(fpr, tpr, label="auc={0:.4f}".format(auc))
    plt.legend(loc=4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    f.savefig(args.output_path + '/AUCs/AUC_curve_step_{:.2f}-auc_{:.4}-{}-{}.png'.format(epoch, auc, args.data_source, args.train_or_test))
    np.savetxt(args.output_path + '/AUCs/AUC_curve_step_{:.2f}-auc_{:.4}-{}-{}.csv'.format(epoch, auc, args.data_source, args.train_or_test),
               np.hstack((np.argmax(data["test_labels"], 1).reshape(-1,1), data["test_logits"][:, 1].reshape(-1,1))), fmt="%.8f", delimiter=',', header="labels,pred[:,1]")
    plt.close()
    return auc


def accuracy_loss_figure(args, data, training=False, epoch=0):
    f = plt.figure()
    ax = f.add_subplot(121)
    max_acc = accuracy_plot(ax, data, training=training)
    ax = f.add_subplot(122)
    loss_plot(ax, data, training=training)
    plt.tight_layout()
    if training:
        f.savefig(args.output_path + '/accuracy_loss_step_{}-epoch-{:.2f}_acc_{:.4}.png'.format(data["current_step"], epoch, max_acc))
    else:
        f.savefig(args.output_path + '/accuracy_loss-step_{}-acc_{:.4f}.png'.format("VALID", max_acc))
    plt.close()
    logger.info("Accuracy + Loss plot saved")


def all_figures(sess, args, data, training=False, epoch=0):
    # print("labels: ", np.argmax(data["test_labels"], axis=1))
    accuracy_figure(args, data, training=training, epoch=epoch)
    # accuracy_loss_figure(args, data, training=training, epoch=epoch)
    # plot_confusion_matrix(args, data, ifnormalize=True, training=training)
    if args.num_classes == 2:
        auc = plot_auc_curve(args, data, epoch=epoch)
        print("accuracy: ", data["test_accuracy"], "auc: ", auc)
    else:
        print("accuracy: ", data["test_accuracy"])
        auc = np.Inf
    # plot_wrong_examples(args, data, training=training)

    if not training:
        if 'CAM' in args.model_name:
            class_maps, rand_inds = get_class_map(data["test_labels"], data["test_conv"], data["test_gap_w"], args.height, 1, number2use=1000)

            plot_class_activation_map(sess, class_maps, data["test_features"][rand_inds], data["test_labels"][rand_inds], data["test_logits"][rand_inds], epoch, args, auc=auc)
        # plot_wrong_examples(args, data, training=training)
        plot_prob_distr_on_ids(data, args.output_path)

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
        f.savefig(args.output_path + '/confusion_matrix_step_{}_{}.png'.format(data["current_step"], args.data_source))
    else:
        f.savefig(args.output_path + '/confusion_matrix_step_{}_{}.png'.format("VALID", args.data_source))
    plt.close()
    logger.info("Confusion matrix saved")


def plot_wrong_examples(args, data, training=False):
    """
    Plot the wrongly classified examples
    :param args: contains hyperparams
    :param data: dict,
    :return:
    """
    labels = data["test_wrong_labels"]
    colors = ["orchid", "deepskyblue", "plum", "darkturquoise", "m", "darkcyan"]
    num_classes = args.num_classes
    f, axs = plt.subplots(num_classes, 1, 'col')
    plt.suptitle("Mistakes", x=0.5, y=0.925, fontsize=base)
    for i in range(num_classes):
        if len(data["test_wrong_features"][labels == i, :]) != 0:
            axs[i].plot(data["test_wrong_features"][labels == i, :].T, color=colors[i])
            axs[i].plot(data["test_wrong_features"][labels == i, :][0], color=colors[i], label="True label {}".format(i))  # for labeling
        if i < num_classes - 1:
            plt.setp(axs[i].get_xticklabels(), visible=False)
        axs[i].legend(loc="best")
    f.subplots_adjust(hspace=0)
    if training:
        f.savefig(args.output_path + '/wrong_examples/{}-class_wrong_examples_{}.png'.format(num_classes, data["current_step"]))
    else:
        f.savefig(args.output_path + '/wrong_examples/{}-class_wrong_examples_{}.png'.format(num_classes, "VALID"))

    plt.close()
    logger.info("Mistakes plot saved")


def plot_tsne_activity(args, data):
    """
    Plot the wrongly classified examples
    :param args: contains hyperparams
    :param acti: dict,
    :return:
    #
    # colors = ["orchid", "deepskyblue", "plum", "darkturquoise", "m", "darkcyan"]
    # markers = np.random.choice(['o', '*', '^', 'D', 's', 'p'], args.num_classes)
    # target_names = ["label {}".format(i) for i in range(args.num_classes)]
    """
    f = plt.figure()
    acti = data["test_activity"].astype(np.float64)
    labels = np.argmax(data["test_labels"], axis=1).astype(np.int)
    tsne_results = tsne.bh_sne(acti, d=2)
    colors = ["orchid", "deepskyblue"]
    markers = ['o', '^']
    target_names = ["label {}".format(i) for i in [1, 3]] #

    ax = f.add_subplot(111)
    for color, marker, i, target_name in zip(colors, markers, [1, 3], target_names):
        ax.scatter(tsne_results[labels == i, 0], tsne_results[labels == i, 1], color=color, alpha=.8, linewidth=2, marker=marker, label=target_name)###lw=2,
    plt.setp(ax.get_xticklabels(), visible = False)
    plt.setp(ax.get_yticklabels(), visible = False)
    plt.legend(loc='best', shadow=False, scatterpoints=3)
    f.savefig(args.output_path + '/{}-tsne-on-activity-{}.png'.format(args.num_classes, data["current_step"]))
    plt.close()
    logger.info("TSNE saved")


def plot_tsne(data, labels, epoch=0, n_components=3, random_pick=500, save_dir="../results", postfix="latent"):
    """
	Plot the wrongly classified examples
	:param args: contains hyperparams
	:param acti: dict,
	:return:
	#
	# colors = ["orchid", "deepskyblue", "plum", "darkturquoise", "m", "darkcyan"]
	# markers = np.random.choice(['o', '*', '^', 'D', 's', 'p'], args.num_classes)
	# target_names = ["label {}".format(i) for i in range(args.num_classes)]
	"""
    tsne = TSNE(n_components=n_components, random_state=0)
    inds = []
    for i in range(2):
        rand_inds = list(np.random.choice(np.where(labels == i)[0], min(random_pick, len(np.where(labels == i)[0]))))
        inds += rand_inds
    tsne_results = tsne.fit_transform(data[inds])
    labels = labels[inds].astype(np.int)
    
    colors = ["deepskyblue", "orchid"]
    markers = ['o', '^']
    target_names = ["label {}".format(i) for i in [0, 1]]  #
    center_color = ["b", "purple"]
    
    if n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import animation
        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
        for i, c, label in zip([0, 1], colors, target_names):
            ax.scatter(tsne_results[np.where(labels == i)[0], 0], tsne_results[np.where(labels == i)[0], 1],
                       tsne_results[np.where(labels == i)[0], 2], s=30, c=c, label=label, alpha=0.5)
            meanx, meany, meanz = np.mean(tsne_results[np.where(labels == i)[0], 0]), np.mean(
                tsne_results[np.where(labels == i)[0], 1]), np.mean(tsne_results[np.where(labels == i)[0], 2])
            ax.scatter(meanx, meany, meanz, s=50, c=center_color[i])
        
        for i, plt_label in zip(np.arange(n_components), [ax.set_xlabel, ax.set_ylabel, ax.set_zlabel]):
            label0 = tsne_results[np.where(labels == 0)[0], i]
            label1 = tsne_results[np.where(labels == 1)[0], i]
            stat, p = ks_2samp(label0, label1)
            plt_label("p-value: {:.2e}".format(p))
        
        fig.legend()
        fig.savefig(save_dir + '/3d-tsne-on-{}-{}.png'.format(postfix, epoch))
        plt.close()
    elif n_components == 2:
        f = plt.figure()
        ax = f.add_subplot(111)
        for color, marker, i, target_name in zip(colors, markers, [0, 1], target_names):
            ax.scatter(tsne_results[labels == i, 0], tsne_results[labels == i, 1], color=color, alpha=.8, linewidth=2,
                       marker=marker, label=target_name)  ###lw=2,
            meanx, meany = np.mean(tsne_results[np.where(labels == i)[0], 0]), np.mean(
                tsne_results[np.where(labels == i)[0], 1])
            ax.scatter(meanx, meany, s=50, c=center_color[i])
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        
        for i, plt_label in zip([0, 1], [plt.xlabel, plt.ylabel]):
            label0 = tsne_results[np.where(labels == 0)[0], i]
            label1 = tsne_results[np.where(labels == 1)[0], i]
            stat, p = ks_2samp(label0, label1)
            plt_label("p-value: {:.2e}".format(p))
        
        plt.legend(loc='best', shadow=False, scatterpoints=3)
        f.savefig(save_dir + '/2d-tsne-on-{}-{}.png'.format(postfix, epoch))
        plt.close()


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


def get_class_map(labels, conv_out, weights, seq_len, seq_width, number2use=200):
    """
    Get the map for a specific sample with class. Some samples can be assigned to different classes.
    :param number2use: int, [batch_size,], the int labels in one batch
    :param labels_int: 1d array, [batch_size,], the int labels in one batch
    :param conv_out: list, list of arry that contains the output from the conv_layer from the network
    :param weights: weights of the last GAP feature map
    :param im_width: The length of the input sample
    :return: class map
    """
    # conv_out = np.expand_dims(conv_out, 2)
    channels = conv_out.shape[-1]
    num_samples = min(conv_out.shape[0], number2use)
    classmaps = []
    labels_int = np.argmax(labels, axis=1)
    conv_resized = tf.compat.v1.image.resize_nearest_neighbor(conv_out, [seq_len, seq_width])
    rand_inds = np.random.choice(conv_out.shape[0], min(num_samples, conv_out.shape[0]))
    for ind, label in enumerate(labels_int[rand_inds]):
        label_w = tf.gather(tf.transpose(weights), label)
        label_w = tf.reshape(label_w, [-1, channels, 1])
        resized = tf.reshape(conv_resized[ind], [-1, seq_len * seq_width, channels])
        classmap = tf.matmul(resized, label_w)
        classmaps.append(tf.reshape(classmap, [-1, seq_len, seq_width]))

    return classmaps, rand_inds


def plot_class_activation_map(sess, class_activation_map,
                              samples_test, labels_test,
                              predictions, global_step,
                              args, auc=0.5):
    """TODO, the labels are all stacked not like test_conv is only the first batch
    Plot the class activation
    :param sess:
    :param class_activation_map:
    :param top_conv:
    :param samples_test:
    :param labels_test:
    :param predictions: softmax output
    :param global_step:
    :param args:
    :param rand_inds:
    :return:
    """
    labels_int = np.argmax(labels_test, axis=1)
    classmap_answer = sess.run(class_activation_map)

    # classmap_high = list(map(lambda x: (np.where(np.squeeze(x) > np.percentile(np.squeeze(x), 90))[0]), classmap_answer))
    classmap_high = list(map(lambda x: ((np.squeeze(x) - np.min(np.squeeze(x))) / (np.max(np.squeeze(x) - np.min(np.squeeze(x))))), classmap_answer))

    # PLot samples from different classes seperately
    for class_id in range(args.num_classes):
        plot_sep_class_maps(labels_int, classmap_high, samples_test, predictions, save_dir=args.output_path, row=8, box_position=(0.15, 1.05, 2, 0.1), class_id=class_id, global_step=global_step, postfix=args.data_source, auc=auc)
        # plot_class_maps_in1(labels_int, classmap_high, samples_test, pred_labels, save_dir=args.output_path, box_position=(0.15, 1.05, 0.7, 0.1), class_id=class_id, global_step=global_step)

def plot_sep_class_maps(labels_int, classmap_high, samples_test, predictions, save_dir='./', row=None, box_position=(0.15, 1.05, 2, 0.1), auc=0.5, class_id=0, global_step=0, postfix="data1"):
    """
    PLot class maps grouped by class ids
    :param labels_int:
    :param classmap_high:
    :param samples_test:
    :param predictions: softmax output
    :param save_dir:
    :param row:
    :param box_position:
    :param class_id:
    :param global_step:
    :param postfix:
    :return:
    """
    inds = np.where(labels_int == class_id)[0]
    class_maps_plot = np.array(classmap_high)[inds]
    samples_plot = samples_test[inds]
    labels_plot = labels_int[inds]

    pred_plot = np.argmax(predictions, axis=1)[inds]
    if not row:
        row = np.int(np.sqrt(len(inds)))
    else:
        row = row
    col = min(row, 5)
    counts = (np.arange(row*col).reshape(-1, col)[np.arange(0, row, 2)]).reshape(-1)# put attention beneath the signal
    fig, axs = plt.subplots(row, col, sharex=True)
    plt.suptitle("Individual samples from class {} with attention".format(class_id))
    for j, vis, ori, label, pred in zip(counts, class_maps_plot, samples_plot, labels_plot, pred_plot):
        att_c = 'deepskyblue' if label == pred else 'r'
        axs[j // col, np.mod(j, col)].plot(np.arange(ori.size), ori, 'darkorchid', label='original', linewidth=0.8)
        axs[j // col + 1, np.mod(j, col)].plot(np.arange(vis.size), vis, '--', color=att_c, label="attention")
        axs[0, 0].legend(bbox_to_anchor=box_position, loc="lower left", mode="expand", borderaxespad=0, ncol=1)  # [x, y, width, height]
        # axs[1, col-1].legend(bbox_to_anchor=box_position, loc="lower left", mode="expand", borderaxespad=0, ncol=1)  # [x, y, width, height]
        if np.mod(j, col) > 0:
            plt.setp(axs[j // col + 1, np.mod(j, col)].get_yticklabels(), visible=False)
            plt.setp(axs[j // col, np.mod(j, col)].get_yticklabels(), visible=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.01, wspace=0.01)
    plt.savefig(save_dir + '/CAMs/class_activity_map-step-test{:.1f}-auc-{:.3f}-class{}.png'.format(global_step, auc, class_id), format='png')
    plt.savefig(save_dir + '/CAMs/class_activity_map-step-test{:.1f}-auc-{:.3f}-class{}.pdf'.format(global_step, auc, class_id), format='pdf')
    plt.close()
    
    ## Plot the mean attention
    right_inds = np.where(labels_plot == pred_plot)[0]
    plt.figure()
    mean = np.mean(np.array(class_maps_plot)[right_inds], axis=0)
    std = np.std(np.array(class_maps_plot)[right_inds], axis=0)
    plt.errorbar(np.arange(288), mean, yerr=std, fmt='--o', ecolor='deepskyblue', label='mean attention')
    plt.plot(samples_plot[0], 'darkorchid', label='original', linewidth=0.8)
    plt.legend(loc="best")
    plt.title("Mean attention for class {}".format(class_id))
    plt.savefig(save_dir + '/CAMs/mean/mean_class_activity_map-step-{:.1f}-auc-{:.3f}-class{}-{}.png'.format(global_step, auc, class_id, postfix), format='png')
    plt.close()


def plot_class_maps_in1(labels_int, classmap_high, samples_test, pred_labels, save_dir='./', box_position=(0.15, 1.05, 0.8, 0.1), class_id=0, global_step=0):
    inds = np.where(labels_int == class_id)[0]
    class_maps_plot = np.array(classmap_high)[inds]
    samples_plot = samples_test[inds]
    labels_plot = labels_int[inds]
    pred_plot = pred_labels[inds]
    row = np.int(np.sqrt(len(inds)))
    col = min(row, 5)
    counts = np.arange(row * col)
    fig, ax = plt.subplots(1,1,sharex=True)
    first_correct = np.where(labels_plot == pred_plot)[0][0]
    first_wrong = np.where(labels_plot != pred_plot)[0][0]

    for j, vis, ori, label, pred in zip(counts, class_maps_plot, samples_plot, labels_plot, pred_plot):
        my_ori_label = None
        my_att_label = None
        att_c = 'deepskyblue' if label == pred else 'r'
        if j == first_wrong or j == first_correct:
            if j == 0:
                my_att_label = "correct attetion" if label == pred else "wrong attention"
                my_ori_label = 'original'
            else:
                my_att_label = "correct attetion" if label == pred else "wrong attention"

        ax.plot(np.arange(ori.size), ori, 'darkorchid', label=my_ori_label, linewidth=0.8)
        ax.plot(np.arange(vis.size), vis, '-.', color=att_c, label=my_att_label)
    plt.title("Class {} with attention".format(class_id))
    ax.legend(bbox_to_anchor=box_position, loc="lower left", mode="expand", borderaxespad=0, ncol=3, numpoints=3)  # [x, y, width, height]
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(save_dir + '/CAMs/class_activity_map-step-test{}-class{}_{}-all-in-one.png'.format(global_step, class_id, box_position),
                format='png')
    plt.close()

def plot_random_class_maps(labels_int, classmap_answer, classmap_high, samples_test, predictions, args, global_step=0, num_images=20):
    """
    Plot class maps randomly
    :param labels_int:
    :param classmap_answer:
    :param classmap_high:
    :param samples_test:
    :param predictions:
    :param args:
    :param global_step:
    :param num_images:
    :return:
    """
    plots_per_fig = 10
    counts = np.arange(plots_per_fig)
    rand_inds = np.random.choice(np.arange(len(classmap_answer)), num_images)
    classmap_high = np.array(classmap_high)[rand_inds]
    samples_plot = samples_test[rand_inds]
    labels_plot = labels_int[rand_inds]
    pred_plot = np.argmax(predictions, axis=1)[inds]
    auc = metrics.roc_auc_score(labels_int, predictions[:, 1])
    row = plots_per_fig//2
    col = 2
    fig, axs = plt.subplots(row, col, sharex=True)
    for j in range(num_images // plots_per_fig):
        for count, vis, ori in zip(counts, classmap_high[j*plots_per_fig : (j+1)*plots_per_fig], samples_plot[j*plots_per_fig : (j+1)*plots_per_fig]):
            label_c = 'k' if labels_plot[j*plots_per_fig+count] == pred_plot[j*plots_per_fig+count] else 'r'
            axs[count // col, np.mod(count, col)].plot(np.arange(ori.size), ori, 'darkorchid', label='original', linewidth=0.8)
            axs[count // col, np.mod(count, col)].plot(np.array(vis), np.repeat(ori.max(), vis.size), '.', color='deepskyblue', label='attention')
            axs[0, 0].legend(bbox_to_anchor=(0.15, 1.05, 0.7, 0.1), loc="lower left", mode="expand", borderaxespad=0, ncol=2, scatterpoints=3)

            att_indices = collect_and_plot_atten(vis, 1)  # collect the attention part indices
            plt.xlim([0, ori.size])
            plt.xlabel("label: {} - pred: {}".format(np.str(labels_plot[j*plots_per_fig+count]), np.str(pred_plot[j*plots_per_fig+count])), color=label_c)

            if count == plots_per_fig - 1:
                plt.xlabel("time / s, label: {} - pred: {}".format(np.str(labels_plot[j*plots_per_fig+count]), np.str(pred_plot[j*plots_per_fig+count])))

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig(args.output_path + '/CAMs/class_activity_map-step-test{:1f}-auc-{:.3f}-count{}.pdf'.format(global_step, auc, j), format='pdf')
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
    for xc in ([0] + xcoords + [len(indices) - 1]):
        plt.axvline(x=indices[xc] / np.float(scale), color='b', linewidth=0.35)

    return start_end_inds


def plot_prob_distr_on_ids(test_data, output_dir, num_classes=2):
    """
    Get the prob distribution histogram of samples of each id
    :param test_data:
    :return:
    """
    predictions = test_data["test_logits"]
    pred_int = np.argmax(test_data["test_logits"], axis=1)
    ids = test_data["test_ids"]
    labels_int = np.argmax(test_data["test_labels"], axis=1)
    count = dict(Counter(list(ids)))  # c
    
    rignt_patients = np.empty((0))
    wrong_patients = np.empty((0))

    # Plot prob. histogram all the voxels from each class
    distr_color = ["m", "lawngreen"]
    plt.figure()
    for class_id in range(num_classes):
        # plt.subplot(2, 1, class_id+1)
        inds = np.where(labels_int == class_id)[0]
        plt.hist(np.array(predictions[inds][:, 1]), align='mid', bins=25, range=(0.0, 1.0), color=distr_color[class_id], label="samples from class {} patients ".format(class_id), alpha=0.55, density=True)
    plt.xlabel("probability for class 1")
    plt.ylabel("Normalized frequency")
    plt.legend(loc="best")
    # plt.savefig(output_dir + '/prob_distri_of_all_class_voxels.png'.format(class_id), format="png")
    plt.savefig(output_dir + '/prob_distri_of_all_class_voxels.pdf'.format(class_id), format="pdf")
    plt.close()

    # # Plot prob. histogram of each individual
    for id in count.keys():
        plt.figure()
        id_inds = np.where(ids == id)[0]
        vote_label = np.sum(labels_int[id_inds]) * 1.0 / id_inds.size
        vote_pred = np.sum(predictions[id_inds][:, 1]) / id_inds.size

        label_of_id = 0 if vote_label < 0.5 else 1
        pred_of_id = 0 if vote_pred < 0.5 else 1

        color = "slateblue" if label_of_id == pred_of_id else "r"
        eval = "right" if label_of_id == pred_of_id else "wrong"
            
        ax = plt.subplot(1, 1, 1)
        pred_hist = plt.hist(np.array(predictions[id_inds][:, 1]), align='mid', bins=10, range=(0.0, 1.0), color=color, label="predicted")
        ymin, ymax = ax.get_ylim()
        ymax += 1
        plt.vlines(0.5, ymin, ymax, colors='k', linestyles='--')
        plt.text(0.25, ymax, str(np.int(np.sum(predictions[id_inds][:, 1] < 0.5))), fontsize=base-4)
        plt.text(0.75, ymax, str(np.int(np.sum(predictions[id_inds][:, 1] >= 0.5))), fontsize=base-4)
        plt.legend()
        plt.ylabel("frequency")
        plt.xlabel("probability of classified as class 1")
        plt.title("True label {} - pred as {} / (in total {} voxels for id {})".format(label_of_id, pred_of_id, id_inds.size, id))
        plt.tight_layout()
        plt.savefig(output_dir + '/{}_prob_distri_of_id_{}.png'.format(eval, id), format="png")
        plt.close()
        
        if label_of_id == pred_of_id:  # record all right/wrong classifications
            rignt_patients = np.append(rignt_patients, id)
        else:
            wrong_patients = np.append(wrong_patients, id)
        
    np.savetxt(output_dir + '/wrong_patient_ids.csv', wrong_patients, fmt="%d", delimiter=",")
    np.savetxt(output_dir + '/right_patient_ids.csv', rignt_patients, fmt="%d", delimiter=",")


def plot_aug_examples(new_mean, num2average, spec, true_labels, args):
    """
    new_mean, num2average, X_train[:, 3:], X_train[:, 2], args
    :param new_mean:
    :param num2average:
    :param spec:
    :param true_labels:
    :param args:
    :return:
    """
    row, col = 4, 2
    true_labels = true_labels.astype(np.int)
    plt.figure()
    rand_inds = np.random.choice(spec.shape[0], row * col)
    f, axs = plt.subplots(row, col, sharex=True)
    colors = ["royalblue", "darkviolet"]
    class_names = ["healthy", "tumor"]
    num_to_plot = 5
    for jj in range(row):
        for ii in range(col):
            start = np.random.choice((len(new_mean) - 40))
            end = start + num_to_plot
            ind = jj * col + ii
            NEW = args.aug_scale * new_mean[start:end] + np.tile((1 - args.aug_scale) * spec[ind], (num_to_plot, 1))
            axs[jj, ii].plot(NEW.T, linewidth=0.8)
            axs[jj, ii].plot(spec[ind], colors[true_labels[ind]], linewidth=2.0, label=class_names[true_labels[ind]])
            axs[jj, ii].legend(loc="best")
    f.text(0.5, 0.05, 'index', fontsize=16),
    f.text(0.05, 0.5, 'Normalized amplitude', fontsize=16, rotation=90, verticalalignment='center'),
    f.subplots_adjust(hspace=0, wspace=0.06)
    f.savefig(os.path.join(args.output_path, "augmenting_with_{}*{}_using_{}-samples.png".format(args.aug_method, args.aug_scale, num2average)), format="png")
    f.savefig(os.path.join(args.output_path, "augmenting_with_{}*{}_using_{}-samples.pdf".format(args.aug_method, args.aug_scale, num2average)), format="pdf")
    plt.close()


def plot_mean_of_each_class(train_spec, train_lbs, test_spec, test_args):
    row, col = 2, 2
    plt.figure()
    num_to_plot = 50
    rand_inds = np.random.choice(test_spec.shape[0], num_to_plot)
    f, axs = plt.subplots(row, col, sharex=True)
    colors = ["royalblue", "darkviolet"]
    class_names = ["healthy", "tumor"]
    mean_train = np.mean(train_spec, axis=0)
    mean_test = np.mean(test_spec, axis=0)
    for class_id in range(args.num_classes):
        inds = np.where()
    for jj, ind in enumerate(rand_inds):
        NEW = args.aug_scale * new_mean[start:end] + np.tile((1 - args.aug_scale) * spec[ind], (num_to_plot, 1))
        axs[jj].plot(NEW.T, alpha=0.65, linewidth=0.8)
        axs[jj].plot(mean_train, colors[true_labels[ind]], label=class_names[true_labels[ind]])
        axs[jj].legend(loc="best")

    f.text(0.5, 0.05, 'index', fontsize=16),
    f.text(0.05, 0.5, 'Normalized amplitude', fontsize=16, rotation=90, verticalalignment='center'),
    f.savefig(os.path.join(args.output_path,
                           "augmenting_with_{}*{}_using_{}-samples.png".format(args.aug_method, args.aug_scale,
                                                                               num2average)), format="png")
    plt.close()


def plot_train_samples(samples, true_labels, args, postfix="samples", data_dim="1d"):
    """plot the trainin samples"""
    if data_dim == "1d":
        row, col = 6, 4
        for class_id in range(args.num_classes):
            indices = np.random.choice(np.where(true_labels == class_id)[0], min(row*col, np.where(true_labels == class_id)[0].size))
            f, axs = plt.subplots(row, col, sharex=True, figsize=[12, 8.5])
            if len(indices) > 0:
                samp_plot = samples[indices]
                f.suptitle("Training samples after aug. in class {}".format(class_id), fontsize=base)
                for ii in range(indices.size):
                    axs[ii // col, np.mod(ii, col)].plot(samp_plot[ii])
                plt.tight_layout()
                # plt.setp(ax1.get_xticklabels(), visible=False)
                # plt.subplots_adjust(hspace=0.00, wspace=0.15)
                plt.savefig(args.output_path + '/augmented_samples-class-{}-{}-{}.png'.format(class_id, args.aug_method, postfix), format = 'png'),
                plt.savefig(args.output_path + '/augmented_samples-class-{}-{}-{}.pdf'.format(class_id, args.aug_method, postfix), format = 'pdf')

                plt.close()
    elif data_dim == "2d" or args.data_mode == "mnist":
        row, col = 6, 6
        for class_id in range(args.num_classes):
            f, axs = plt.subplots(row, col, sharex=True, figsize=[12, 12])
            indices = np.random.choice(np.where(true_labels == class_id)[0],
                                       min(row*col, np.where(true_labels == class_id)[0].size))
            if len(indices) > 0:
                samp_plot = samples[indices]
                f.suptitle("Training samples after aug. in class {}".format(class_id), fontsize=base)
                for ii in range(indices.size):
                    axs[ii // col, np.mod(ii, col)].imshow(samp_plot[ii].reshape(args.height, args.width), interpolation="nearest", aspect="auto")
                plt.tight_layout()
                plt.subplots_adjust(hspace=0.00, wspace=0.00)
                plt.savefig(args.output_path + '/augmented_samples-class-{}-{}-{}.png'.format(class_id, args.aug_method, postfix),
                            format='png')
                plt.savefig(args.output_path + '/augmented_samples-class-{}-{}-{}.pdf'.format(class_id, args.aug_method, postfix),
                            format='pdf')
                plt.close()
