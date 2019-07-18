import fnmatch
import os
import random
import itertools
import scipy as scipy

from sklearn import metrics
from scipy import interp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as pylab


def find_files(directory, pattern='*.csv'):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

    return files

def plot_confusion_matrix(confm, num_classes, title='Confusion matrix', cmap='Blues', normalize=False, save_name="results/"):  # plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = (confm * 1.0 / confm.sum(axis=1)[:, np.newaxis])*1.0
        # cm = cm.astype('float16') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        cm = confm
        print('Confusion matrix, without normalization')
    
    plt.figure(figsize=[20, 13.6])
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)
    plt.colorbar()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] != 0:
            plt.text(j, i, np.int(cm[i, j]*100)/100.0, horizontalalignment="center", color="orangered", fontsize=20)


    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_name + '.png', format='png')
    plt.close()


def combine_probabilities_test(files):
    """
    Combine the probabilities from class0 and class1 into one csv file
    :param files:
    :return:
    """
    
    plt.figure()
    nn = np.empty((50, 0))
    for fn in files:
        id = os.path.basename(fn).split('-')[-1][0:-4]
        values = pd.read_csv(fn, header=None).values
        if 'class_0' in fn:
            values[:, 0] = 'class-0-' + values[:, 0]
        elif 'class_1' in fn:
            values[:, 0] = 'class-1-' + values[:, 0]
        nn = np.hstack((nn, values))
    
    np.savetxt(os.path.join(data_dir, "new-combined_test_probabilities-{}.csv".format(id)), np.array(nn), fmt='%s',
               delimiter=',', header='class0-dates,class0-prob,class1-dates,class1-prob')



def plot_auc_curve(labels_hot, pred, epoch=0, save_dir='./results'):
    """
    Plot AUC curve
    :param args:
    :param labels: 2d array, one-hot coding
    :param pred: 2d array, predicted probabilities
    :return:
    """
    f = plt.figure()
    fpr, tpr, _ = metrics.roc_curve(np.argmax(labels_hot, 1), pred_prob[:, 1])  # input the positive label's prob distribution
    auc = metrics.roc_auc_score(labels_hot, pred_prob)
    plt.plot(fpr, tpr, label="auc={0:.4f}".format(auc))
    plt.legend(loc=4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    f.savefig(save_dir + '/AUC_curve_step_{:.2f}-auc_{:.4}.png'.format(epoch, auc))

    plt.close()

# ------------------------------------------------


original = "../data/20190325/20190325-3class_lout40_val_data5-2class_human_performance844_with_labels.mat"

plot_name = "indi_rating_with_model"

if plot_name == "indi_rating_with_model":
    data_dir = "../data/20190325"
    model_results = "../results/2019-07-02T14-05-36-data-20190325-3class_lout40_val_data5-2class_human_performance844_with_labels-class-2-Res_ECG_CAM-test/AUC_curve_step_0.00-auc_0.7257-labels.csv"
    human_indi_rating = "../data/20190325/lout40-data5-doctor_ratings_individual.mat"
    # Get individual rater's prediction
    true_indi_lbs = {}
    human_indi_lbs = {}
    model_indi_lbs = {}
    indi_mat = scipy.io.loadmat(human_indi_rating)['a']
    indi_ratings = np.array(indi_mat)

    # Get model's prediction
    model_auc = pd.read_csv(model_results, header=0).values
    true_label = model_auc[:, 0].astype(np.int)
    pred_logits = model_auc[:, 1]
    pred_lb = np.argmax(pred_logits, axis=0)

    mean_fpr = []
    mean_tpr = []
    mean_score = []
    base_fpr = np.linspace(0, 1, 20)
    tpr_model = []

    start = 0
    plt.figure(figsize=[12, 10])
    colors = pylab.cm.cool(np.linspace(0, 1, 8))
    for i in range(indi_ratings.shape[1]):
        key = "{}".format(i)
        true_indi_lbs[key] = true_label[start: start+len(indi_ratings[0, i])]

        human_indi_lbs[key] = indi_ratings[0, i][:, 0]
        model_indi_lbs[key] = pred_logits[start: start+len(indi_ratings[0, i])]
        start = start + len(indi_ratings[0, i])

        indi_fpr, indi_tpr, _ = metrics.roc_curve(true_indi_lbs[key], human_indi_lbs[key])
        indi_score = metrics.roc_auc_score(true_indi_lbs[key], human_indi_lbs[key])
        mean_fpr.append(indi_fpr)
        mean_tpr.append(indi_tpr)
        mean_score.append(indi_score)

        indi_model_fpr, indi_model_tpr, _ = metrics.roc_curve(true_indi_lbs[key], model_indi_lbs[key])
        indi_model_score = metrics.roc_auc_score(true_indi_lbs[key], model_indi_lbs[key])
        tpr_temp = interp(base_fpr, indi_model_fpr, indi_model_tpr)
        tpr_model.append(tpr_temp)

        plt.plot(indi_fpr[1], indi_tpr[1], color="m", marker="*", alpha=0.35)
        plt.plot(indi_model_fpr, indi_model_tpr, color='royalblue', alpha=0.25, label='model {} AUC:  {:.3f}'.format(i+1, indi_model_score))
    mean_model_tpr = np.mean(np.array(tpr_model), axis=0)
    std_model_tpr = np.std(np.array(tpr_model), axis=0)
    mean_model_score = metrics.auc(base_fpr, mean_model_tpr)

    plt.plot(indi_fpr[1], indi_tpr[1], alpha=0.35, color="m", marker="*", label="individual radiologists")
    plt.plot(np.mean(np.array(mean_fpr)[:, 1]), np.mean(np.array(mean_tpr)[:, 1]), color="m", marker="d", markersize=6, label='human average AUC:  {:.3f}'.format(np.mean(np.array(mean_score))))
    plt.plot(base_fpr, mean_model_tpr, color="royalblue", linewidth=3.0, label='model average AUC:  {:.3f}'.format(mean_model_score))

    plt.title("ROC", fontsize=20)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.legend(loc=4)
    plt.ylabel('true positive rate', fontsize=18)
    plt.xlabel('false positive rate', fontsize=18)
    plt.savefig(os.path.join(data_dir, "model_with_human_rating_individual.png"), format='png')
    plt.savefig(os.path.join(data_dir, "model_with_human_rating_individual.pdf"), format='pdf')
    plt.close()

elif plot_name == "human_whole_with_model":
    data_dir = "../data/20190325"
    human_rating = "../data/20190325/human-ratings-20190325-3class_lout40_val_data5-2class.mat"
    # Get model's prediction
    model_auc = pd.read_csv(model_results, header=0).values
    label = model_auc[:, 0].astype(np.int)
    pred_logits = model_auc[:, 1]
    pred_lb = np.argmax(pred_logits, axis=0)

    # Get human's total labels
    hum_whole = scipy.io.loadmat(human_rating)["data_ratings"]
    human_lb = hum_whole[:, 0]
    human_features = hum_whole[:, 1:]
    hum_fpr, hum_tpr, _ = metrics.roc_curve(true_label, human_lb)
    hum_score = metrics.roc_auc_score(true_label, human_lb)

    # PLot human average rating
    plt.figure()
    plt.plot(hum_fpr[1], hum_tpr[1], 'purple', marker="*", markersize=4, label='human AUC: {:.3f}'.format(hum_score))

    # Plot trained model prediction
    fpr, tpr, _ = metrics.roc_curve(label, pred_logits)
    score = metrics.roc_auc_score(label, pred_logits)
    plt.plot(fpr, tpr, 'royalblue', linewidth=2, label='model AUC: {:.3f}'.format(score))

    plt.title("ROC", fontsize=20)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.legend(loc=4)
    plt.ylabel('true positive rate', fontsize=18)
    plt.xlabel('false positive rate', fontsize=18)
    plt.savefig(os.path.join(data_dir, "model_with_human_rating_collectively.pdf"), format='pdf')
    # plt.savefig(os.path.join(data_dir, "model_with_human_rating.eps"), format='eps')
    plt.savefig(os.path.join(data_dir, "model_with_human_rating_collectively.png"), format='png')
    plt.close()
elif plot_name == "all_ROCs":
    data_dir = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/1-20190325-data-trained-models/TEST/Res_CNN_CAM-LOUT40"
    files = find_files(data_dir, pattern="AUC_curve_step_0.00-auc*.csv")
    plt.figure(figsize=[10, 6.8])
    base_fpr = np.linspace(0, 1, 20)
    tprs = []
    for ind, fn in enumerate(files):
        values = pd.read_csv(fn, header=0).values
        true_lbs = values[:, 0]
        prob_1 = values[:, 1]
        fpr, tpr, _ = metrics.roc_curve(true_lbs, prob_1)
        score = metrics.roc_auc_score(true_lbs, prob_1)
        plt.plot(fpr, tpr, 'royalblue', alpha=0.35, label='cross val {} AUC: {:.3f}'.format(ind, score))

        tpr_temp = interp(base_fpr, fpr, tpr)
        tprs.append(tpr_temp)
        print("ok")
    mean_model_tpr = np.mean(np.array(tprs), axis=0)
    std_model_tpr = np.std(np.array(tprs), axis=0)
    mean_model_score = metrics.auc(base_fpr, mean_model_tpr)

    plt.plot(base_fpr, mean_model_tpr, 'violet', linewidth=4.0, label='average AUC: {:.3f}'.format(mean_model_score))
    plt.title("ROC curves in cross validation test trials", fontsize=20)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.legend(loc="best")
    plt.ylabel('true positive rate', fontsize=18)
    plt.xlabel('false positive rate', fontsize=18)
    plt.savefig(os.path.join(data_dir, "All ROC curves in cross validation test-lout40-validation.png"), format='png')
    plt.savefig(os.path.join(data_dir, "All ROC curves in cross validation test-lout40-validation.pdf"), format='pdf')
    plt.close()



