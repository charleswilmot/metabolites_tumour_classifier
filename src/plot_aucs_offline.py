import fnmatch
import os
import random
import itertools
import pickle

from sklearn.metrics import confusion_matrix
from scipy import io
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
from collections import Counter
from sklearn import metrics
from textwrap import wrap

from dataio import introduce_label_noisy

# import matplotlib.pylab as pylab
# base = 22
# args = {'legend.fontsize': base - 8,
#           'figure.figsize': (10, 7),
#          'axes.labelsize': base-4,
#         #'weight' : 'bold',
#          'axes.titlesize':base,
#          'xtick.labelsize':base-8,
#          'ytick.labelsize':base-8}
# pylab.rcParams.update(args)

import matplotlib.pylab as pylab
base = 20
args = {
    # 'legend.fontsize': base - 4,
          'figure.figsize': (6, 4.8),
          # 'axes.labelsize': base-4,
          # 'axes.titlesize': base,
          # 'xtick.labelsize': base-10,
          # 'ytick.labelsize': base-10
    }
pylab.rcParams.update(args)

def find_files(directory, pattern='*.csv'):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

    return files


def find_folderes(directory, pattern='*.csv'):
    folders = []
    for root, dirnames, filenames in os.walk(directory):
        for subfolder in fnmatch.filter(dirnames, pattern):
            folders.append(os.path.join(root, subfolder))

    return folders


def find_optimal_cutoff(true_lbs, predicted):
    fpr, tpr, threshold = metrics.roc_curve(true_lbs, predicted)
    auc = metrics.roc_auc_score(true_lbs, predicted)
    ind = np.argsort(np.abs(tpr - (1-fpr)))[1]
    # i = np.arange(len(tpr))
    # roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    # roc_t = roc.ix[(roc.tf-0).abs().argsort()[:2]]

    # return list(roc_t['threshold']), auc
    return threshold[ind], auc


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



def plot_auc_curve(labels_hot, pred_prob, epoch=0, save_dir='./results'):
    """
    Plot AUC curve
    :param args:
    :param labels: 2d array, one-hot coding
    :param pred_prob: 2d array, predicted probabilities
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


def get_auc_as_factor(data_dir, fold=None, epoch=5, factor=0.5, aug_meth=["same", "ops"], colors=pylab.cm.Set2(np.linspace(0, 1, 6))):
    """
    Get auc as a function of aug factor with three different aug methods
    header = np.array(["method", "fold", "factor", "epoch", "auc"])
    :param epoch:
    :param aug_meth:
    :return:
    """
    colors = ["royalblue", "paleturquoise", "orangered", "mistyrose", "limegreen", "palegreen"]
    fold_ind = 1
    factor_ind = 2
    epoch_ind = 3
    header = ["mothod", "fold", "factor", "epoch", "auc"]
    if not epoch:
        base_var = [1, 3, 5, 8, 10]
        var_name = "epoch"
        var_ind = 3
        fix_ind1 = 1
        fix_ind2 = 2
        fix_value1 = fold
        fix_value2 = factor
    elif not factor:
        base_var = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
        # base_var = np.linspace(0.05, 1.0, 7)
        var_name = "factor"
        fix_value1 = fold
        fix_value2 = epoch
        var_ind = 2
        fix_ind1 = 1
        fix_ind2 = 3
    elif not fold:
        base_var = [1, 3, 5, 7, 9, 11]
        var_name = "fold"
        var_ind = 1
        fix_ind1 = 2
        fix_ind2 = 3
        fix_value1 = factor
        fix_value2 = epoch

    baseline_fn = 0.7
    baseline_rf = 0.635
    baseline_svm = 0.623


    plt.figure()
    for ind, method in enumerate(aug_meth):
        files = find_files(data_dir, pattern='*aug_{}*.txt'.format(method))

        sum_aucs = []

        for fn in files:
            print("Model {}, fix_name1 {}: {}, fix_name2 {}: {}".format(os.path.basename(fn).split("_")[1], header[fix_ind1], fix_value1, header[fix_ind2], fix_value2))
            data = pd.read_csv(fn, header=0).values
            need_inds = np.where(
                (data[:, fix_ind1] == fix_value1) &
                (data[:, fix_ind2] == fix_value2))[0]
            new_data = data[need_inds]

            sort_data = sorted(new_data, key=lambda x: x[var_ind])
            var_values = np.array(sort_data)[:, var_ind].astype(np.float)
            auc = np.array(sort_data)[:, -1].astype(np.float)
            auc_interp = interp(base_var, var_values, auc)
            sum_aucs.append(auc_interp)
            print("ok")

        mean_auc = np.mean(np.array(sum_aucs), axis=0)
        std_auc = scipy.stats.sem(np.array(sum_aucs))
        # std_auc = np.std(np.array(sum_aucs), axis=0) / np.sqrt(np.array(sum_aucs).size)
        

        plt.plot(base_var, mean_auc, color=colors[ind * 2], marker="o", s=8, linewidth=2.5, label=method)
        plt.errorbar(base_var, mean_auc, yerr=std_auc, capsize=5, color=colors[ind * 2])
        # plt.fill_between(base_var, mean_auc - std_auc, mean_auc + std_auc, color=colors[ind * 2 + 1])

    # plt.hlines(baseline_fn, base_var[0], base_var[-1], label="MLP")
    plt.hlines(baseline_rf, base_var[0], base_var[-1], label="Random forest")
    plt.hlines(baseline_svm, base_var[0], base_var[-1], label="SVM")
    plt.legend()
    plt.ylim([0.4, 0.82])
    plt.xlabel("{}".format(var_name))
    plt.ylabel("area under the curve")
    plt.savefig(os.path.dirname(data_dir) + "/auc_as_factor_all_fix_{}-{}_and_{}-{}_var_{}-errbar.png".format(header[fix_ind1], fix_value1, header[fix_ind2], fix_value2, header[var_ind]), format="png")
    plt.savefig(os.path.dirname(data_dir) + "/auc_as_factor_all_fix_{}-{}_and_{}-{}_var_{}-errbar.pdf".format(header[fix_ind1], fix_value1, header[fix_ind2], fix_value2, header[var_ind]), format="pdf")
    plt.close()


def plot_mean_spec_in_cluster(mean, std, cluster_id, num_clusters, postfix, crosstab_count=[10, 100], save_folder="/results"):

    """

    :param fea:
    :param cluster_id:
    :param num_clusters:
    :param postfix:
    :param save_folder:
    :return:
    """
    data = np.hstack((mean.reshape(-1, 1), std.reshape(-1, 1)))
    plt.figure()
    # plt.errorbar(np.arange(spec_mean.size), spec_mean, spec_std, alpha=0.25)
    plt.fill_between(np.arange(mean.size), mean - std, mean + std,  facecolor='c')
    plt.plot(np.arange(mean.size), mean, 'm')
    plt.xlabel("index")
    plt.title("\n".join(wrap("{}-clusters No. {} cluster, count {}".format(num_clusters, cluster_id, crosstab_count), 60)))
    ylabel = "normalized value [a.u.]"
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(save_folder, "{}_clusters_label_{}-spec-{}-mean.png".format(num_clusters, cluster_id, postfix)), format="png")
    plt.savefig(os.path.join(save_folder, "{}_clusters_label_{}-spec-{}-mean.pdf".format(num_clusters, cluster_id, postfix)), format="pdf")
    plt.close()


def get_data_from_certain_ids(certain_fns, mat_file="../data/lout40_train_val_data5.mat"):
    """
    Load data from previous certain examples
    :param args:
    :param certain_fns: list of filenames, from train and validation
    :return:
    """
    mat = scipy.io.loadmat(mat_file)["DATA"]  # [id, label, features]
    labels = mat[:, 1]

    new_mat = np.zeros((mat.shape[0], mat.shape[1] + 1))
    new_mat[:, 0] = np.arange(mat.shape[0])  # tag every sample
    new_mat[:, 1:] = mat

    sub_inds = np.empty((0))
    for class_id in range(2):
        sub_inds = np.append(sub_inds, np.where(labels == class_id)[0])
    sub_inds = sub_inds.astype(np.int32)
    sub_mat = new_mat[sub_inds]

    # certain_mat = np.empty((0, new_mat.shape[1]))
    picked_ids = np.arange(len(new_mat))
    if certain_fns is not None:
        sort_data = pd.read_csv(certain_fns, header=0).values
        sort_samp_ids = sort_data[:, 0].astype(np.int)
        sort_rate = sort_data[:, 1].astype(np.float32)
        picked_ids = sort_samp_ids[-np.int(0.2 * len(sort_data)):]
        print(os.path.basename(certain_fns), len(picked_ids), "samples\n")
    
    return sub_mat, new_mat[picked_ids]


def get_data_from_mat(mat_file):
    """
    Load data from previous certain examples
    :param args:
    :param certain_fns: list of filenames, from train and validation
    :return:
    """
    mat = scipy.io.loadmat(mat_file)["DATA"]  # [id, label, features]
    labels = mat[:, 1]

    new_mat = np.zeros((mat.shape[0], mat.shape[1] + 1))
    new_mat[:, 0] = np.arange(mat.shape[0])  # tag every sample
    new_mat[:, 1:] = mat

    sub_inds = np.empty((0))
    for class_id in range(2):
        sub_inds = np.append(sub_inds, np.where(labels == class_id)[0])
    sub_inds = sub_inds.astype(np.int32)
    sub_mat = new_mat[sub_inds]
    
    return sub_mat


def two_axis_in_one_plot():
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel("sample index (sorted)")
    ax1.set_ylabel("rate over 100 runs", color=color)
    ax1.plot(rates, label="correct clf. rate", color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0,1.0])
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('rate (100 runs) ', color=color)  # we already handled the x-label with ax1
    ax2.plot(np.arange(len(ct_sele_rates)), ct_sele_rates, label="distilled selection rate", color=color)
    ax2.plot(np.arange(len(ct_corr_rates)), ct_corr_rates, label="distilled corr. rate", color='m')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc="upper right")
    print("ok")
    

def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),
                        'tpr' : pd.Series(tpr, index = i),
                        '1-fpr' : pd.Series(1-fpr, index = i),
                        'tf' : pd.Series(tpr - (1-fpr), index = i),
                        'threshold' : pd.Series(threshold, index = i)})

    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return roc_t, list(roc_t['threshold'])


def get_auc_from_d_prime(tpr=0.5, fpr=0.5):
    """
    Get auc from d' for human-level comparison
    :param tpr:
    :param fpr:
    :return:
    """
    from scipy.stats import norm
    exp_d_prime = norm.ppf(tpr) - norm.ppf(fpr)
    exp_auc = norm.cdf(exp_d_prime / np.sqrt(2))
    return exp_auc


def get_scalar_performance_matrices_2classes(true_labels, pred_logits, if_with_logits=False):
    """
    Get all relavant performance metrics
    :param true_labels: 1d array, int labels
    :param predictions: 1d array, logits[:, 1]
    :param if_with_logits: if with logits, it is with probabilities, otherwise are predicted int labels
    :return:
    """
    # get predicted labels based on optimal threshold
    if if_with_logits:
        cutoff_thr, auc = find_optimal_cutoff(true_labels, pred_logits)
        pred_labels = (pred_logits > cutoff_thr).astype(np.int)
    else:
        pred_labels = pred_logits
        auc = metrics.roc_auc_score(true_labels, pred_logits)
    
    confusion = metrics.confusion_matrix(true_labels, pred_labels)
    TN = confusion[0][0]
    FN = confusion[1][0]
    TP = confusion[1][1]
    FP = confusion[0][1]
    
    # accuracy
    accuracy = (TP + TN) / np.sum(confusion)
    # Sensitivity, hit rate, recall, or true positive rate
    sensitivity = TP / (TP + FN)
    # Specificity or true negative rate
    specificity = TN / (TN + FP)
    # Precision or positive predictive value
    precision = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # F1-score
    F1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    # F1_score = TP / (TP + 0.5*(FN + FP))
    # get tpr, fpr
    fpr, tpr, _ = metrics.roc_curve(true_labels, pred_logits)
    # Matthews corrrelation coefficient
    MCC = metrics.matthews_corrcoef(true_labels, pred_labels)
    MCC = (TP*TN - FP*FN)  / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

    return accuracy, sensitivity, specificity, precision, F1_score, auc, fpr, tpr, MCC


def split_data_for_lout_val(data):
    """
    Split the original data in leave several subjects
    :param data: 2d array, [n_samples, 291]: [sample_id, pat_id, lb, features288]
    :return: save two .mat files
    """
  
    count = dict(Counter(list(data[:, 1])))

    for i in range(len(count) // args.num_lout):
        validate = {}
        validate["features"] = np.empty((0, 288))
        validate["labels"] = np.empty((0))
        validate["ids"] = np.empty((0))

        lout_ids = pick_lout_ids(ids, count, num_lout=args.num_lout, start=i)  # leave 10 subjects out

        all_inds = np.empty((0))
        for id in lout_ids:
            inds = np.where(ids == id)[0]
            all_inds = np.append(all_inds, inds)
            validate["features"] = np.vstack((validate["features"], spectra[inds, :]))
            validate["labels"] = np.append(validate["labels"], labels[inds])
            validate["ids"] = np.append(validate["ids"], ids[inds])
        train_test_data = np.delete(mat, all_inds, axis=0)  # delete all leaved-out subjects
        print("Leave out: \n", lout_ids, "\n num_lout\n", len(validate["labels"]))

        # ndData
        val_mat = {}
        train_test_mat = {}
        val_mat["DATA"] = np.zeros((validate["labels"].size, 290))
        val_mat["DATA"][:, 0] = validate["ids"]
        val_mat["DATA"][:, 1] = validate["labels"]
        val_mat["DATA"][:, 2:] = validate["features"]
        train_test_mat["DATA"] = np.zeros((len(train_test_data), 290))
        train_test_mat["DATA"][:, 0] = train_test_data[:, 0]
        train_test_mat["DATA"][:, 1] = train_test_data[:, 1]
        train_test_mat["DATA"][:, 2:] = train_test_data[:, 2:]
        print("num_train\n", len(train_test_mat["DATA"][:, 1]))
        scipy.io.savemat(
            os.path.dirname(args.input_data) + '/20190325-{}class_lout{}_val_data{}.mat'.format(args.num_classes,
                                                                                                args.num_lout, i),
            val_mat)
        scipy.io.savemat(
            os.path.dirname(args.input_data) + '/20190325-{}class_lout{}_train_test_data{}.mat'.format(args.num_classes,
                                                                                                       args.num_lout,
                                                                                                       i),
            train_test_mat)

# ------------------------------------------------


original = "../data/20190325/20190325-3class_lout40_val_data5-2class_human_performance844_with_labels.mat"

plot_name = "certain_tsne_distillation"

# filename = r"C:\Users\LDY\Desktop\one_0_num_60000-59999_mnist_theta_1_s3142_for_checking.csv"
#
# data = pd.read_csv(filename, header=0).values
# #
# print("ok")
if plot_name == "plot_random_roc":
    print("Plot_name: ", plot_name)
    filename = "C:/Users/LDY/Desktop/1-all-experiment-results/Gk-patient-wise-classification/2021-01-06T22-46-36-classifier4-20spec-gentest-non-overlap-filter-16-aug-add_additive_noisex5/classifier4-spec51-CV9--ROC-AUC-[n_cv_folds,n_spec_per_pat].csv"
    
    values = pd.read_csv(filename, header=None).values
    
    print("ok")
    mean_values = np.mean(values, axis=0)
    std_values = np.std(values, axis=0)
    plt.plot(np.arange(1, 52, 5),values[5])
    plt.errorbar(np.arange(1, 52, 5), mean_values, yerr=std_values, capsize=5)

    
elif plot_name == "indi_rating_with_model":
    """
    Get performance of doctors, model_without_aug and model_with_aug on individual chuncks
    give individual doctors' ratings, model logits
    """
    print("Plot_name: ", plot_name)
    data_dir = "../data/20190325"
    # Get individual rater's prediction
    human_indi_rating = "../data/20190325/20190325-3class_lout40_test_data5-2class_doctor_ratings_individual_new.mat"

    indi_mat = scipy.io.loadmat(human_indi_rating)['a']
    indi_ratings = np.array(indi_mat)

    # Get model's prediction
    true_data = scipy.io.loadmat("../data/20190325/20190325-3class_lout40_test_data5-2class_human_performance844_with_labels.mat")["DATA"]
    true_label = true_data[:, 1].astype(np.int)
    
    model_res_wo_aug_fn = "C:/Users/LDY/Desktop/1-all-experiment-results/metabolites/auc-func-as-augmentation-parameters/with-randDA-AUC_curve_step_0.00-auc_0.7198-data5-test.csv"
    model_auc_wo_aug = pd.read_csv(model_res_wo_aug_fn, header=0).values
    true_model_label_wo_aug = model_auc_wo_aug[:, 0].astype(np.int)
    pred_logits_wo_aug = model_auc_wo_aug[:, 1]
    
    model_res_with_aug_fn = "C:/Users/LDY/Desktop/1-all-experiment-results/metabolites/auc-func-as-augmentation-parameters/with-DA-dist-AUC_curve_step_0.00-auc_0.7672-data5-test.csv"
    model_auc_with_aug = pd.read_csv(model_res_with_aug_fn, header=0).values
    true_model_label_with_aug = model_auc_with_aug[:, 0].astype(np.int)
    pred_logits_with_aug = model_auc_with_aug[:, 1]

    # Get human's cumulative labels
    human_rating = "../data/20190325/20190325-3class_lout40_test_data5-2class_human-ratings.mat"
    hum_whole = scipy.io.loadmat(human_rating)["data_ratings"]
    human_lb = hum_whole[:, 0]
    human_features = hum_whole[:, 1:]
    hum_fpr, hum_tpr, _ = metrics.roc_curve(true_label, human_lb)
    hum_score = metrics.roc_auc_score(true_label, human_lb)
    
    # trace each individual rating
    mean_fpr = []
    mean_tpr = []
    mean_score = []
    base_fpr = np.linspace(0, 1, 20)
    coll_tpr_model_wo_aug = []
    coll_tpr_model_with_aug = []
    start = 0
    
    colors = pylab.cm.cool(np.linspace(0, 1, 8))
    coll_indiv_doctor_AUCs = []
    coll_indiv_doctor_d_prime_aucs = []
    coll_indiv_doctor_f1_scores = []
    coll_indiv_doctor_MCCs = []
    coll_indiv_doctor_SEN = []
    coll_indiv_doctor_SPEC = []
    coll_indiv_doctor_ACC = []
    doctor_performance_str = []
    
    coll_indiv_model_AUCs_wo_aug = []
    coll_indiv_model_d_prime_aucs_wo_aug = []
    coll_indiv_model_f1_scores_wo_aug = []
    coll_indiv_model_MCCs_wo_aug = []
    coll_indiv_model_SEN_wo_aug = []
    coll_indiv_model_SPEC_wo_aug = []
    coll_indiv_model_ACC_wo_aug = []
    model_performance_str_wo_aug = []
    
    coll_indiv_model_AUCs_with_aug = []
    coll_indiv_model_d_prime_AUCs_with_aug = []
    coll_indiv_model_f1_scores_with_aug = []
    coll_indiv_model_MCCs_with_aug = []
    coll_indiv_model_SEN_with_aug = []
    coll_indiv_model_SPEC_with_aug = []
    coll_indiv_model_ACC_with_aug = []
    model_performance_str_with_aug = []
    
    plt.figure(figsize=[8, 5.8])
    true_indi_doctor_lbs = {}
    true_indi_model_lbs_wo_aug = {}
    true_indi_model_lbs_with_aug = {}

    indi_doctor_logits = {}
    indi_model_logits_wo_aug = {}
    indi_model_logits_with_aug = {}
    for i in range(indi_ratings.shape[1]):
        key = "{}".format(i)
        print(key)
        end = start + min(len(indi_ratings[0, i]), len(true_model_label_wo_aug) - start)
        true_indi_doctor_lbs[key] = true_label[start: start + len(indi_ratings[0, i])]
        true_indi_model_lbs_wo_aug[key] = true_model_label_wo_aug[start: end]
        true_indi_model_lbs_with_aug[key] = true_model_label_with_aug[start: end]

        indi_doctor_logits[key] = indi_ratings[0, i][:, 0]
        indi_model_logits_wo_aug[key] = pred_logits_wo_aug[start: end]
        indi_model_logits_with_aug[key] = pred_logits_with_aug[start: end]
       
        # get all summary of performance metrics of doctor's performance
        indi_doctor_acc, indi_doctor_sensitivity, indi_doctor_specificity, indi_doctor_precision, indi_doctor_F1_score, indi_doctor_auc, indi_doctor_fpr, indi_doctor_tpr, indi_doctor_mcc =  get_scalar_performance_matrices_2classes(true_indi_doctor_lbs[key], indi_doctor_logits[key], if_with_logits=False)
        # collect individual doctor performance
        coll_indiv_doctor_AUCs.append(np.int(indi_doctor_auc*1000)/1000)
        coll_indiv_doctor_d_prime_aucs.append(get_auc_from_d_prime(tpr=indi_doctor_tpr[1], fpr=indi_doctor_fpr[1]))
        coll_indiv_doctor_f1_scores.append(np.int(indi_doctor_F1_score*1000)/1000)
        coll_indiv_doctor_MCCs.append(np.int(indi_doctor_mcc*1000)/1000)
        coll_indiv_doctor_SEN.append(np.int(indi_doctor_sensitivity*1000)/1000)
        coll_indiv_doctor_SPEC.append(np.int(indi_doctor_specificity*1000)/1000)
        coll_indiv_doctor_ACC.append(np.int(indi_doctor_acc*1000)/1000)

        # get summary of without_aug_model performance's metrics
        indi_model_acc_wo_aug, \
        indi_model_sensitivity_wo_aug, \
        indi_model_specificity_wo_aug, \
        indi_model_precision_wo_aug, \
        indi_model_F1_score_wo_aug, \
        indi_model_AUC_wo_aug, \
        indi_model_fpr_wo_aug, \
        indi_model_tpr_wo_aug, \
        indi_model_MCC_wo_aug = get_scalar_performance_matrices_2classes(true_indi_model_lbs_wo_aug[key], indi_model_logits_wo_aug[key], if_with_logits=True)
        # collect individual corresponding model_WO_AUG performance
        coll_indiv_model_AUCs_wo_aug.append(np.int(indi_model_AUC_wo_aug * 1000) / 1000)
        coll_indiv_model_f1_scores_wo_aug.append(np.int(indi_model_F1_score_wo_aug * 1000) / 1000)
        coll_indiv_model_MCCs_wo_aug.append(np.int(indi_model_MCC_wo_aug * 1000) / 1000)
        coll_indiv_model_SEN_wo_aug.append(np.int(indi_model_sensitivity_wo_aug * 1000) / 1000)
        coll_indiv_model_SPEC_wo_aug.append(np.int(indi_model_specificity_wo_aug * 1000) / 1000)
        coll_indiv_model_ACC_wo_aug.append(np.int(indi_model_acc_wo_aug * 1000) / 1000)
        
        # get summary of dist+da+model performance
        indi_model_acc_with_aug, \
        indi_model_sensitivity_with_aug, \
        indi_model_specificity_with_aug, \
        indi_model_precision_with_aug, \
        indi_model_F1_score_with_aug, \
        indi_model_AUC_with_aug, \
        indi_model_fpr_with_aug, \
        indi_model_tpr_with_aug, \
        indi_model_MCC_with_aug = get_scalar_performance_matrices_2classes(true_indi_model_lbs_with_aug[key], indi_model_logits_with_aug[key], if_with_logits=True)
        # collect individual corresponding model_WITH_AUG performance
        coll_indiv_model_AUCs_with_aug.append(np.int(indi_model_AUC_with_aug * 1000) / 1000)
        coll_indiv_model_f1_scores_with_aug.append(np.int(indi_model_F1_score_with_aug * 1000) / 1000)
        coll_indiv_model_MCCs_with_aug.append(np.int(indi_model_MCC_with_aug * 1000) / 1000)
        coll_indiv_model_SEN_with_aug.append(np.int(indi_model_sensitivity_with_aug * 1000) / 1000)
        coll_indiv_model_SPEC_with_aug.append(np.int(indi_model_specificity_with_aug * 1000) / 1000)
        coll_indiv_model_ACC_with_aug.append(np.int(indi_model_acc_with_aug * 1000) / 1000)
        
        # collect the interpolated tpr
        tpr_temp_wo_aug = np.interp(base_fpr, indi_model_fpr_wo_aug, indi_model_tpr_wo_aug)
        coll_tpr_model_wo_aug.append(tpr_temp_wo_aug)

        tpr_temp_with_aug = np.interp(base_fpr, indi_model_fpr_with_aug, indi_model_tpr_with_aug)
        coll_tpr_model_with_aug.append(tpr_temp_with_aug)
        
        # plot scatter and ROC curve
        plt.scatter(indi_doctor_fpr[1], indi_doctor_tpr[1], color="r", marker="o", s=40, alpha=0.65)

        start = end
        
    ### one by one compare the metrix of doctors and the model
    # print("doctors Sensitivity {}\n: mean-{:.3f}, std-{:.3f}\n".format(coll_doctor_SEN,
    #     np.mean(coll_doctor_SEN),
    #     np.std(coll_doctor_SEN)))
    # print("Model Sensitivity {}\n: mean-{:.3f}, std-{:.3f}\n".format(coll_model_SEN_wo_aug,
    #                                                                  np.mean(coll_model_SEN_wo_aug),
    #                                                                  np.std(coll_model_SEN_wo_aug)))
    # print("doctors specificity {}\n: mean-{:.3f}, std-{:.3f}\n".format(coll_doctor_SPEC,
    #     np.mean(coll_doctor_SPEC),
    #     np.std(coll_doctor_SPEC)))
    # print("Model specificity {}\n: mean-{:.3f}, std-{:.3f}\n".format(
    #     coll_model_SPEC_wo_aug,
    #     np.mean(coll_model_SPEC_wo_aug),
    #     np.std(coll_model_SPEC_wo_aug)))
    # print("doctors AUC {}\n: mean-{:.3f}, std-{:.3f}\n".format(coll_doctor_aucs, np.mean(coll_doctor_aucs),
    #                                               np.std(coll_doctor_aucs)))
    # print("Model AUC {}\n: mean-{:.3f}, std-{:.3f}\n".format(coll_model_aucs_wo_aug,
    #                                                          np.mean(
    #                                                              coll_model_aucs_wo_aug),
    #                                                          np.std(
    #                                                              coll_model_aucs_wo_aug)))
    # print("doctors acc {}\n: mean-{:.3f}, std-{:.3f}\n".format(coll_doctor_ACC,
    #     np.mean(coll_doctor_ACC),
    #     np.std(coll_doctor_ACC)))
    # print("Model acc {}\n: mean-{:.3f}, std-{:.3f}\n".format(coll_model_ACC_wo_aug,
    #                                                          np.mean(
    #                                                              coll_model_ACC_wo_aug),
    #                                                          np.std(
    #                                                              coll_model_ACC_wo_aug)))
    # print("doctors F1-score {}\n: mean-{:.3f}, std-{:.3f}\n".format(coll_doctor_f1_scores,
    #     np.mean(coll_doctor_f1_scores),
    #     np.std(coll_doctor_f1_scores)))
    # print("Model F1-score {}\n: mean-{:.3f}, std-{:.3f}\n".format(
    #     coll_model_f1_scores_wo_aug,
    #     np.mean(coll_model_f1_scores_wo_aug),
    #     np.std(coll_model_f1_scores_wo_aug)))
    # print("doctors MCC {}\n: mean-{:.3f}, std-{:.3f}\n".format(coll_doctor_MCCs, np.mean(coll_doctor_MCCs),
    #                                               np.std(coll_doctor_MCCs)))
    # print("Model MCC {}\n: mean-{:.3f}, std-{:.3f}\n".format(coll_model_MCCs_wo_aug, np.mean(coll_model_MCCs_wo_aug),
    #                                                          np.std(coll_model_MCCs_wo_aug)))

    
    mean_model_tpr_wo_aug = np.mean(np.array(coll_tpr_model_wo_aug), axis=0)
    mean_model_tpr_wo_aug = np.insert(mean_model_tpr_wo_aug, 0, 0)
    std_model_tpr_wo_aug = np.std(np.array(coll_tpr_model_wo_aug), axis=0)
    
    mean_model_tpr_with_aug = np.mean(np.array(coll_tpr_model_with_aug), axis=0)
    mean_model_tpr_with_aug = np.insert(mean_model_tpr_with_aug, 0, 0)
    std_model_tpr_with_aug = np.std(np.array(coll_tpr_model_with_aug), axis=0)
    
    mean_model_score_with_aug = np.mean(coll_indiv_model_AUCs_with_aug)
    mean_model_score_wo_aug = np.mean(coll_indiv_model_AUCs_wo_aug)

    plt.scatter(indi_doctor_fpr[1], indi_doctor_tpr[1], alpha=0.65, color="r", marker="o", s=40, label="individual radiologists")
    # plt.scatter(hum_fpr[1], hum_tpr[1], color="purple", marker="*", s=120, label='cumulative radiologists performance(F1: {:.2f}, MCC: {:.2f})'.format(np.mean(coll_indiv_doctor_f1_scores), np.mean(coll_indiv_doctor_MCCs)))  #  label='cumulative radiologists performance'
    # plt.plot(np.insert(base_fpr, 0, 0), mean_model_tpr_wo_aug, linestyle=":",
    #          linewidth=3.0, color="royalblue",
    #          label='model AUC:  {:.2f} (F1: {:.2f}, MCC: {:.2f})'.format(np.mean(coll_indiv_model_AUCs_wo_aug), np.mean(coll_indiv_model_f1_scores_wo_aug), np.mean(coll_indiv_model_MCCs_wo_aug)))  # 'model average'
    # plt.plot(np.insert(base_fpr, 0, 0), mean_model_tpr_with_aug, linewidth=3.0, color="crimson", label='model+DA+dist AUC:  {:.2f} (F1: {:.2f}, MCC: {:.2f})'.format(np.mean(coll_indiv_model_AUCs_with_aug), np.mean(coll_indiv_model_f1_scores_with_aug), np.mean(coll_indiv_model_MCCs_with_aug)))#  ,label='model + DA + dist'
    plt.scatter(hum_fpr[1], hum_tpr[1], color="purple", marker="*", s=120, label='cumulative radiologists performance')  #  label='cumulative radiologists performance'
    plt.plot(np.insert(base_fpr, 0, 0), mean_model_tpr_wo_aug, linestyle=":",
             linewidth=3.0, color="royalblue",
             label='model AUC:  {:.2f}'.format(np.mean(coll_indiv_model_AUCs_wo_aug)))  # 'model average'
    plt.plot(np.insert(base_fpr, 0, 0), mean_model_tpr_with_aug, linewidth=3.0, color="crimson", label='model+DA+dist AUC:  {:.2f}'.format(np.mean(coll_indiv_model_AUCs_with_aug)))#  ,label='model + DA + dist'
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.legend(loc=4, frameon=False)
    plt.ylabel('true positive rate')
    plt.xlabel('false positive rate')
    plt.tight_layout()
    
    plt.savefig(os.path.join(data_dir, "Model_with_indi_human_rating_only_AUC.png"), format='png')
    plt.savefig(os.path.join(data_dir, "Model_with_indi_human_rating_only_AUC.pdf"), format='pdf')
    plt.close()
    

elif plot_name == "human_whole_with_model":
    print("Plot_name: ", plot_name)
    data_dir = "../data/20190325"
    original = "../data/20190325/20190325-3class_lout40_test_data5-2class_human_performance844_with_labels.mat"
    ori = scipy.io.loadmat(original)["DATA"]
    true_label = ori[:, 1]

    # Get model's prediction
    model_res_with_aug = "C:/Users/LDY/Desktop/1-all-experiment-results/metabolites/auc-func-as-augmentation-parameters/with-DA-dist-AUC_curve_step_0.00-auc_0.7672-data5-test.csv"
    model_res_wo_aug = "C:/Users/LDY/Desktop/1-all-experiment-results/metabolites/auc-func-as-augmentation-parameters/with-randDA-AUC_curve_step_0.00-auc_0.7198-data5-test.csv"
    model_auc_with_aug = pd.read_csv(model_res_with_aug, header=0).values
    label_with_aug = model_auc_with_aug[:, 0].astype(np.int)
    pred_logits_with_aug = model_auc_with_aug[:, 1]

    model_auc_wo_aug = pd.read_csv(model_res_wo_aug, header=0).values
    label_wo_aug = model_auc_wo_aug[:, 0].astype(np.int)
    pred_logits_wo_aug = model_auc_wo_aug[:, 1]

    # Get human's total labels
    human_rating = "../data/20190325/20190325-3class_lout40_test_data5-2class_human-ratings.mat"
    hum_whole = scipy.io.loadmat(human_rating)["data_ratings"]
    human_lb = hum_whole[:, 0]
    human_features = hum_whole[:, 1:]
    hum_fpr, hum_tpr, _ = metrics.roc_curve(true_label, human_lb)
    hum_score = metrics.roc_auc_score(true_label, human_lb)
    
    # PLot human average rating
    plt.figure(figsize=[10, 7])
    plt.scatter(hum_fpr[1], hum_tpr[1], color='purple', marker="*", s=50, label='cumulative performance'.format(hum_score))

    # Plot trained model prediction
    fpr_with_aug, tpr_with_aug, _ = metrics.roc_curve(label_with_aug, pred_logits_with_aug)
    fpr_wo_aug, tpr_wo_aug, _ = metrics.roc_curve(label_wo_aug, pred_logits_wo_aug)
    score_with_aug = metrics.roc_auc_score(label_with_aug, pred_logits_with_aug)
    score_wo_aug = metrics.roc_auc_score(label_wo_aug, pred_logits_wo_aug)

    plt.plot(fpr_with_aug, tpr_with_aug, 'royalblue', linestyle="-", linewidth=2, label='With aug. AUC: {:.2f}'.format(score_with_aug))  #, label='With aug. AUC: {:.2f}'.format(score_with_aug)
    plt.plot(fpr_wo_aug, tpr_wo_aug, 'violet', linestyle="-.", linewidth=2, label='Without aug. AUC: {:.2f}'.format(score_wo_aug))
    plt.title("Receiver Operating Characteristic", fontsize=20)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.legend(loc=4)
    plt.ylabel('true positive rate', fontsize=18)
    plt.xlabel('false positive rate', fontsize=18)
    plt.savefig(os.path.join(data_dir, "model_with_human_rating_collectively_certain.pdf"), format='pdf')
    # plt.savefig(os.path.join(data_dir, "model_with_human_rating.eps"), format='eps')
    plt.savefig(os.path.join(data_dir, "model_with_human_rating_collectively_certain.png"), format='png')
    plt.close()


elif plot_name == "all_ROCs":
    print("Plot_name: ", plot_name)
    data_dir = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/certain-DA-Res7-updateDataio-Res_ECG_CAM/great"

    files = find_files(data_dir, pattern="AUC_curve_step_0.00-auc*.csv")
    tprs = []
    plt.figure(figsize=[10, 6.8])
    base_fpr = np.linspace(0, 1, 20)
    for ind, fn in enumerate(files):
        values = pd.read_csv(fn, header=0).values
        true_lbs = values[:, 0]
        prob_1 = values[:, 1]
        fpr_with_aug, tpr_with_aug, _ = metrics.roc_curve(true_lbs, prob_1)
        score = metrics.roc_auc_score(true_lbs, prob_1)
        plt.plot(fpr_with_aug, tpr_with_aug, 'royalblue', label='cross val {} AUC: {:.3f}'.format(ind, score))

        tpr_temp = np.interp(base_fpr, fpr_with_aug, tpr_with_aug)
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


elif plot_name == "average_models":
    print("Plot_name: ", plot_name)
    file_dirs = ["/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-09-09T14-27-42-data-20190325-3class_lout40_val_data5-class-2-Res_ECG_CAM--filter144-bl5-ch16-aug0.1-mean-test/AUCs/AUC_curve_step_0.00-auc_0.7176-lout40-data5.csv", "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-09-09T14-27-11-data-20190325-3class_lout40_val_data5-class-2-Res_ECG_CAM--filter144-bl5-ch16-aug0.1-mean-test/AUCs/AUC_curve_step_0.00-auc_0.6669-lout40-data5.csv", "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-09-09T14-23-45-data-20190325-3class_lout40_val_data5-class-2-Res_ECG_CAM--filter144-bl5-ch16-aug0.1-mean-test/AUCs/AUC_curve_step_0.00-auc_0.6662-lout40-data5.csv"]
    predictions = {}

    for ind, fn in enumerate(file_dirs):
        values = pd.read_csv(fn, header=0).values
        true_lbs = values[:, 0]
        predictions["{}".format(ind)] = values[:, 1]
        if ind == 0:
            agg_pred = values[:, 1]
        else:
            agg_pred += values[:, 1]
    print("ok")


elif plot_name == "plot_mean_cluster":
    print("Plot_name: ", plot_name)
    data_dir = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/src/KMeans/metabolites_clustering-spec-whole-20190325/2019-12-16T13-34-47/7-cluster/plots"
    mean_files = find_files(data_dir, pattern="*.csv")
    for fn in mean_files:
        num_clusters = fn.split("_")[0]
        cluster_id = fn.split("_")[-1].split("-")[0]
        data = pd.read_csv(fn, header=0).values
        mean = data[:, 0]
        std = data[:, 1]

        plot_mean_spec_in_cluster(mean, std, cluster_id, num_clusters, "train", crosstab_count=[None], save_folder=data_dir)


elif plot_name == "test_performance_with_different_data_aug_parameters":
    print("Plot_name: ", plot_name)
    from_dirs = True   #False  #
    if from_dirs:
        data_dir = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-with-distillation-MLP"
        model = os.path.basename(data_dir).split("-")[-1]
        exp_mode = os.path.basename(data_dir).split("-")[-2]

        for data_source in ["data1", "data2", "data3", "data4", "data5", "data6", "data7", "data8", "data9"]:  #, "data7", "data9", "data1", "data3"
            # data_source = "data7"
            pattern = "*-{}-test-*".format(data_source)
            folders = find_folderes(data_dir, pattern=pattern)
            if len(folders) == 0:
                continue

            configs = []  # "aug_method": [], "aug_factor": [], "aug_fold": [], "from_epoch":
            indplus2 = 2 if "CAM" in model else 0

            aug_name_encode = {"same":0, "ops":1, "both":2}
            for fn in folders:
                print(fn)
                splits = os.path.basename(fn).split("-")
                aug_name = aug_name_encode[splits[7+indplus2]]
                aug_fold = np.int(splits[8+indplus2].split("x")[-1])
                aug_factor = np.float(splits[10+indplus2])
                test_auc = np.float(splits[-1])
                if "random" in pattern:
                    theta = 1
                else:
                    theta = 1
                configs.append((aug_name, aug_fold, aug_factor, theta, test_auc))

            print("ok")
            configs = np.array(configs)
            aug_same = configs[np.where(configs[:, 0] == aug_name_encode["same"])[0]]
            aug_ops = configs[np.where(configs[:, 0] == aug_name_encode["ops"])[0]]
            aug_both = configs[np.where(configs[:, 0] == aug_name_encode["both"])[0]]

            scale_style = {0.05: "-", 0.2: "-.", 0.35: "--", 0.5: ":"}
            meth_color = {"ops":"tab:orange", "same":"tab:green", "both":"tab:brown"}
            fold_markers = {1: "-d", 3: "-*", 5: "-o", 9: "-^"}
            styles = {1:":", 3:"-.", 5:"--", 9:"-"}
            
            # plot aug. method with error bar
            plt.figure(figsize=[12, 8])
            for res, method in zip([aug_same, aug_ops, aug_both], ["same", "ops", "both"]):
                for fold in [1,3,5,9]:
                    fd_configs = np.array(res[np.where(res[:,1] == fold)[0]])
                    if len(fd_configs) > 0:
                        value_per_scale = []
                        for scale in [0.05, 0.2, 0.35, 0.5]:
                            print("{}, {}, {}, auc".format(method, fold, scale))
                            if len(np.where(fd_configs[:,2] == scale)[0]) >= 1:
                                value_per_scale.append([np.float(scale), np.mean(fd_configs[np.where(fd_configs[:, 2] == scale)[0], -1])])

                        plt.plot(np.array(value_per_scale)[:, 0], np.array(value_per_scale)[:, 1], fold_markers[fold], label="{}-fold{}-mean-{:.3f}".format(method, fold, np.mean(np.array(value_per_scale)[:, 1])), color=meth_color[method])
            plt.legend(),
            plt.title("\n".join(wrap("{} with {} on {}".format(exp_mode, model, data_source), 60)))
            plt.savefig(os.path.join(data_dir, "new-{}-with-{}-on-{}.png".format(exp_mode, model, data_source))),
            plt.savefig(os.path.join(data_dir, "new-{}-with-{}-on-{}.pdf".format(exp_mode, model, data_source)), format="pdf")
            plt.close()
            
            # plot each fold and aug-method w.r.t augmentation factor alpha
            # plt.figure(figsize=[12, 8])
            # for res, method in zip([aug_same, aug_ops, aug_both], ["same", "ops", "both"]):
            #     for fold in [1,3,5,9]:
            #         fd_configs = np.array(res[np.where(res[:,1] == fold)[0]])
            #         if len(fd_configs) > 0:
            #             plot_vl = []
            #             for scale in [0.05, 0.2, 0.35, 0.5]:
            #                 print("{}, {}, {}".format(method, fold, scale))
            #                 if len(np.where(fd_configs[:,2] == scale)[0]) >= 1:
            #                     plot_vl.append([np.float(scale), np.mean(fd_configs[np.where(fd_configs[:,2] == scale)[0],-1])])
            #
            #             plt.plot(np.array(plot_vl)[:,0], np.array(plot_vl)[:,1], markers[fold], label="{}-fold{}-mean-{:.3f}".format(method, fold, np.mean(np.array(plot_vl)[:,1])), color=meth_color[method])
            # plt.legend(),
            # plt.title("\n".join(wrap("{} with {} on {}".format(exp_mode, model, data_source), 60)))
            # plt.savefig(os.path.join(data_dir, "{}-with-{}-on-{}-best-same-fold3-scale0.5-both.png".format(exp_mode, model, data_source))),
            # plt.savefig(os.path.join(data_dir, "{}-with-{}-on-{}-best-same-fold3-scale0.5-both.pdf".format(exp_mode, model, data_source)), format="pdf")
            # plt.close()
            #

            np.savetxt(os.path.join(data_dir, 'new-model_{}_all_different_config_theta{}-{}+DA-with-{}-on-{}.txt'.format(model, len(aug_same), exp_mode, model, data_source)), configs, header="aug_name,aug_fold,aug_factor,cer_th,test_auc", delimiter=",", fmt="%s")
    else:
        file_dirs = [
            "C:/Users/LDY/Desktop/1-all-experiment-results/metabolites/auc-func-as-augmentation-parameters/new-model_Res_ECG_CAM_all_different_config_theta39-DA+DA-with-Res_ECG_CAM-on-data5.txt",
            "C:/Users/LDY/Desktop/1-all-experiment-results/metabolites/auc-func-as-augmentation-parameters/new-model_Res_ECG_CAM_all_different_config_theta16-DA+DA-with-Res_ECG_CAM-on-data1.txt",
            "C:/Users/LDY/Desktop/1-all-experiment-results/metabolites/auc-func-as-augmentation-parameters/new-model_Res_ECG_CAM_all_different_config_theta16-DA+DA-with-Res_ECG_CAM-on-data2.txt",
            "C:/Users/LDY/Desktop/1-all-experiment-results/metabolites/auc-func-as-augmentation-parameters/new-model_Res_ECG_CAM_all_different_config_theta16-DA+DA-with-Res_ECG_CAM-on-data3.txt",
            "C:/Users/LDY/Desktop/1-all-experiment-results/metabolites/auc-func-as-augmentation-parameters/new-model_Res_ECG_CAM_all_different_config_theta16-DA+DA-with-Res_ECG_CAM-on-data4.txt",
            "C:/Users/LDY/Desktop/1-all-experiment-results/metabolites/auc-func-as-augmentation-parameters/new-model_Res_ECG_CAM_all_different_config_theta16-DA+DA-with-Res_ECG_CAM-on-data6.txt",
            "C:/Users/LDY/Desktop/1-all-experiment-results/metabolites/auc-func-as-augmentation-parameters/new-model_Res_ECG_CAM_all_different_config_theta16-DA+DA-with-Res_ECG_CAM-on-data7.txt",
            "C:/Users/LDY/Desktop/1-all-experiment-results/metabolites/auc-func-as-augmentation-parameters/new-model_Res_ECG_CAM_all_different_config_theta16-DA+DA-with-Res_ECG_CAM-on-data8.txt",
            "C:/Users/LDY/Desktop/1-all-experiment-results/metabolites/auc-func-as-augmentation-parameters/new-model_Res_ECG_CAM_all_different_config_theta16-DA+DA-with-Res_ECG_CAM-on-data9.txt"
        ]
        
        aug_name_encode = {"same": 0, "ops": 1,"both": 2}
        model_name = "Res_ECG_CAM"
        data_source = "all-CVs"  #os.path.basename(fn).split("-")[-1].split(".")[0]
        
        configs = np.empty((0, 5))
        for fn in file_dirs:
            load_data = pd.read_csv(fn, header=0).values
            configs = np.vstack((configs, load_data))
            #
            load_data = load_data[load_data[:, 0].argsort()]
            load_data = load_data[load_data[:, 1].argsort(kind='mergesort')]
            load_data = load_data[load_data[:, 2].argsort(kind='mergesort')]
            # np.savetxt(fn, np.array(load_data), delimiter=",", fmt="%.3f")
    
        configs = np.array(configs)
        aug_same = configs[np.where(configs[:, 0] == aug_name_encode["same"])[0]]
        aug_ops = configs[np.where(configs[:, 0] == aug_name_encode["ops"])[0]]
        aug_both = configs[np.where(configs[:, 0] == aug_name_encode["both"])[0]]
    
        scale_style = {0.05: "-", 0.2: "-.", 0.35: "--", 0.5: ":"}
        meth_color = {"other":"tab:orange", "same":"tab:green", "both":"tab:brown"}
        fold_markers = {1: "d", 3: "*", 5: "o", 9: "^"}
        styles = {1:":", 3:"-.", 5:"--", 9:"-"}
        
        # plot aug. method with boxplot and error bar
        plot_style = "imshow"   #"boxplot"  #
        if plot_style == "boxplot":
            plt.figure(figsize=[8, 5.5])
            for res, method, case in zip([aug_same, aug_ops, aug_both], ["same", "other", "both"], np.arange(3)):
                value_per_scale, names, xs = [], [], []
                for ind, scale in enumerate([0.05, 0.2, 0.35, 0.5]):
                    scale_configs = np.array(res[np.where(res[:,2] == scale)[0]])
                    vals = np.empty((0))
                    if len(scale_configs) > 0:
                        for jj, fold in enumerate([1,3,5,9]) :
                            fold_inds = np.where(scale_configs[:,1] == fold)[0]
                            fd_configs = np.array(scale_configs[fold_inds])
                            print("method_{}-scale_{}-fold_{} num {}".format(method, scale, fold, len(fold_inds)))
                            if len(fold_inds) >= 1:
                                plt.scatter(np.ones(len(fold_inds)) * ind * 4 + 1 + case, fd_configs[:,-1], color=meth_color[method], marker=fold_markers[fold], s=100)
                                vals = np.append(vals, fd_configs[:,-1])
                    value_per_scale.append(vals)
                    names.append(scale)
                    xs.append(np.random.normal(ind, 0.04, len(vals)))
                
                bp_positions = [jj*4+1+case for jj in range(4)]
                bplot = plt.boxplot(value_per_scale, labels=names, positions=bp_positions, widths = 0.85)
                
                for bpind in range(len(bplot["boxes"])):
                    plt.setp(bplot["boxes"][bpind], color=meth_color[method]),
                    plt.setp(bplot['caps'][bpind*2], color=meth_color[method]),
                    plt.setp(bplot['caps'][bpind*2+1], color=meth_color[method]),
                    plt.setp(bplot['whiskers'][bpind*2], color=meth_color[method]),
                    plt.setp(bplot['whiskers'][bpind*2+1], color=meth_color[method]),
                    plt.setp(bplot['fliers'][bpind], color=meth_color[method]),
                    plt.setp(bplot['medians'][bpind], color=meth_color[method])
                print("{} Done!".format(method))
                
            for fold in [1,3,5,9]:
                hide_pts = plt.scatter(1 + case, 0.6, color=meth_color[method], marker=fold_markers[fold], s=100, label="$\Phi$={}".format(fold))
                
            meth_color = {"other":"tab:orange", "same":"tab:green", "both":"tab:brown"}
            for jj, method in enumerate(["same", "other", "both"]):
                hide_line, = plt.plot([0.6,0.6], color=meth_color[method], label="aug-with-{}".format(method))
            plt.legend(scatterpoints=3, ncol=2, frameon=False)
            plt.xlabel(r"mixing weight $\alpha$")
            plt.ylabel("ROC-AUC")
            print("ok")
            
            plt.title("\n".join(wrap("{} with {} on {}".format(method, model_name, data_source), 60)))
            plt.savefig(os.path.join(os.path.dirname(fn), "{}-all-methods-in-one-{}-with-{}-on-{}.png".format(plot_style, method, model_name, data_source))),
            plt.savefig(os.path.join(os.path.dirname(fn), "{}-all-methods-in-one-{}-with-{}-on-{}.pdf".format(plot_style, method, model_name, data_source)), format="pdf")
            plt.close()
        elif plot_style == "imshow":
            fig, axes = plt.subplots(1, 3, figsize=(21, 7))
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            for res, method, md_case in zip([aug_same, aug_ops, aug_both], ["same", "other", "both"], np.arange(3)):
                # each one is for each scale case
                matrix_values = np.zeros((len(scale_style), len(fold_markers)))
                matrix_stds = np.zeros((len(scale_style), len(fold_markers)))
                temp_coll = np.empty((0, 5))
                for scl_ind, scale in enumerate([0.05, 0.2, 0.35, 0.5]):
                    scale_configs = np.array(res[np.where(res[:,2] == scale)[0]])
                    scale_values = []
                    for fd_ind, fold in enumerate([1,3,5,9]) :
                        fold_inds = np.where(scale_configs[:,1] == fold)[0]
                        fd_configs = np.array(scale_configs[fold_inds]).reshape(-1, 5)
                        print("method_{}-scale_{}-fold_{} num {}".format(method, scale, fold, len(fold_inds)))
                        matrix_values[fd_ind, scl_ind] = np.mean([fd_configs[:,-1]])
                        matrix_stds[fd_ind, scl_ind] = np.std([fd_configs[:,-1]])
                        temp_coll = np.vstack((temp_coll, fd_configs))
                    print("folds done")
                    

                im = axes[md_case].imshow(matrix_values, interpolation='none', vmin=matrix_values.min(), vmax=matrix_values.max(), aspect='equal', cmap="Blues")
                axes[md_case].set_xlabel(r"mixing weight $\alpha$")
                axes[md_case].set_ylabel(r"augmentation factor $\Phi$")
                divider = make_axes_locatable(axes[md_case])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                clb = fig.colorbar(im, cax=cax)   #, orientation='horizontal'
                clb.set_label('AUC', labelpad=-40, y=1.05, rotation=0)

                axes[md_case].set_xticks(np.arange(0, 4, 1), [0.05, 0.2, 0.35, 0.5]),
                axes[md_case].set_yticks(np.arange(0, 4, 1), [1,3,5,9])
                
                threshold = np.mean(matrix_values)
                for scl_ind in range(4):
                    for fd_inds in range(4):
                        color = "black" if matrix_values[scl_ind, fd_inds] < threshold else "white"
                        axes[md_case].text(fd_inds, scl_ind,r'${:.2f} \pm {:.2f}$'.format(matrix_values[scl_ind, fd_inds], matrix_stds[scl_ind, fd_inds]), color=color, horizontalalignment='center', fontsize=15)
                md = "other" if method=="ops" else method
                axes[md_case].set_title("\n".join(wrap("Aug-with-{}".format(md), 60)))
                print("oki")
                
                
            plt.tight_layout()
            plt.setp(axes, xticks=np.arange(0, 4, 1), xticklabels=[0.05, 0.2, 0.35, 0.5],
                    yticks=np.arange(0, 4, 1), yticklabels=[1,3,5,9])

            plt.savefig(os.path.join(os.path.dirname(fn), "{}-3-in-1-method_{}-in-one-with-{}-on-{}.png".format(plot_style, md, model_name, data_source)), bbox_inches='tight'),
            plt.savefig(os.path.join(os.path.dirname(fn), "{}-3-in-1-method_{}-in-one-with-{}-on-{}.pdf".format(plot_style, md, model_name, data_source)), format="pdf")
            plt.close()


elif plot_name == "rename_test_folders":
    print("Plot_name: ", plot_name)
    results = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN"
    folders = find_folderes(results, pattern="*-test")
    pattern = "accuracy_step_0.0_acc_*"
    for fn in folders:
        print(fn)
        test_result = find_files(fn, pattern=pattern)

        if len(test_result) >= 1:
            splits = os.path.basename(test_result[0]).split("_")
            new_name = os.path.basename(fn).replace("_", "-")
            auc = splits[-2]
            # os.rename(fn, os.path.join(os.path.dirname(fn), new_name))
            os.rename(fn, os.path.join(os.path.dirname(fn), new_name+"-{}".format(auc)))

        # new_name = os.path.basename(fn).replace("MLP", "Res_ECG_CAM")
        # os.rename(fn, os.path.join(os.path.dirname(fn), new_name))


elif plot_name == "rename_files":
    print("Plot_name: ", plot_name)
    results = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/9-train-with-MNIST-MLP/2021-02-10T08-18-38--MLP-both_meanx0-factor-0-from-mnist-certainFalse-theta-1-s5058-100rns-train/certains"
    filenames = find_files(results, pattern="*.csv")
    for fn in filenames:
        print(fn)
        new_name = os.path.basename(fn).replace("-", "_")
        os.rename(fn, os.path.join(os.path.dirname(fn), new_name))


elif plot_name == "get_performance_metrices":
    """
        Get overall performance metrices across different cross-validation sets
        :param pool_len:
        :param task_id:
        :return:
        """
    print("Plot_name: ", plot_name)
    postfix = ""
    data_dirs = ["/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-with-distillation-Inception"]
    for data_source, prefix in zip(["", "data5"], ["all-CVs","data5"]):
        for data_dir in data_dirs:
            for method in ["both"]: # "None","same", "both", "ops"
                for fold in [9, 5, 0, 1, 3]: #
                    for alpha in [0.5, 0, 0.05, 0.2, 0.35]:
                        folders = find_folderes(data_dir, pattern="*{}*x{}-factor-{}-from-{}*-test-0.*".format(method, fold, alpha, data_source))
                        if len(folders) > 0:
                            print(os.path.basename(data_dir),
                                  "{}x{}-{}-[{}]!".format(method, fold, alpha, [os.path.basename(fdn).split("-")[-3] for fdn in folders]))
                            performance = {"ACC": np.empty((0,)), "patient_ACC": np.empty((0,)),
                                           "AUC": np.empty((0,)), "SEN": np.empty((0,)),
                                           "SPE": np.empty((0,)), "F1_score": np.empty((0,)),
                                           "MCC": np.empty((0,))}
                            performance_summary = []
                            data_names = []
                            for fd in folders:
                                file = find_files(fd, pattern="AUC_curve_step*.csv")
                                num_patient = find_files(fd, pattern="*prob_distri_of*.png")
                                # rat_id = os.path.basename(fn).split("-")[-3]
                                data_names.append("{}-{}-{}\n".format(os.path.basename(fd).split("-")[-3], os.path.basename(fd).split("-")[-5], os.path.basename(fd).split("-")[-1]))
                                if len(file) > 0:
                                    values = pd.read_csv(file[0], header=0).values
                                    true_labels = values[:, 0]  # assign true-lbs and probs in aggregation
                                    pred_logits = values[:, 1]
                                    
                                    patient_acc = np.sum(["right" in name for name in num_patient]) / len(
                                        num_patient)
                                    # get summary of model performance's metrics
                                    accuracy, sensitivity, specificity, precision, F1_score, auc, fpr, tpr, mcc = get_scalar_performance_matrices_2classes(
                                        true_labels, pred_logits,
                                        if_with_logits=True)
                                    
                                    performance["ACC"] = np.append(performance["ACC"], accuracy)
                                    performance["SEN"] = np.append(performance["SEN"], sensitivity)
                                    performance["SPE"] = np.append(performance["SPE"], specificity)
                                    performance["AUC"] = np.append(performance["AUC"], auc)
                                    performance["F1_score"] = np.append(performance["F1_score"], F1_score)
                                    performance["MCC"] = np.append(performance["MCC"], mcc)
                                    performance["patient_ACC"] = np.append(performance["patient_ACC"],
                                                                           patient_acc)
                
                                    performance_summary.append(["{}-{}x{}-{}-data[{}]\n".format(os.path.basename(data_dir), method, fold, alpha, data_names),
                                                                "Sensitivity: mean-{:.3f}, std-{:.3f}\n".format(np.mean(performance["SEN"]), np.std(performance["SEN"]))
                                                                + "specificity: mean-{:.3f}, std-{:.3f}\n".format(np.mean(performance["SPE"]), np.std(performance["SPE"]))
                                                                +"AUC: mean-{:.3f}, std-{:.3f}\n".format(np.mean(performance["AUC"]),
                                                                                                                      np.std(performance["AUC"]))
                                                                + "patient acc: mean-{:.3f}, std-{:.3f}\n".format(np.mean(performance["patient_ACC"]),
                                                                                                                  np.std(performance["patient_ACC"]))
                                                                +"F1-score: mean-{:.3f}, std-{:.3f}\n".format(np.mean(performance["F1_score"]),
                                                                                                                           np.std(performance["F1_score"]))
                                                                +"MCC: mean-{:.3f}, std-{:.3f}\n".format(np.mean(performance["MCC"]),
                                                                                                                      np.std(performance["MCC"]))
                                                                +"ACC: mean-{:.3f}, std-{:.3f}\n".format(np.mean(performance["ACC"]),
                                                                                                                      np.std(performance["ACC"]))
                                                                
                                                                ])
                            np.savetxt(os.path.join(data_dir, "{}-AUC-{:.4f}-performance-summarries-of-{}x{}-{}-num{}-CVs.csv".format(prefix, np.mean(performance["AUC"]), method, fold, alpha, len(folders))), np.array(performance_summary), fmt="%s", delimiter=",")
            
                            print("{}-{}x{}-{}-data[{}]\n".format(os.path.basename(data_dir), method, fold, alpha, data_names))
                            print("Sensitivity: mean-{:.3f}, std-{:.3f}\n".format(np.mean(performance["SEN"]),
                                                                          np.std(performance["SEN"])))
                            print("specificity: mean-{:.3f}, std-{:.3f}\n".format(np.mean(performance["SPE"]),
                                                                          np.std(performance["SPE"])))
                            print("AUC: mean-{:.3f}, std-{:.3f}\n".format(np.mean(performance["AUC"]),
                                                                  np.std(performance["AUC"])))
                            print("patient acc: mean-{:.3f}, std-{:.3f}\n".format(np.mean(performance["patient_ACC"]),
                                                                          np.std(performance["patient_ACC"])))
                            print("F1-score: mean-{:.3f}, std-{:.3f}\n".format(np.mean(performance["F1_score"]),
                                                                       np.std(performance["F1_score"])))
                            print("MCC: mean-{:.3f}, std-{:.3f}\n".format(np.mean(performance["MCC"]),
                                                                  np.std(performance["MCC"])))
                            print("ACC: mean-{:.3f}, std-{:.3f}\n".format(np.mean(performance["ACC"]),
                                                                  np.std(performance["ACC"])))
                        else:
                            print(os.path.basename(data_dir), "{}x{}-{}-No data!".format(method, fold, alpha))
        

elif plot_name == "move_folder":
    import shutil

    print("Plot_name: ", plot_name)

    dirs = [
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA2-Res_ECG_CAM",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Inception",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-RandomDA-MLP",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-RNN",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA+noise-Res_ECG_CAM",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-Inception",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-MLP",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-Res_ECG_CAM",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-Res7-Res_ECG_CAM",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN"
    ]
    # dirs = ["C:/Users/LDY/Desktop/metabolites-0301/metabolites_tumour_classifier/results/1-Pure-new-Inception"]

    for dd in dirs:
        sub_folders = find_folderes(dd, pattern=os.path.basename(dd)+"-data*")
        for sub_fd in sub_folders:
            # test_folders = find_folderes(fd, pattern="*-test-0.*")
            # os.rename(fd, fd+"-len{}".format(len(test_folders)))
            test_folders = find_folderes(sub_fd, pattern="*-train")
            for t_fd in test_folders:
                print("sub folders", test_folders)
                data_cv = os.path.basename(t_fd).split("-")[-3]
                new_dest_root = dd
    
                if not os.path.isdir(new_dest_root):
                    os.mkdir(new_dest_root)
                else:
                    print("Move {} to {}".format(os.path.basename(t_fd),
                                                 os.path.join(new_dest_root, os.path.basename(t_fd))))
                    shutil.move(t_fd, os.path.join(new_dest_root, os.path.basename(t_fd)))
                
            
elif plot_name == "generate_empty_folders":
    print("Plot_name: ", plot_name)
    dirs = [
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Res_ECG_CAM",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA2-Res_ECG_CAM",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA+noise-Res_ECG_CAM",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-Res_ECG_CAM",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-Res7-Res_ECG_CAM",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-Inception",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-RNN",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-RandomDA-MLP",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Inception",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-RNN",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-MLP",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-new-Inception",
    ]

    for fd in dirs:
        for jj in range(10):
            new_dirs = os.path.join(fd, os.path.basename(fd)+"-data{}".format(jj))
            print("Make dir ", new_dirs)
            os.mkdir(new_dirs)


elif plot_name == "certain_tsne_distillation":
    print("Plot_name: ", plot_name)
    from scipy.io import loadmat as loadmat
    import scipy.io as io
    from scipy.stats import ks_2samp

    pattern = "full_summary-*.csv"
    data_dir = "C:\\Users\\LDY\\Desktop\\metabolites-0301\\metabolites_tumour_classifier\\data"
    # data_dir = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T22-07-37--Res_ECG_CAM-nonex0-factor-0-from-data5-certainFalse-theta-0-s989-100rns-train"
    # ori_data_dir = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/DATA.mat"
    ori_data_dir = "C:\\Users\\LDY\\Desktop\\metabolites-0301\\metabolites_tumour_classifier\\data\\2019.03.25-DATA.mat"
    whole_data = get_data_from_mat(ori_data_dir)   # sample_id, pat_id, lb, features
    feature_start_id = 3
    data_source = "whole"
    # data_source = data_dir.split("-")[-7]
    reduction_method = "PCA"
    if_save_data = True
    # data_dir = os.path.dirname(ori_data_dir)
    if_get_distilldata = False
    ## get the whole tsne projection
    if reduction_method == "tsne":
        if if_save_data:
            from bhtsne import tsne as TSNE
            reduced_proj_whole = TSNE(whole_data[:, feature_start_id:], dimensions=2)
            np.savetxt(os.path.join(data_dir, "{}-whole_data-2d.csv".format(reduction_method)), reduced_proj_whole, fmt="%.5f", delimiter=","),
            if if_get_distilldata:
                reduced_proj_distill= TSNE(distill_data[:, feature_start_id:], dimensions=2)
                np.savetxt(os.path.join(data_dir, "{}-distill_data-2d.csv".format(reduction_method)), reduced_proj_distill, fmt="%.5f", delimiter=",")
        else:
            filename_whole = os.path.join(data_dir, "{}-whole_data-lout5-2d.csv".format(reduction_method))
            reduced_proj_whole = pd.read_csv(filename_whole, header=None).values
            # filename_distill = os.path.join(data_dir, "distill_data_tsne-2d-from-whole-tsne.csv.csv".format(reduction_method))
            # reduced_proj_distill = pd.read_csv(filename_distill, header=None).values
    elif reduction_method == "UMAP":
        if if_save_data:
            import umap.umap_ as umap
            reduced_proj_whole = umap.UMAP(random_state=42).fit_transform(whole_data[:, feature_start_id:])
            np.savetxt(os.path.join(data_dir, "{}-whole_data-2d.csv".format(reduction_method)), reduced_proj_whole, fmt="%.5f", delimiter=","),
            if if_get_distilldata:
                reduced_proj_distill = umap.UMAP(random_state=42).fit_transform(distill_data[:, feature_start_id:])
                np.savetxt(os.path.join(data_dir, "{}-distill_data-2d.csv".format(reduction_method)), reduced_proj_distill, fmt="%.5f", delimiter=",")
        else:
            filename_whole = os.path.join(data_dir, "{}-whole_data-2d.csv".format(reduction_method))
            reduced_proj_whole = pd.read_csv(filename_whole, header=None).values
            filename_distill = os.path.join(data_dir, "{}-distill_data-2d-from-whole.csv".format(reduction_method))
            reduced_proj_distill = pd.read_csv(filename_distill, header=None).values
    elif reduction_method == "MDS":
        if if_save_data:
            from sklearn.manifold import MDS
            MDS_whole = MDS(n_components=2, random_state=199)
            MDS_distill = MDS(n_components=2, random_state=199)
            reduced_proj_whole = MDS_whole.fit_transform(whole_data[:, feature_start_id:])
            if if_get_distilldata:
                reduced_proj_distill = MDS_distill.fit_transform(distill_data[:, feature_start_id:])
                np.savetxt(os.path.join(data_dir, "{}-distill_data-2d-from-whole.csv".format(reduction_method)), reduced_proj_distill, fmt="%.5f", delimiter=",")
            np.savetxt(os.path.join(data_dir, "{}-whole_data-2d.csv".format(reduction_method)), reduced_proj_whole, fmt="%.5f", delimiter=","),
        else:
            filename_whole = os.path.join(data_dir, "{}-whole_data-2d.csv".format(reduction_method))
            filename_distill = os.path.join(data_dir, "{}-distill_data-2d-from-whole.csv".format(reduction_method))
            reduced_proj_whole = pd.read_csv(filename_whole, header=None).values
            reduced_proj_distill = pd.read_csv(filename_distill, header=None).values
    elif reduction_method == "PCA":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced_proj_whole = pca.fit_transform(whole_data[:, feature_start_id:].astype('float64'))
        

    # ori_colors = ["c", "violet"]
    ori_colors = ["c", "m"]
    # distill_colors = ["darkblue", "crimson"]
    distill_colors = ["c", "m"]

    # plot the whole set
    fig = plt.figure(figsize=[8, 6.5])
    ax = fig.add_subplot(111)
    for c in range(2):
        inds = np.where(whole_data[:, 2] == c)[0]
        im = ax.scatter(reduced_proj_whole[inds, 0], reduced_proj_whole[inds, 1], color=ori_colors[c], alpha=0.35, s=15, facecolor=None, label="original class {}".format(c))
    inds0 = np.where(whole_data[:, 2] == 0)[0]
    inds1 = np.where(whole_data[:, 2] == 1)[0]
    _, p_x_whole = ks_2samp(reduced_proj_whole[inds0, 0], reduced_proj_whole[inds1, 0])
    _, p_y_whole = ks_2samp(reduced_proj_whole[inds0, 1], reduced_proj_whole[inds1, 1])
    plt.legend(scatterpoints=4)
    # plt.title("{} of original data ()".format(reduction_method))
    # plt.xlabel("dimension #1 (p={:.2E})".format(p_x_whole)),
    # plt.ylabel("dimension #2 (p={:.2E})".format(p_y_whole))
    plt.xlabel("dimension #1")
    plt.ylabel("dimension #2")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "{}-whole-{}.png".format(reduction_method, data_source))),
    plt.savefig(os.path.join(data_dir, "{}-whole-{}.pdf".format(reduction_method, data_source)), format="pdf")
    plt.close()

    # plot the whole set w.r.t patients
    new_pat_ids = np.empty((0))
    new_order = np.empty((0))
    for pat in np.unique(whole_data[:, 1]):
        pat_inds = np.where(whole_data[:, 1] == pat)[0]
        temp = whole_data[pat_inds, 1]
        label = np.mean(whole_data[pat_inds, 2])
        if label == 1:
            temp = temp + 5000
        new_pat_ids = np.append(new_pat_ids, temp)
        new_order = np.append(new_order, pat_inds).astype(np.int)
    plt.figure(figsize=[10, 7])
    # plt.scatter(reduced_proj_whole[:, 0], reduced_proj_whole[:, 1], c=new_pat_ids.astype(np.int), s=15, cmap="jet", facecolor=None)
    plt.scatter(reduced_proj_whole[new_order, 0], reduced_proj_whole[new_order, 1], c=new_pat_ids, s=15, cmap="jet", facecolor=None),
    plt.colorbar(),
    plt.title("\n".join(wrap("Grouped by the patients", 40))),
    plt.xlabel("dimension #1"),
    plt.ylabel("dimension #2")
    plt.savefig(os.path.join(data_dir, "grouped-by-samples-{}-whole-{}.png".format(reduction_method, data_source))),
    plt.savefig(os.path.join(data_dir, "grouped-by-samples-{}-whole-{}.pdf".format(reduction_method, data_source)), format="pdf")
    plt.close()

    # plot the distill
    fig = plt.figure(figsize=[8, 6.5])
    dis_inds0 = np.where(distill_data[:, 2] == 0)[0]
    dis_inds1 = np.where(distill_data[:, 2] == 1)[0]
    _, p_x_dis = ks_2samp(reduced_proj_distill[dis_inds0, 0], reduced_proj_distill[dis_inds1, 0])
    _, p_x_dis = ks_2samp(reduced_proj_distill[dis_inds0, 1], reduced_proj_distill[dis_inds1, 1])
    ax = fig.add_subplot(111)
    for c in range(2):
        inds = np.where(distill_data[:, 2] == c)[0]
        im = ax.scatter(reduced_proj_distill[inds, 0], reduced_proj_distill[inds, 1], color=distill_colors[c], alpha=0.35, s=15, facecolor=None, label="distill class {}".format(c))
    plt.legend(scatterpoints=4, loc=3)
    plt.title("TSNE of both classes (distilled)")
    plt.xlabel("dimension #1 (p={:.2E})".format(p_x_dis)),
    plt.ylabel("dimension #2 (p={:.2E})".format(p_x_dis))
    plt.savefig(os.path.join(data_dir, "umap_visualization", "tsne-distill-{}.png".format(data_source)))
    plt.savefig(os.path.join(data_dir, "umap_visualization", "tsne-distill-{}.pdf".format(data_source)), format="pdf")
    plt.savefig(os.path.join(data_dir, "Distilled tumor samples-{}-from-{}.png".format(reduction_method, data_source))),
    plt.savefig(os.path.join(data_dir, "Distilled tumor samples-{}-from-{}.pdf".format(reduction_method, data_source)), format="pdf")
    plt.close()

    # plot the same number as in distill from the whole set
    fig = plt.figure(figsize=[8, 6.5])
    ax = fig.add_subplot(111)
    for c in range(2):
        inds = np.where(whole_data[:, 2] == c)[0]
        if c == 0:
            need_number = len(dis_inds0)
        else:
            need_number = len(dis_inds1)
        sub_set = np.random.choice(inds, need_number, replace=False)
        im = ax.scatter(reduced_proj_whole[sub_set, 0], reduced_proj_whole[sub_set, 1], color=ori_colors[c], alpha=0.35, facecolor=None, s=15, label="original class {}".format(c))
    inds0 = np.where(whole_data[:, 2] == 0)[0]
    inds1 = np.where(whole_data[:, 2] == 1)[0]
    _, p_x_whole = ks_2samp(reduced_proj_whole[inds0, 0], reduced_proj_whole[inds1, 0])
    _, p_y_whole = ks_2samp(reduced_proj_whole[inds0, 1], reduced_proj_whole[inds1, 1])
    plt.legend(scatterpoints=4, loc=3)
    plt.title("\n".join(wrap("{} of both classes (random sampled from the whole)".format(reduction_method), 60)))
    plt.xlabel("dimension #1 (p={:.2E})".format(p_x_whole)),
    plt.ylabel("dimension #2 (p={:.2E})".format(p_y_whole))
    plt.savefig(os.path.join(data_dir, "umap_visualization", "{}-subset-the-same-number-as-distill-from-whole-{}.png".format(reduction_method, data_source))),
    plt.savefig(os.path.join(data_dir, "umap_visualization", "{}-subset-the-same-number-as-distill-from-whole-{}.pdf".format(reduction_method, data_source)),
                format="pdf")
    plt.close()

    # plt.figure(figsize=[8, 6.5])
    # plt.scatter(tsne_distill[:, 0], tsne_distill[:, 1], c=distill_data[:,1], cmap="jet", facecolor=None)
    # plt.colorbar()
    # plt.title("Patients from the distilled set  (healthy<1000, tumor>1000)")
    # plt.xlabel("dimension #1"),
    # plt.ylabel("dimension #2")
    # plt.savefig(os.path.join(data_dir, "grouped-by-patients-{}-distill-{}.png".format(reduction_method, data_source))),
    # plt.savefig(os.path.join(data_dir, "grouped-by-patients-{}-distill-{}.pdf".format(reduction_method, data_source)), format="pdf")
    # plt.close()



    ## overlay the certain ones' tsne


elif plot_name == "plot_metabolites_statistics":
    print("Plot_name: ", plot_name)
    from scipy.io import loadmat as loadmat
    import scipy.io as io

    data_dir = r"C:\Users\LDY\Desktop\metabolites-0301\metabolites_tumour_classifier\data\20190325"
    file_patterns = "*.csv"

    # mat_file = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/DATA.mat"
    mat_file = "C:\\Users\\LDY\\Desktop\\metabolites-0301\\metabolites_tumour_classifier\\data\\2019.03.25-DATA.mat"
    
    original_data = get_data_from_mat(mat_file)
    
    ####################################
    # get num of spectra distribution among patients
    stat_num_per_pat = Counter(original_data[:, 1])
    stat_num_per_pat.items()
    num_spectra = np.array([vl for _, vl in stat_num_per_pat.items()])
    plt.hist(num_spectra, bins=100)
    # plt.title("Distribution of number of spectra in patients")
    plt.xlabel("number of voxels per patient")
    plt.ylabel("number of patients")
    # plt.vlines(np.percentile(num_spectra, 90), 0, 40, label="90th")
    # plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(mat_file), "Distribution-of-number-of-spectra-in-patients-whole2.png")),
    plt.savefig(os.path.join(os.path.dirname(mat_file), "Distribution-of-number-of-spectra-in-patients-whole2.pdf"))
    plt.close()

    ###################################################################
    # plot the spectra of all patients. One plot one patient
    # for pat_id, num in stat_num_per_pat.items():
    #     plt.figure()
    #     pat_inds = np.where(original_data[:, 1] == pat_id)[0]
    #     assert num == len(pat_inds)
    #     label = np.mean(original_data[pat_inds, 2])
    #     plt.plot(original_data[pat_inds, 3:].T)
    #     plt.title("Patient({})-lb{}-num({})".format(pat_id, label, num))
    #     plt.xlabel("metabolite index")
    #     plt.ylabel("norm. amplitude")
    #     plt.savefig(os.path.join(os.path.dirname(mat_file), "plot-spectra-patient-wise", "Spectra-Patient-{}-lb{}-({}).png".format(pat_id, label, num))),
    #     plt.savefig(os.path.join(os.path.dirname(mat_file), "plot-spectra-patient-wise", "Spectra-Patient-{}-lb{}-({}).pdf".format(pat_id, label, num))),
    #     plt.close()
    
    labels = original_data[:, 1]

    train_data = {}
    test_data = {}
    true_lables = original_data[:, 0].astype(np.int)

    need_inds = np.where(true_lables == 6)[0]
    need_spec = original_data[need_inds, 2:]

    pre_id = np.int(original_data[0, 0])
    pre_lb = [np.int(original_data[0, 1])]
    pat_ids_lb = []  #(pre_id, pre_lb, 0)id, all labels, unique labels
    pat_count = 0
    for iid, lb in zip(original_data[1:, 0], original_data[1:, 1]):
        if iid == pre_id:
            pre_lb.append(np.int(lb))
        else:
            pre_lb = np.array(pre_lb).astype(np.int)
            pat_ids_lb.append(["patient ID: {}".format(np.int(pre_id)), "labels: {}".format(Counter(pre_lb))])
            pre_id = np.int(iid)
            pre_lb = [np.int(lb)]
    print("ok")

    class_names = ["healthy", "tumor"]
    class_colors = ["lightblue", "violet"]
    # class_dark = ["darkblue", "crimson"]
    class_dark = ["c", "m"]
    
    ###################################################################
    # plot PCA of original data
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_emb = pca.fit(original_data[:, 2:].astype('float64'))

    files = find_files(data_dir, pattern=file_patterns)
    # certain_mat = np.empty((0, new_mat.shape[1]))
    certain_inds_tot = np.empty((0))
    for fn in files:
        certain = pd.read_csv(fn, header=0).values
        certain_inds = certain[:, 0].astype(np.int)
        certain_inds_tot = np.append(certain_inds_tot, certain_inds)
        print(os.path.basename(fn), len(certain_inds), "samples\n")

    uniq_inds = np.unique(certain_inds_tot).astype(np.int)
    certain_mat = original_data[uniq_inds]

    ###################################################################
    # plot the certain samples' PCA
    # np.savetxt(os.path.join(data_dir, "certain_samples_lout40_fold5[smp_id,pat_id,label,meta]_class(0-1)=({}-{}).csv".format(len(np.where(certain_mat[:,2]==0)[0]), len(np.where(certain_mat[:,2]==1)[0]))), certain_mat, delimiter=",", fmt="%.5f")
    #
    # for c in range(2):
    #     inds = np.where(certain_mat[:,2]==c)[0]
    #
    #     samples = certain_mat[inds]
    #
    #     plt.plot(samples[:, 3:].T, class_colors[c])
    #     plt.plot(np.mean(samples[:, 3:], axis=0), class_dark[c], lw=3.5, label="{}-mean".format(class_names[c])),
    #     plt.legend()
    #     plt.xlabel("sample index"),
    #     plt.ylabel("normalized amp.")
    #     plt.title("Certain samples from class {}".format(class_names[c]))
    #     plt.savefig(os.path.join(data_dir, "certain_samples_class{}.png".format(c)))
    #     plt.close()

    for c in range(2):
        inds = np.where(certain_mat[:, 2] == c)[0]
        rand_inds = np.random.choice(inds, 100, replace=False)
        rand_samps = certain_mat[rand_inds]
        for ii in range(3):
            plot_smps = rand_samps[ii * 30:(ii + 1) * 30]
            f, axs = plt.subplots(6, 5, sharex=True)
            plt.suptitle("Certain samples from class {}".format(class_names[c]),
                         x=0.5,
                         y=0.98)
            for j in range(6 * 5):
                axs[j // 5, np.mod(j, 5)].plot(plot_smps[j, 3:], class_dark[c])
                plt.setp(axs[j // 5, np.mod(j, 5)].get_yticklabels(), visible=False)

            f.text(0.5, 0.05, 'index'),
            f.text(0.02, 0.5, 'Normalized amplitude', rotation=90,
                   verticalalignment='center'),
            f.subplots_adjust(hspace=0, wspace=0)
            plt.savefig(
                os.path.join(data_dir, "certain_samples_class{}_fig_{}.png".format(c, ii)))
            plt.close()


elif plot_name == "100_single_ep_corr_classification_rate_with_certain":
    print("Plot_name: ", plot_name)
    """
    Get the correct classification rate with 100 runs of single-epoch-training
    """
    import ipdb
    from scipy.stats import spearmanr
    
    data_dirs = [
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-05T13-54-31-MLP-nonex0-factor-0-from-ep-0-from-lout40-data7-theta-None-s129-100rns-train/certains",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-05T13-54-30-MLP-nonex0-factor-0-from-ep-0-from-lout40-data5-theta-None-s129-100rns-train/certains",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-05T13-54-29-MLP-nonex0-factor-0-from-ep-0-from-lout40-data3-theta-None-s129-100rns-train",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-05T13-54-28-MLP-nonex0-factor-0-from-ep-0-from-lout40-data1-theta-None-s129-100rns-train",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-05T13-54-32-MLP-nonex0-factor-0-from-ep-0-from-lout40-data9-theta-None-s129-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-09T21-42-58-MLP-nonex0-factor-0-from-ep-0-from-lout40-MNIST-theta-None-s129-100rns-train/certains"
    ]
    num_smp_dataset = {"data0": 8357, "data1": 8326, "data2": 8566,
                         "data3": 8454, "data4": 8440, "data5": 8231,
                         "data6": 8371, "data7": 8357, "data8": 8384, "data9": 7701,
                       "mnist":70000}
    
    for data_dir in data_dirs:
        files = find_files(data_dir, pattern="one_ep_data_train*.csv")
    
        # get correct count in 100 rauns
        data_source = os.path.basename(files[0]).split("_")[8]
        for theta in [90]: #, 92.5, 95, 97.5, 99
            print(data_source, "theta:", theta)
            for ind, fn in enumerate(files[:10]):
                values = pd.read_csv(fn, header=0).values
                smp_ids = values[:, 0].astype(np.int)
                true_lables = values[:, 1].astype(np.int)
                noisy_lbs = values[:, 2]
                prob = values[:, 3:]
                if ind == 0:  #the first file to get the total number (3-class) of samples
                    total_num = num_smp_dataset[data_source]  #3-class samples id
                    correct_ids_w_count = []
                    certain_w_count = []
                    certain_w_corr_count = []
                    correct_dict_id_w_count = {key: 0 for key in np.arange(total_num)}  #total number 9243
                    dict_count_certain = {key: 0 for key in np.arange(total_num)}
                    dict_corr_count_certain = {key: 0 for key in np.arange(total_num)}
                    
                pred_lbs = np.argmax(prob, axis=1)
                right_inds = np.where(pred_lbs == noisy_lbs)[0]
                correct_sample_ids = np.unique(smp_ids[right_inds])
                correct_ids_w_count += list(correct_sample_ids)
                
                # Get certain with differnt threshold
                if theta > 1:  #percentile
                    larger_prob = [pp.max() for pp in prob]
                    threshold = np.percentile(larger_prob, theta)
                    slc_ratio = 1 - theta / 100.
                else:  # absolute prob. threshold
                    threshold = theta
                    slc_ratio = 1 - theta
                ct_smp_ids = np.where([prob[i] > threshold for i in range(len(prob))])[0]
                ct_corr_inds = ct_smp_ids[np.where(noisy_lbs[ct_smp_ids] == pred_lbs[ct_smp_ids])[0]]
                certain_w_count += list(np.unique(smp_ids[ct_smp_ids]))
                certain_w_corr_count += list(np.unique(smp_ids[ct_corr_inds]))
                num_certain = len(ct_smp_ids)
    
            correct_id_count_all = Counter(correct_ids_w_count)
            correct_dict_id_w_count.update(correct_id_count_all)
    
            dict_count_certain.update(Counter(certain_w_count))
            dict_corr_count_certain.update(Counter(certain_w_corr_count))
            
            # if theta == 0.975:
            #     ipdb.set_trace()
            counter_array = np.array([[key, val] for (key, val) in correct_dict_id_w_count.items()])
            sort_inds = np.argsort(counter_array[:, 1])
            sample_ids_key = counter_array[sort_inds, 0]
            # rates = counter_array[sort_inds, 1]/counter_array[:, 1].max()
            rates = counter_array[sort_inds, 1]/len(files)

            ct_counter_array = np.array([[key, val] for (key, val) in dict_count_certain.items()])
            ct_counter_array_corr = np.array([[key, val] for (key, val) in dict_corr_count_certain.items()])
            
            
            ct_sele_rates = ct_counter_array[sort_inds, 1] / len(files)
            ct_corr_rates = ct_counter_array_corr[sort_inds, 1] / len(files)
            # rates_certain_corr = counter_array_certain_corr[sort_inds, 1] / counter_array_certain_corr[:, 1].max()
            # sort_samp_ids_certain = ct_counter_array[sort_inds, 0]
    
            fig, ax1 = plt.subplots()
            ax1.set_xlabel("sample index (sorted)")
            ax1.set_ylabel("rate over 100 runs")
            ax1.plot(rates, label="correct clf. rate")
            ax1.tick_params(axis='y')
            ax1.set_ylim([0,1.0])
            ax1.plot(np.arange(len(ct_sele_rates)), ct_sele_rates, label="distilled selection rate")
            ax1.plot(np.arange(len(ct_corr_rates)), ct_corr_rates, label="distilled corr. rate", color='m')
            plt.legend()
            plt.title("\n".join(wrap("distillation effect-(theta-{})-{}.png".format(theta, data_source), 60)))
            plt.savefig(data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}-({}_theta-{}).png".format(os.path.basename(files[0]).split("_")[7], total_num, data_source, num_certain, theta)),
            plt.savefig(data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}-({}_theta-{}).pdf".format(os.path.basename(files[0]).split("_")[7], total_num, data_source, num_certain, theta), format="pdf")
            print("ok")
            plt.close()
            
            num2select = np.int(np.int(os.path.basename(files[0]).split("_")[7]) * slc_ratio)
            ct_concat_data = np.concatenate((np.array(sort_inds).reshape(-1,1)[-num2select:], rates.reshape(-1,1)[-num2select:], ct_sele_rates.reshape(-1, 1)[-num2select:], ct_corr_rates.reshape(-1, 1)[-num2select:]), axis=1)
            np.savetxt(data_dir + "/certain_{}_({}-{})-({}_theta-{}).csv".format(data_source, os.path.basename(files[0]).split("_")[7], total_num, num2select, theta), ct_concat_data, fmt="%.5f", delimiter=",", header="ori_sort_rate_id,ori_sort_rate,certain_sele_rate,certain_corr_rate")
    concat_data = np.concatenate((np.array(sort_inds).reshape(-1,1), rates.reshape(-1,1), ct_sele_rates.reshape(-1, 1), ct_corr_rates.reshape(-1, 1)), axis=1)
    np.savetxt(data_dir + "/full_summary-{}_100_runs_sort_inds_rate_({}-{}).csv".format(data_source, os.path.basename(files[0]).split("_")[7], total_num), concat_data, fmt="%.5f", delimiter=",", header="ori_sort_rate_id,ori_sort_rate,certain_sele_rate,certain_corr_rate")


elif plot_name == "100_single_ep_corr_classification_rate_mnist":
    """
    Get the correct classification rate with 100 runs of single-epoch-training
    """
    import ipdb
    from scipy.stats import spearmanr
    
    print("Plot_name: ", plot_name)
    original_data_dirs = [
        "/home/epilepsy-data/data/metabolites/noisy-MNIST/0.2_noisy_train_val_mnist_[samp_id,true,noise]-s5058.csv"
    ]
    data_dirs = [
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/9-train-with-MNIST-MLP/2021-02-13T07-08-29--MLP-both_meanx0-factor-0-from-mnist-ctFalse-theta-1-s8522-100rns-train-trainOnTrue"
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/9-train-with-MNIST-MLP/2021-01-27T13-14-34--MLP-both_meanx3-factor-0.5-from-mnist-certainFalse-theta-1-s5506-0.5-noise-100rns-train-with"
        # r"C:\Users\LDY\Desktop\EPG\PPS-EEG-anomaly"
    ]
    num_smp_dataset = {"data0": 8357, "data1": 8326, "data2": 8566,
                       "data3": 8454, "data4": 8440, "data5": 8231,
                       "data6": 8371, "data7": 8357, "data8": 8384, "data9": 7701,
                       "mnist": 60000}

    for data_dir, original in zip(data_dirs, original_data_dirs):
        files = find_files(data_dir, pattern="one*.csv")

        spearmanr_rec = []
        # get correct count in 100 rauns
        data_source = os.path.basename(files[0]).split("_")[-4]
        for ind in tqdm(range(len(files))):
            # fn = find_files(data_dir, pattern="one_ep_data_train_epoch_{}*.csv".format(ind))
            values = pd.read_csv(files[ind], header=0).values
            smp_ids = values[:, 0].astype(np.int)
            true_lables = values[:, 1].astype(np.int)  # true labels
            noisy_lbs = values[:, 2]  # noisy labels
            prob = values[:, 3:]
            if ind == 0:  # the first file to get the total number (3-class) of samples
                total_num = num_smp_dataset[data_source]  # 3-class samples id
                correct_ids_w_count = []
                noisy_lb_ids_w_count = []
                correct_dict_id_w_count = {key: 0 for key in np.arange(total_num)}  # total number 9243
                noisy_lb_rec = {key: 0 for key in np.arange(total_num)}  # total number 9243
                pre_rank = np.arange(total_num)  #indices

            pred_lbs = np.argmax(prob, axis=1)
            right_inds = np.where(pred_lbs == true_lables)[0]
            correct_sample_ids = np.unique(smp_ids[right_inds])
            correct_ids_w_count += list(correct_sample_ids)

            noisy_lb_inds = np.where(true_lables != noisy_lbs)[0]
            noisy_lb_sample_ids = np.unique(smp_ids[noisy_lb_inds])  # sample ids that with noisy labels
            noisy_lb_ids_w_count += list(noisy_lb_sample_ids)  # sample ids that with noisy labels

            # if ind % 10 == 0:
            #     count_all = Counter(ids_w_count)
            #     dict_count.update(count_all)
            #     curr_count_array = np.array([[key, val] for (key, val) in dict_count.items()])
            #     curr_rank = curr_count_array[np.argsort(curr_count_array[:, 1]),0]
            #     # spearmanr_rec.append([ind, np.sum(curr_rank==pre_rank)])
            #     spearmanr_rec.append([ind, spearmanr(pre_rank, curr_rank)[0]])
            #     pre_rank = curr_rank.copy()

        correct_id_count_all = Counter(correct_ids_w_count)
        correct_dict_id_w_count.update(correct_id_count_all)
        noisy_lb_rec.update(Counter(noisy_lb_ids_w_count))
        
        counter_array = np.array([[key, val] for (key, val) in correct_dict_id_w_count.items()])
        noisy_lb_rec_array = np.array([[key, val] for (key, val) in noisy_lb_rec.items()])
        noisy_inds_array = np.array([[key, val*1.0/len(files)] for (key, val) in noisy_lb_rec.items()])
        sort_inds = np.argsort(counter_array[:, 1])
        sample_ids_key = counter_array[sort_inds, 0]
        # rates = counter_array[sort_inds, 1]/counter_array[:, 1].max()
        rates = counter_array[sort_inds, 1] / len(files)
        noisy_lb_rate = noisy_inds_array[sort_inds, 1]
    
        assert np.sum(counter_array[sort_inds, 0] == noisy_inds_array[sort_inds, 0]), "sorted sample indices mismatch"
    
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("sample sorted by the correct clf. rate"),
        ax1.set_ylabel("correct clf. rate (over 100 runs)"),
        ax1.plot(rates, label="original data set"),
        ax1.tick_params(axis='y'),
        ax1.set_ylim([0, 1.0])
        ax1.legend(loc="upper left")
    
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('counts'),  # we already handled the x-label with ax1
        ax2.plot(noisy_lb_rate.cumsum(), "m", label="accum. # of noisy labels"),
        ax2.plot(np.ones(total_num).cumsum(), "c", label="accum. # of all samples")
        ax2.set_ylim([0, total_num])
        ax2.tick_params(axis='y')
        ax2.legend(loc="upper right")
        plt.title("distillation effect-{}.png".format(data_source))
        plt.savefig(
            data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}.png".format(
                os.path.basename(files[0]).split("_")[-6], total_num, data_source)),
        plt.savefig(
            data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}.pdf".format(
                os.path.basename(files[0]).split("_")[-6], total_num, data_source), format="pdf")
        print("ok")
        plt.close()
        

        original_data = pd.read_csv(original, header=None).values
        ordered_data_w_lbs = original_data[sort_inds]
        concat_data = np.concatenate((
                                     np.array(sort_inds).reshape(-1, 1),
                                     rates.reshape(-1, 1)), axis=1)
        np.savetxt(data_dir + "/full_summary-{}_100_runs_sort_inds_rate_({}-{}).csv".format(data_source, os.path.basename(
            files[0]).split("_")[7], total_num), concat_data, fmt="%.5f", delimiter=",",
                   header="ori_id,sort_rate")
        
        concat_2 = np.concatenate((np.array(sort_inds).reshape(-1, 1),
                                   ordered_data_w_lbs,
                                   rates.reshape(-1, 1)), axis=1)
        np.savetxt(
            data_dir + "/full_summary-{}_100_runs_sort_inds_rate_({}-{}).csv".format(
                data_source, os.path.basename(
                    files[0]).split("_")[7], total_num), concat_data, fmt="%.5f",
            delimiter=",",
            header="ori_id,sort_rate")
        

elif plot_name == "100_single_ep_corr_classification_rate_mnist_old":
    """
    Get the correct classification rate with 100 runs of single-epoch-training
    """
    import ipdb
    from scipy.stats import spearmanr

    data_dirs = [
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/9-train-with-MNIST-MLP/2021-02-19T17-42-25--MLP-both_meanx0-factor-0-from-mnist-ctFalse-theta-1-s9311-100rns-train-trainOnTrue",
    ]
    num_smp_dataset = {"data0": 8357, "data1": 8326, "data2": 8566,
                       "data3": 8454, "data4": 8440, "data5": 8231,
                       "data6": 8371, "data7": 8357, "data8": 8384, "data9": 7701,
                       "mnist": 60000}

    for data_dir in data_dirs:
        files = find_files(data_dir, pattern="one*.csv")

        spearmanr_rec = []
        # get correct count in 100 rauns
        data_source = os.path.basename(files[0]).split("_")[-4]
        for ind in tqdm(range(len(files))):
            values = pd.read_csv(files[ind], header=0).values
            smp_ids = values[:, 0].astype(np.int)
            pat_ids = values[:, 1].astype(np.int)
            lbs = values[:, 2]
            prob = values[:, 3:]
            if ind == 0:  # the first file to get the total number (3-class) of samples
                total_num = num_smp_dataset[data_source]  # 3-class samples id
                ids_w_count = []
                noisy_lb_counts = []
                dict_count = {key: 0 for key in np.arange(total_num)}  # total number 9243
                noisy_lb_rec = {key: 0 for key in np.arange(total_num)}  # total number 9243
                pre_rank = np.arange(total_num)  #indices

            pred_lbs = np.argmax(prob, axis=1)
            right_inds = np.where(pred_lbs == lbs)[0]
            # right_inds = np.where(pred_lbs == pat_ids)[0]
            correct = np.unique(smp_ids[right_inds])
            ids_w_count += list(correct)
            # noisy_lb_counts += list(np.unique(smp_ids[pat_ids != lbs]))  #  it should be the same for every file
            

            if ind % 10 == 0:
                count_all = Counter(ids_w_count)
                dict_count.update(count_all)
                curr_count_array = np.array([[key, val] for (key, val) in dict_count.items()])
                curr_rank = curr_count_array[np.argsort(curr_count_array[:, 1]),0]
                # spearmanr_rec.append([ind, np.sum(curr_rank==pre_rank)])
                spearmanr_rec.append([ind, spearmanr(pre_rank, curr_rank)[0]])
                pre_rank = curr_rank.copy()

        count_all = Counter(ids_w_count)
        dict_count.update(count_all)
        # noisy_lb_rec.update(Counter(noisy_lb_counts))
        noisy_lb_counts = list(np.unique(
            smp_ids[pat_ids != lbs]))  # it should be the same for every file
        
        noisy_lb_rec.update(Counter(noisy_lb_counts))

        counter_array = np.array([[key, dict_count[key]] for key in np.arange(total_num)])
        noisy_inds_array = np.array([[key, noisy_lb_rec[key]] for key in np.arange(total_num)])
        ipdb.set_trace()
        # noisy_inds_array = np.array([[key, val] for (key, val) in noisy_lb_rec.items()])
        sort_inds = np.argsort(counter_array[:, 1])
        sample_ids_key = counter_array[sort_inds][:, 0]
        # rates = counter_array[sort_inds, 1]/counter_array[:, 1].max()
        rates = counter_array[sort_inds][:, 1] / len(files)
        noisy_lb_rate = noisy_inds_array[sort_inds][:, 1]

        assert np.sum(counter_array[sort_inds, 0] == noisy_inds_array[sort_inds, 0]), "sorted sample indices mismatch"

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("sample sorted by the correct clf. rate"),
        ax1.set_ylabel("correct clf. rate (over 100 runs)"),
        ax1.plot(rates, label="original data set"),
        ax1.tick_params(axis='y'),
        ax1.set_ylim([0, 1.0])
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('counts'),  # we already handled the x-label with ax1
        ax2.plot(noisy_lb_rate.cumsum(), "m", label="accum. # of noisy labels"),
        ax2.plot(np.ones(total_num).cumsum(), "c", label="accum. # of all samples")
        ax2.set_ylim([0, total_num])
        ax2.tick_params(axis='y')
        ax2.legend(loc="upper right")
        plt.title("distillation effect-{}.png".format(data_source))
        plt.savefig(
            data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}.png".format(
                os.path.basename(files[0]).split("_")[-6], total_num, data_source)),
        plt.savefig(
            data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}.pdf".format(
                os.path.basename(files[0]).split("_")[-6], total_num, data_source), format="pdf")
        print("ok")
        plt.close()
        
    concat_data = np.concatenate((
                                 np.array(sort_inds).reshape(-1, 1), rates.reshape(-1, 1)), axis=1)
    np.savetxt(data_dir + "/full_summary-{}_100_runs_sort_inds_rate_({}-{}).csv".format(data_source, os.path.basename(
        files[0]).split("_")[7], total_num), concat_data, fmt="%.5f", delimiter=",",
               header="ori_sort_rate_id,ori_sort_rate,true_lbs,noisy_lbs")


elif plot_name == "100_single_ep_corr_classification_rate_mnist_old2":
    """
    Get the correct classification rate with 100 runs of single-epoch-training
    """
    import ipdb
    from scipy.stats import spearmanr

    data_dirs = [
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/9-train-with-MNIST-MLP/2021-02-10T13-20-28--MLP-both_meanx0-factor-0-from-mnist-certainFalse-theta-1-s379-100rns-train"
    ]
    num_smp_dataset = {"data0": 8357, "data1": 8326, "data2": 8566,
                       "data3": 8454, "data4": 8440, "data5": 8231,
                       "data6": 8371, "data7": 8357, "data8": 8384, "data9": 7701,
                       "mnist": 60000}

    for data_dir in data_dirs:
        files = find_files(data_dir, pattern="one*.csv")

        spearmanr_rec = []
        print("number of files: ", len(files))
        data_source = os.path.basename(files[0]).split("_")[-4]
        for ind in tqdm(range(len(files))):
            fn = find_files(data_dir, pattern="one_{}*.csv".format(ind))
            values = pd.read_csv(fn[0], header=0).values
            smp_ids = values[:, 0].astype(np.int)
            pat_ids = values[:, 1].astype(np.int)
            lbs = values[:, 2]
            prob = values[:, 3:]
            if ind == 0:  # the first file to get the total number (3-class) of samples
                total_num = num_smp_dataset[data_source]  # 3-class samples id
                ids_w_count = []
                noisy_lb_counts = []
                dict_count = {key: 0 for key in np.arange(total_num)}  # total number 9243
                noisy_lb_rec = {key: 0 for key in np.arange(total_num)}  # total number 9243
                pre_rank = np.arange(total_num)  #indices

            pred_lbs = np.argmax(prob, axis=1)
            right_inds = np.where(pred_lbs == pat_ids)[0]
            correct = np.unique(smp_ids[right_inds])
            ids_w_count += list(correct)
            noisy_lb_counts += list(smp_ids[pat_ids != lbs])  # sample ids that with noisy labels

            if ind % 10 == 0:
                count_all = Counter(ids_w_count)
                dict_count.update(count_all)
                curr_count_array = np.array([[key, val] for (key, val) in dict_count.items()])
                curr_rank = curr_count_array[np.argsort(curr_count_array[:, 1]),0]
                # spearmanr_rec.append([ind, np.sum(curr_rank==pre_rank)])
                spearmanr_rec.append([ind, spearmanr(pre_rank, curr_rank)[0]])
                pre_rank = curr_rank.copy()

        count_all = Counter(ids_w_count)
        dict_count.update(count_all)
        noisy_lb_rec.update(Counter(noisy_lb_counts))

        counter_array = np.array([[key, val] for (key, val) in dict_count.items()])
        noisy_inds_array = np.array([[key, val/len(files)] for (key, val) in noisy_lb_rec.items()])
        sort_inds = np.argsort(counter_array[:, 1])
        sample_ids_key = counter_array[sort_inds, 0]
        rates = counter_array[sort_inds, 1] / len(files)
        noisy_lb_rate = noisy_inds_array[sort_inds, 1]

        ipdb.set_trace()
        assert np.sum(counter_array[sort_inds, 0] == noisy_inds_array[sort_inds, 0]) == total_num, "sorted sample indices mismatch"

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("sample sorted by the correct clf. rate"),
        ax1.set_ylabel("correct clf. rate (over 100 runs)"),
        ax1.plot(rates, label="original data set"),
        ax1.tick_params(axis='y'),
        ax1.set_ylim([0, 1.0])
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('counts'),  # we already handled the x-label with ax1
        ax2.plot(noisy_lb_rate.cumsum(), "m", label="accum. # of noisy labels"),
        ax2.plot(np.ones(total_num).cumsum(), "c", label="accum. # of all samples")
        ax2.set_ylim([0, total_num])
        ax2.tick_params(axis='y')
        ax2.legend(loc="upper right")
        plt.title("distillation effect-{}.png".format(data_source))
        plt.savefig(
            data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}.png".format(
                os.path.basename(files[0]).split("_")[-5], total_num, data_source)),
        plt.savefig(
            data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}.pdf".format(
                os.path.basename(files[0]).split("_")[-5], total_num, data_source), format="pdf")
        print("ok")
        plt.close()
        
    ipdb.set_trace()
    concat_data = np.concatenate((
                                 np.array(sort_inds).reshape(-1, 1), rates.reshape(-1, 1)), axis=1)
    np.savetxt(data_dir + "/full_summary-{}_100_runs_sort_inds_rate_({}-{}).csv".format(data_source, os.path.basename(
        files[0]).split("_")[-4], total_num), concat_data, fmt="%.5f", delimiter=",",
               header="ori_sort_rate_id,ori_sort_rate,true_lbs,noisy_lbs")
    

elif plot_name == "100_single_ep_corr_classification_rate":
    """
    Get the correct classification rate with 100 runs of single-epoch-training
    """
    import ipdb

    print("Plot_name: ", plot_name)
    from scipy.stats import spearmanr

    data_dirs = [
       "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2021-01-12T11-47-50--Inception-nonex0-factor-0-from-data9-certainFalse-theta-0-s989-100rns-train",
       "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2021-01-12T11-47-51--Inception-nonex0-factor-0-from-data8-certainFalse-theta-0-s989-100rns-train",
       "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2021-01-12T11-47-52--Inception-nonex0-factor-0-from-data7-certainFalse-theta-0-s989-100rns-train",
       "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2021-01-12T11-47-54--Inception-nonex0-factor-0-from-data6-certainFalse-theta-0-s989-100rns-train",
       "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2021-01-12T11-47-55--Inception-nonex0-factor-0-from-data4-certainFalse-theta-0-s989-100rns-train",
       "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2021-01-12T11-47-56--Inception-nonex0-factor-0-from-data3-certainFalse-theta-0-s989-100rns-train"
    ]
    num_smp_dataset = {"data0": 8357, "data1": 8326, "data2": 8566,
                       "data3": 8454, "data4": 8440, "data5": 8231,
                       "data6": 8371, "data7": 8357, "data8": 8384, "data9": 7701,
                       "mnist": 70000}

    for data_dir in data_dirs:
        print(data_dir)
        files = find_files(data_dir, pattern="one_ep_data_train*.csv")

        # get correct count in 100 rauns
        data_source = os.path.basename(files[0]).split("_")[8]
        for ind, fn in enumerate(files):
            values = pd.read_csv(fn, header=0).values
            smp_ids = values[:, 0].astype(np.int)
            true_lables = values[:, 1].astype(np.int)
            noisy_lbs = values[:, 2]
            prob = values[:, 3:]
            if ind == 0:  # the first file to get the total number (3-class) of samples
                total_num = num_smp_dataset[data_source]  # 3-class samples id
                correct_ids_w_count = []
                correct_dict_id_w_count = {key: 0 for key in np.arange(total_num)}  #

            pred_lbs = np.argmax(prob, axis=1)
            right_inds = np.where(pred_lbs == noisy_lbs)[0]
            correct_sample_ids = np.unique(smp_ids[right_inds])
            correct_ids_w_count += list(correct_sample_ids)

        correct_id_count_all = Counter(correct_ids_w_count)
        correct_dict_id_w_count.update(correct_id_count_all)

        # if theta == 0.975:
        #     ipdb.set_trace()
        counter_array = np.array([[key, val] for (key, val) in correct_dict_id_w_count.items()])
        sort_inds = np.argsort(counter_array[:, 1])
        sample_ids_key = counter_array[sort_inds, 0]
        # rates = counter_array[sort_inds, 1]/counter_array[:, 1].max()
        rates = counter_array[sort_inds, 1] / len(files)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("sample index (sorted)"),
        ax1.set_ylabel("correct clf. rate (over 100 runs)"),
        ax1.plot(rates, label="whole data set"),
        ax1.tick_params(axis='y'),
        ax1.set_ylim([0, 1.0])
        ax1.legend(loc="upper left")

        plt.title("distillation effect {}.png".format(data_source))
        plt.savefig(data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}.png".format(
                os.path.basename(files[0]).split("_")[7], total_num, data_source)),
        plt.savefig(data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}.pdf".format(
                os.path.basename(files[0]).split("_")[7], total_num, data_source), format="pdf")
        print("ok")
        plt.close()

        concat_data = np.concatenate((
                                     np.array(sort_inds).reshape(-1, 1), rates.reshape(-1, 1)), axis=1)
        np.savetxt(data_dir + "/full_summary-{}_100_runs_sort_inds_rate_({}-{}).csv".format(data_source, os.path.basename(
            files[0]).split("_")[7], total_num), concat_data, fmt="%.5f", delimiter=",",
                   header="sort_samp_ids,sort_corr_rate")


elif plot_name == "100_single_ep_patient_wise_rate":
    # load original data to get patient-wise statistics
    from scipy.io import loadmat as loadmat
    import scipy.io as io

    print("Plot_name: ", plot_name)
    ori_data = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325_DATA.mat"
    ## Get the selection rate patien-wise, corr_rate also patient-wise
    sort_inds_files = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/100_runs_sort_inds_rate_certain.csv"
    sort_data = pd.read_csv(sort_inds_files, header=0).values
    sort_inds = sort_data[:, 0].astype(np.int)
    sort_ori_corr_rate = sort_data[:, 1]
    sort_select_rate = sort_data[:, 2]
    sort_select_corr_rate = sort_data[:, 3]

    original_data = loadmat(ori_data)["DATA"]
    sort_pat_ids = original_data[:, 0][sort_inds]
    labels = original_data[:, 1]
    new_mat = np.zeros((original_data.shape[0], original_data.shape[1] + 1))
    new_mat[:, 0] = np.arange(original_data.shape[0])  # tag every sample
    new_mat[:, 1:] = original_data
    train_data = {}
    test_data = {}
    true_lables = original_data[:, 0].astype(np.int)

    uniq_pat_ids = np.unique(sort_pat_ids)
    pat_summary = []
    for pid in uniq_pat_ids:
        pat_inds = np.where(sort_pat_ids == pid)[0]
        corr_rate = np.mean(sort_ori_corr_rate[pat_inds])
        select_rate = np.mean(sort_select_rate[pat_inds])
        select_corr_rate = np.mean(sort_select_corr_rate[pat_inds])
        pat_summary.append([pid, corr_rate, select_rate, select_corr_rate, len(pat_inds)])
    print("ok")
    pat_sort_sum = sorted(pat_summary, key=lambda x: x[1])
    np.savetxt(os.path.dirname(
        sort_inds_files) + "/100-runs-pat-ids-sorted-[pid,corr_rate,select_rate, select_corr_rate,num_samples].csv",
               np.array(pat_sort_sum), fmt="%.5f", delimiter=",")

    plt.plot(np.array(pat_sort_sum)[:, 1], label="ori. corr rate"),
    plt.plot(np.array(pat_sort_sum)[:, 2], label="dist. select rate"),
    plt.plot(np.array(pat_sort_sum)[:, 3], label="dist. corr rate"),
    plt.legend()
    plt.xlabel("patient index (sorted)")
    plt.ylabel("normalized rate (100 runs)")
    plt.title(
        "Patient-wise sorted by correct rate")
    plt.savefig(os.path.dirname(sort_inds_files) + "/100-runs-patient-wise-statistics-sort-by-ori-corr-rate.png")
    plt.savefig(os.path.dirname(sort_inds_files) + "/100-runs-patient-wise-statistics-sort-by-ori-corr-rate.pdf",
                format="pdf")
    plt.close()

    plt.plot(np.array(pat_sort_sum)[:, 4], color="c",
             label="# of samples")
    plt.xlabel("patient index (sorted)")
    plt.ylabel("# of samples")
    plt.savefig(os.path.dirname(sort_inds_files) + "/100-runs-patient-wise-statistics-sort-by-num-amples.png")
    plt.savefig(os.path.dirname(sort_inds_files) + "/100-runs-patient-wise-statistics-sort-by-num-amples.pdf",
                format="pdf")
    plt.close()


elif plot_name == "K_NN_stats_test_for_distillation":
    from scipy.io import loadmat
    from sklearn.metrics import pairwise_distances
    from scipy import stats

    print("Plot_name: ", plot_name)
    
    data_dir = "../data/20190325/20190325-3class_lout40_train_test_data5.mat"
    corr_clf_rate_file = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T22-07-37--Res_ECG_CAM-nonex0-factor-0-from-data5-certainFalse-theta-0-s989-100rns-train/full_summary-data5_100_runs_sort_inds_rate_(6592-8229-8231).csv"
    theta = 0.5
    num_2class = np.int(os.path.basename(corr_clf_rate_file).split("(")[1].split("-")[0])
    num_3class = np.int(os.path.basename(corr_clf_rate_file).split(")")[0].split("-")[-1])

    original_data = loadmat(data_dir)["DATA"]
    labels = original_data[:, 1]
    whole_set = np.zeros((original_data.shape[0], original_data.shape[1] + 1))
    whole_set[:, 0] = np.arange(original_data.shape[0])  # tag every sample
    whole_set[:, 1:] = original_data

    sub_inds = np.empty((0))
    for class_id in range(2):
        sub_inds = np.append(sub_inds, np.where(labels == class_id)[0])
    sub_inds = sub_inds.astype(np.int32)
    sub_set= whole_set[sub_inds]
    sub_set_lbs= labels[sub_inds]
    sub_set_smp_ids= whole_set[sub_inds, 0]

    dd = pd.read_csv(corr_clf_rate_file, header=0).values
    distill_samp_ids = dd[:,0][-np.int(theta*num_2class):].astype(np.int)
    distill_set = whole_set[distill_samp_ids,:]
    distill_lbs = distill_set[:,2]

    ## get pair-wise distance of whole set
    dist_whole = pairwise_distances(original_data[:, 2:])

    dist_2class_temp = dist_whole[sub_inds,:]
    dist_2class = dist_2class_temp[:,sub_inds]
    k_nbs = 11   # the number of neighbors to inspect
    knn_data_2class = []
    for ii in range(len(dist_2class)):
        idx = np.argsort(dist_2class[ii, :])
        small_k_dists = dist_2class[ii, :][idx[:k_nbs]]
        knn_data_2class.append([sub_set_smp_ids[ii], sub_set_lbs[ii], sub_set_lbs[idx][:k_nbs], np.float((np.sum(sub_set_lbs[idx][:k_nbs]==sub_set_lbs[ii])-1)/1.0/(k_nbs-1))])
    knn_data_2class = np.array(knn_data_2class)
    print("whole set median: ", np.mean(knn_data_2class[:, -1]))

    # get percentage of same and ops class in KNN for the whole set
    temp = dist_whole[distill_samp_ids,:]
    dist_distill = temp[:,distill_samp_ids]
    knn_data_distill = []
    for jj in range(len(dist_distill)):
        idx = np.argsort(dist_distill[jj, :])
        small_k_dists = dist_distill[jj, :][idx[:k_nbs]]
        knn_data_distill.append([distill_samp_ids[jj], distill_lbs[jj], distill_lbs[idx][:k_nbs], np.float((np.sum(distill_lbs[idx][:k_nbs]==distill_lbs[jj])-1)/1.0/(k_nbs-1))])
    knn_data_distill = np.array(knn_data_distill)
    print("distill mean: ", np.median(knn_data_distill[:, -1]))

    _, p_tt = stats.ttest_ind(knn_data_2class[:,-1], knn_data_distill[:,-1])
    _, p_rank = stats.ranksums(knn_data_2class[:, -1], knn_data_distill[:, -1])
    _, p_levene = stats.levene(knn_data_2class[:, -1], knn_data_distill[:, -1])

    plt.hist(knn_data_2class[:,-1], color="c", alpha=0.5, label="original", density=True),
    plt.hist(knn_data_distill[:,-1], color="m", alpha=0.5, label="distilled", density=True),
    plt.legend(),
    plt.xlabel("K nearest neightbors with the same label [%]"),
    plt.ylabel("count (density)"),
    plt.title("\n".join(wrap("Hist. of K ({}) nearest points distribution, p={:.2E}".format(k_nbs-1, p_rank), 60)))
    plt.savefig(os.path.join(os.path.dirname(corr_clf_rate_file), "KNN-{}-membership-distribution-p{:.2E}.png".format(k_nbs-1, p_rank)))
    plt.savefig(os.path.join(os.path.dirname(corr_clf_rate_file), "KNN-{}-membership-distribution-p{:.2E}.pdf".format(k_nbs-1, p_rank)), format="pdf")
    plt.close()


elif plot_name == "distill_valid_labels":
    from scipy import stats

    print("Plot_name: ", plot_name)
    # Nenad validated the labels of these samples
    m_file = "C:/Users/LDY/Desktop/all-experiment-results/metabolites/20190325-certain-Validate.mat"
    mat = scipy.io.loadmat(m_file)["Validate"]  # [id, label, features]
    samp_ids = mat[:, 0]
    true_lables = mat[:, 1]
    labels = mat[:, 2]
    corr_or_wrong = mat[:, 3]
    
    wrong_inds = np.where(corr_or_wrong == 0)[0]
    wrong_labels = labels[wrong_inds]
    np.sum(wrong_labels==1), np.sum(wrong_labels==0)
    
    ccr_summary = "C:/Users/LDY/Desktop/all-experiment-results/metabolites/full_summary-data5_100_runs_sort_inds_rate_(6592-8229-8231).csv"
    summary = pd.read_csv(ccr_summary, header=0).values
    ccr_samp_ids = summary[:, 0]
    ccr_samp_ccr = summary[:, 1]
    
    valid_ccr = []
    for sp_id, pat, lb, r_or_w in zip(samp_ids, true_lables, labels, corr_or_wrong):
        ccr = ccr_samp_ccr[np.int(np.where(ccr_samp_ids == sp_id)[0])]
        valid_ccr.append([sp_id, pat, lb, r_or_w, ccr])
        
    valid_ccr = np.array(valid_ccr)
    
    # compute what would be optimal ccr such that
    fpr, tpr, thrs = metrics.roc_curve(valid_ccr[:,-2], valid_ccr[:,-1])
    auc = metrics.roc_auc_score(valid_ccr[:,-2], valid_ccr[:,-1])
    rocDF, threshold = Find_Optimal_Cutoff(valid_ccr[:,-2], valid_ccr[:,-1])
    """
        fpr       tpr  1-fpr        tf    thresholds
        0.38  0.639309   0.62  0.019309     0.49254
    """
    
    
    plt.figure(figsize=[7,7]),
    plt.plot(np.arange(len(tpr))/len(tpr), tpr, label="TPR"),
    plt.plot(np.arange(len(fpr))/len(fpr), 1-fpr, color = 'r', label="1-FPR"),
    plt.xlabel("1-FPR"),
    plt.ylabel("TPR"),
    plt.scatter(rocDF["1-fpr"].values, rocDF["tpr"].values, marker="o")
    plt.text(rocDF["1-fpr"], rocDF["tpr"], "ccr:{}".format(rocDF["threshold"].values))
    plt.xlim([0,1.01]),
    plt.ylim([0,1.01]),
    
    plt.figure(figsize=[7,7]),
    plt.plot(fpr, tpr),
    plt.title("Optimal CCR={} to set theta={:.1f}%".format(threshold, scipy.stats.percentileofscore(ccr_samp_ccr, threshold))),
    plt.plot([0, 1], [1, 0]),
    plt.xlabel("FPR"),
    plt.ylabel("TPR"),
    plt.xlim([0,1.01]),
    plt.ylim([0,1.01]),
    plt.text(rocDF["fpr"], rocDF["tpr"], "ccr:{}".format(rocDF["threshold"].values[0])),
    plt.scatter(rocDF["fpr"].values, rocDF["tpr"].values, marker="o")
    plt.savefig(os.path.join(os.path.dirname(ccr_summary), "optimal-theta-with-validated-labels-ccr-{:.4f}-theta{:.2f}.png".format(rocDF["threshold"].values[0], scipy.stats.percentileofscore(ccr_samp_ccr, threshold))))
    plt.savefig(os.path.join(os.path.dirname(ccr_summary), "optimal-theta-with-validated-labels-ccr-{}-theta{:.2f}.pdf".format(rocDF["threshold"].values[0], scipy.stats.percentileofscore(ccr_samp_ccr, threshold))),  format="pdf")
    plt.close()
    
    correct_ccr = valid_ccr[np.where(valid_ccr[:,3] == 1)[0]]
    wrong_ccr = valid_ccr[np.where(valid_ccr[:,3] == 0)[0]]
    print("ok")
    _, p_rank = stats.ranksums(correct_ccr[:, -1], wrong_ccr[:, -1])

    
    plt.hist(np.array(correct_ccr[:, -1]), alpha=0.5, density=True, label="correct"),
    plt.hist(np.array(wrong_ccr[:, -1]), density=True, alpha=0.5, label="wrong")
    plt.legend()
    plt.title("\n".join(wrap("Validated labels with CCR distribution p{:.2E}".format(p_rank), 60)))
    plt.xlabel("correct classification rate")
    plt.ylabel("frequency (density)")
    plt.savefig(os.path.join("C:/Users/LDY/Desktop", "CCR-distribution-of-NP-validated-samples-p{:.2E}.png".format(p_rank)))
    plt.savefig(os.path.join("C:/Users/LDY/Desktop", "CCR-distribution-of-NP-validated-samples-p{:.2E}.pdf".format(p_rank)), format="pdf")
    plt.close()
    print("ok")


elif plot_name == "re_split_data_0_9_except_5":
    print("Plot_name: ", plot_name)
    src_data = ["data{}".format(jj) for jj in [0,1,2,3,4,6,7,8,9]]
    data_source_dirs = [
        "../data/20190325/20190325-3class_lout40_train_val_{}.mat".format(
            src_dir) for src_dir in src_data]

    # each cross_validation set
    coll_mat = np.empty((0, 290))
    for dd in data_source_dirs:
        ## load original .mat data and split train_val
        mat = scipy.io.loadmat(dd)["DATA"]
        coll_mat = np.vstack((coll_mat, mat))
    np.random.shuffle(coll_mat)
    print("ok")
    for cc in range(3):
        print("class ", cc, np.sum(coll_mat[:, 1] == cc))
    total_pat_ids = Counter(coll_mat[:, 0])
    np.random.shuffle(coll_mat)
    trian_val_part = {key: {} for key in range(5)}
    part_len = len(coll_mat) // 5
    for ii in range(4):
        trian_val_part[ii]["DATA"] = coll_mat[ii * part_len: (ii + 1) * part_len]
        num_pat = len(Counter(trian_val_part[ii]["DATA"][:, 0]))
        scipy.io.savemat(
            os.path.join(os.path.dirname(dd),
                         "5_fold_20190325-3class[{}-{}-{}]_pat_{}_test_data{}.mat".format(
                             np.sum(trian_val_part[ii]["DATA"][:, 1] == 0),
                             np.sum(trian_val_part[ii]["DATA"][:, 1] == 1),
                             np.sum(trian_val_part[ii]["DATA"][:, 1] == 2), num_pat, ii)), trian_val_part[ii])

    trian_val_part[4]["DATA"] = coll_mat[4 * part_len:]
    num_pat = len(Counter(trian_val_part[4]["DATA"][:, 0]))
    ii = 4
    scipy.io.savemat(
        os.path.join(os.path.dirname(dd),
                     "5_fold_20190325-3class[{}-{}-{}]_pat_{}_test_data{}.mat".format(
                         np.sum(trian_val_part[ii]["DATA"][:, 1] == 0),
                         np.sum(trian_val_part[ii]["DATA"][:, 1] == 1),
                         np.sum(trian_val_part[ii]["DATA"][:, 1] == 2), num_pat, ii)), trian_val_part[ii])


    # merge the other 4 sets to form train_validation set
    for jj in range(5):
        train_inds = list(np.arange(5))
        del train_inds[jj]
        train_coll = {"DATA": np.empty((0, 290))}
        for ii in train_inds:
            train_coll["DATA"] = np.vstack((train_coll["DATA"], trian_val_part[ii]["DATA"]))
        num_pat = len(Counter(train_coll["DATA"][:, 0]))
        print("ok")
        scipy.io.savemat(
            os.path.join(os.path.dirname(dd),
                         "5_fold_20190325-3class[{}-{}-{}]_pat_{}_train_val_data{}.mat".format(
                             np.sum(train_coll["DATA"][:, 1] == 0),
                             np.sum(train_coll["DATA"][:, 1] == 1),
                             np.sum(train_coll["DATA"][:, 1] == 2), num_pat, jj)), train_coll)


elif plot_name == "re_split_data_0_9_except_5_patient_wise_get_data_statistics":
    print("Plot_name: ", plot_name)
    src_data = ["data{}".format(jj) for jj in [0,1,2,3,4,6,7,8,9]]
    data_dir_root = "../data/20190325"
    data_source_dirs = [os.path.join(data_dir_root, "20190325-3class_lout40_test_{}.mat".format(
                src_dir)) for src_dir in src_data]

    # each cross_validation set
    coll_mat = np.empty((0, 290))
    for dd in data_source_dirs:
        ## load original .mat data and split train_val
        mat = scipy.io.loadmat(dd)["DATA"]
        coll_mat = np.vstack((coll_mat, mat))
    #------------------------------------------------------------------------------------------------------

    # collect spectra for each patient
    per_pat_spectra= {}
    per_pat_spectra_inds= {}
    for i in range(len(coll_mat)):
        if coll_mat[i, 1] == 2:
            continue
        pat_id = coll_mat[i,0]
        if pat_id in per_pat_spectra:
            per_pat_spectra[pat_id] = np.vstack((per_pat_spectra[pat_id], coll_mat[i]))
            per_pat_spectra_inds[pat_id].append(i)
        else:
            per_pat_spectra.update({pat_id : coll_mat[i]})
            per_pat_spectra_inds.update({pat_id : [i]})
    true_lables = [[x, len(per_pat_spectra[x])] for x in per_pat_spectra.keys()]
    pat_ids_inds = [[x, len(per_pat_spectra_inds[x])] for x in per_pat_spectra_inds.keys()]
    # ------------------------------------------------------------------------------------------------------
    
    n_cv_folds = 5
    n_percentile_folds = 4
    percentile_threshold = []
    # get thresholds for different percentiles
    percentiles = (100 // n_percentile_folds)*np.arange(n_percentile_folds+1)
    for perc_fold in percentiles:
        threshold = np.percentile(np.array(true_lables)[:, 1], perc_fold)
        percentile_threshold.append(threshold)
        
    # last threshold should be infinite
    percentile_threshold[-1]=9999
    
    pat_perc_split = {key: [] for key in np.arange(n_percentile_folds)}
    for pt_id, num in true_lables:
        for i in range(n_percentile_folds):
            if num >= percentile_threshold[i] and num < percentile_threshold[i+1]:
                pat_perc_split[i].append([pt_id, num, i])

    # ------------------------------------------------------------------------------------------------------
    
    new_CV_fold_train_val_ids = {key: np.empty((0, 3)) for key in np.arange(n_cv_folds)}
    new_CV_fold_test_ids = {key: np.empty((0, 3)) for key in np.arange(n_cv_folds)}
    ## check whether the patient split is clean -- YES
    for ii in range(n_percentile_folds):
        for jj in range(n_percentile_folds):
            S_ii = set(np.array(pat_perc_split[ii])[:, 0])
            S_jj = set(np.array(pat_perc_split[jj])[:, 0])
            if ii != jj:
                print("fold {} and fold {} has intersection {}".format(ii, jj, S_ii.intersection(S_jj)))
    
    for percentile_fold in range(n_percentile_folds):
        np.random.shuffle(pat_perc_split[percentile_fold])
    # ------------------------------------------------------------------------------------------------------
    print("----------------------------")
    
    # split each percentile fold into n_cv_folds and assign to different n_cv_folds
    for percentile_fold in range(n_percentile_folds):
        for cv_fold in range(n_cv_folds):
            num2pick = len(pat_perc_split[percentile_fold]) // n_cv_folds
            if cv_fold != n_cv_folds-1:
                new_CV_fold_train_val_ids[cv_fold] = np.vstack((new_CV_fold_train_val_ids[cv_fold], pat_perc_split[percentile_fold][cv_fold * num2pick: (cv_fold + 1) * num2pick]))
                print("{} cv-fold get {} samples from {} percentile-fold".format(cv_fold, num2pick, percentile_fold))
            else:
                new_CV_fold_train_val_ids[cv_fold] = np.vstack((new_CV_fold_train_val_ids[cv_fold], pat_perc_split[percentile_fold][cv_fold * num2pick:]))
                print("{} cv-fold get {} samples from {} percentile-fold".format(
                    cv_fold, len(pat_perc_split[percentile_fold][cv_fold*num2pick: ]), percentile_fold))
    # ------------------------------------------------------------------------------------------------------
                
    ## check whether the CV fold patient-wise split is clean -- YES
    for ii in range(n_cv_folds):
        for jj in range(n_cv_folds):
            S_ii = set(np.array(new_CV_fold_train_val_ids[ii])[:, 0])
            S_jj = set(np.array(new_CV_fold_train_val_ids[jj])[:, 0])
            if ii != jj:
                print("fold {} has {} samples\n".format(ii, len(new_CV_fold_train_val_ids[ii])))
                print("fold {} and fold {} has interaction {}\n".format(ii, jj, S_ii.intersection(S_jj)))
    # ------------------------------------------------------------------------------------------------------
    
    # merge other folds to get the train_val data and test data
    for cv_fold in range(n_cv_folds):
        train_folds = list(np.arange(n_cv_folds))
        del train_folds[cv_fold]
        print(cv_fold, "train ", train_folds)
        
        # combine patient ids for train_val and test set AND check if there is overlap
        trian_val_pat_ids = np.empty((0, 3))
        for train_index in train_folds:
            trian_val_pat_ids = np.vstack((trian_val_pat_ids, new_CV_fold_train_val_ids[train_index]))
            
        test_pat_ids = new_CV_fold_train_val_ids[cv_fold]

        S_ii = set(np.array(test_pat_ids)[:, 0])
        S_jj = set(np.array(trian_val_pat_ids)[:, 0])
        print("cv-{}-fold interaction {}\n".format(cv_fold, S_ii.intersection(S_jj)))

        new_CV_fold_train_val_data = {"DATA": np.empty((0, 290))}  #np.empty((0, 290))
        for pat_id in trian_val_pat_ids[:, 0]:
            new_CV_fold_train_val_data["DATA"] = np.vstack((new_CV_fold_train_val_data["DATA"], coll_mat[per_pat_spectra_inds[pat_id]]))
        np.savetxt(os.path.join(data_dir_root,"{}_fold_pat_split_20190325-2class_train_val_data{}_patient_ids_{}.csv".format(n_cv_folds, cv_fold, len(trian_val_pat_ids))), trian_val_pat_ids, header="pat_id,count,percentile{}".format(n_percentile_folds), fmt="%.3f", delimiter=",")
        np.savetxt(os.path.join(data_dir_root,"{}_fold_pat_split_20190325-2class_test_data{}_patient_ids_{}.csv".format(n_cv_folds, cv_fold, len(test_pat_ids))), test_pat_ids, header="pat_id,count,percentile{}".format(n_percentile_folds), fmt="%.3f", delimiter=",")
        
        new_CV_fold_test_data = {"DATA": np.empty((0, 290))}  #np.empty((0, 290))
        for pat_id in test_pat_ids[:, 0]:
            new_CV_fold_test_data["DATA"] = np.vstack((new_CV_fold_test_data["DATA"], coll_mat[per_pat_spectra_inds[pat_id]]))

        scipy.io.savemat(os.path.join(data_dir_root,"{}_fold_pat_split_20190325-2class_train_val_data{}.mat".format(n_cv_folds, cv_fold)), new_CV_fold_train_val_data)
        scipy.io.savemat(os.path.join(data_dir_root, "{}_fold_pat_split_20190325-2class_test_data{}.mat".format(n_cv_folds, cv_fold)), new_CV_fold_test_data)
        print(os.path.join(data_dir_root, "5_fold_pat_split_20190325-2class_test_data{}.mat".format(cv_fold)))

    print("ok")
    # -------------------------------------------------------------------------

elif plot_name == "get_d_prime":
    """
    {d'={\sqrt {2}}Z({AUC}).}
    """
    from scipy.stats import norm
    def z_of_auc(p):
        """
        Get z of auc where z is the inverse of the cdf of Gaussian distribution
        :return:
        """
        return norm.ppf(p)

    print("Plot_name: ", plot_name)
    
    # get the original labels
    data_dir = "../data/20190325"
    original = "../data/20190325/20190325-3class_lout40_val_data5-2class_human_performance844_with_labels.mat"
    ori = scipy.io.loadmat(original)["DATA"]
    true_label = ori[:, 1]
    
    # Get human's total labels
    human_rating = "../data/20190325/human-ratings-20190325-3class_lout40_val_data5-2class.mat"
    hum_whole = scipy.io.loadmat(human_rating)["data_ratings"]
    human_lb = hum_whole[:, 0]
    human_features = hum_whole[:, 1:]
    hum_fpr, hum_tpr, _ = metrics.roc_curve(true_label, human_lb)
    hum_auc = metrics.roc_auc_score(true_label, human_lb)
    hum_score = metrics.roc_auc_score(true_label, human_lb)
    
    tpr = hum_tpr[1]
    fpr = hum_fpr[1]

    get_auc_from_d_prime(tpr, fpr)
    
elif plot_name == "delete_folders":
    import shutil

    print("Plot_name: ", plot_name)
    target_dirs = [
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/9-train-with-MNIST-MLP/2021-02-13T07-06-49--MLP-both_meanx3-factor-0.5-from-mnist-ctFalse-theta-1-s3512-100rns-train-trainOnTrue",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/9-train-with-MNIST-MLP/2021-02-13T07-08-29--MLP-both_meanx0-factor-0-from-mnist-ctFalse-theta-1-s8522-100rns-train-trainOnTrue",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/9-train-with-MNIST-MLP/2021-02-19T15-55-16--MLP-both_meanx0-factor-0-from-mnist-ctFalse-theta-1-s4396-100rns-train-trainOnTrue",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/9-train-with-MNIST-MLP/2021-02-19T16-06-14--MLP-both_meanx0-factor-0-from-mnist-ctFalse-theta-1-s3397-100rns-train-trainOnTrue"
    ]
    for dd in target_dirs:
        print("Deleting ", dd)
        shutil.rmtree(dd)
        print("Done")
    print("All Done")
    
elif plot_name == "patient_wise_performance_ICLR":
    # data_dirs
    data_dirs = [r"C:\1-study\FIAS\1-My-papers\10-2021.02 ICLR AI for public health workshop\results always test mode\2021-02-22T20-00-10-classifier3-20spec-gentest-non-overlap-filter-16-aug-nonex0"]
    file_patterns = ["*test.csv", "*-test_doc.csv"]
    for data_dir in data_dirs:
        for pattern in file_patterns:
            files = find_files(data_dir, pattern=pattern)
            
            performance = {"ACC": np.empty((0,)), "patient_ACC": np.empty((0,)),
                           "AUC": np.empty((0,)), "SEN": np.empty((0,)),
                           "SPE": np.empty((0,)), "F1_score": np.empty((0,)),
                           "MCC": np.empty((0,))}
            performance_summary = []
            if len(files) > 0:
                for fn in files:
                    data = pd.read_csv(fn, header=None).values
                    sample_ids = data[:, 0]
                    patient_ids = data[:, 1]
                    true_labels = data[:, 2]
                    pred_logits = data[:, 3:]
    
                    # get the prob of one patient, average them, get predicted label, get right or wrong.
                    # get true labels for all unique patients
                    uniq_patients, uniq_patients_ind = np.unique(patient_ids, return_index=True)
                    patient_true_lb = {pat_id: true_labels[index] for pat_id, index in zip(uniq_patients, uniq_patients_ind)}
                    # get the index of each patient bag
                    uniq_patients_inds = {pat_id: [] for pat_id in uniq_patients}
                    uniq_patients_pred_lb = {pat_id: [] for pat_id in uniq_patients}
                    for ind, pat_id in enumerate(patient_ids):
                        uniq_patients_inds[pat_id].append(ind)
                    # get aggregated prob of each patient
                    aggregated_prob_per_patient = {pat_id: np.mean(pred_logits[uniq_patients_inds[pat_id]], axis=0) for pat_id in uniq_patients}
                    # get predicted label from the aggregated prob
                    aggregated_pred_lb_per_patient = {pat_id: np.argmax(aggregated_prob_per_patient[pat_id], axis=0) for pat_id in aggregated_prob_per_patient.keys()}
                    # Get the patient-wise accuracy
                    patient_acc = np.sum([aggregated_pred_lb_per_patient[pat_id]==patient_true_lb[pat_id] for pat_id in uniq_patients]) / len(uniq_patients) * 1.0
    
                    
    
                    accuracy, sensitivity, specificity, precision, F1_score, auc, fpr, tpr, mcc = get_scalar_performance_matrices_2classes(true_labels, pred_logits[:, 1], if_with_logits=True)
        
                    performance["ACC"] = np.append(performance["ACC"], accuracy)
                    performance["SEN"] = np.append(performance["SEN"],
                                                   sensitivity)
                    performance["SPE"] = np.append(performance["SPE"],
                                                   specificity)
                    performance["AUC"] = np.append(performance["AUC"], auc)
                    performance["F1_score"] = np.append(performance["F1_score"],
                                                        F1_score)
                    performance["MCC"] = np.append(performance["MCC"], mcc)
                    performance["patient_ACC"] = np.append(
                        performance["patient_ACC"],
                        patient_acc)
        
                    performance_summary.append(["performance of {}\n".format(os.path.basename(fn)),
                                                "Sensitivity: mean-{:.3f}, std-{:.3f}\n".format(
                                                    np.mean(performance["SEN"]),
                                                    np.std(performance["SEN"]))
                                                + "specificity: mean-{:.3f}, std-{:.3f}\n".format(
                                                    np.mean(performance["SPE"]),
                                                    np.std(performance["SPE"]))
                                                + "AUC: mean-{:.3f}, std-{:.3f}\n".format(
                                                    np.mean(performance["AUC"]),
                                                    np.std(performance["AUC"]))
                                                + "patient acc: mean-{:.3f}, std-{:.3f}\n".format(
                                                    np.mean(performance[
                                                                "patient_ACC"]),
                                                    np.std(performance[
                                                               "patient_ACC"]))
                                                + "F1-score: mean-{:.3f}, std-{:.3f}\n".format(
                                                    np.mean(
                                                        performance["F1_score"]),
                                                    np.std(
                                                        performance["F1_score"]))
                                                + "MCC: mean-{:.3f}, std-{:.3f}\n".format(
                                                    np.mean(performance["MCC"]),
                                                    np.std(performance["MCC"]))
                                                + "ACC: mean-{:.3f}, std-{:.3f}\n".format(
                                                    np.mean(performance["ACC"]),
                                                    np.std(performance["ACC"]))
        
                                                ])
            np.savetxt(os.path.join(data_dir,
                                    "AUC-{:.4f}-performance-summarries-of-{}.csv".format(
                                        np.mean(performance["AUC"]), os.path.basename(data_dir).split("-")[-1])),
                       np.array(performance_summary), fmt="%s", delimiter=",")
            print("---------{}-------{}---------------".format(os.path.basename(data_dir).split("-")[-1], pattern))
            print("ACC: mean-{:.3f}, std-{:.3f}\n".format(
                np.mean(performance["ACC"]),
                np.std(performance["ACC"])))
            print("AUC: mean-{:.3f}, std-{:.3f}\n".format(
                np.mean(performance["AUC"]),
                np.std(performance["AUC"])))
            print("patient acc: mean-{:.3f}, std-{:.3f}\n".format(
                np.mean(performance["patient_ACC"]),
                np.std(performance["patient_ACC"])))
            print("F1-score: mean-{:.3f}, std-{:.3f}\n".format(
                np.mean(performance["F1_score"]),
                np.std(performance["F1_score"])))
            print("MCC: mean-{:.3f}, std-{:.3f}\n".format(
                np.mean(performance["MCC"]),
                np.std(performance["MCC"])))
            print("Sensitivity: mean-{:.3f}, std-{:.3f}\n".format(
                np.mean(performance["SEN"]),
                np.std(performance["SEN"])))
            print("specificity: mean-{:.3f}, std-{:.3f}\n".format(
                np.mean(performance["SPE"]),
                np.std(performance["SPE"])))
            print("--------------END-----------------")
        else:
            print("No data!")
        
    
    

    
    



    








