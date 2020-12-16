import fnmatch
import os
import random
import itertools
import pickle

from sklearn.metrics import confusion_matrix
from scipy import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import scipy as scipy
from collections import Counter
from sklearn import metrics
from textwrap import wrap

from dataio import introduce_label_noisy

base = 22
args = {'legend.fontsize': base - 8,
          'figure.figsize': (10, 7),
         'axes.labelsize': base-4,
        #'weight' : 'bold',
         'axes.titlesize':base,
         'xtick.labelsize':base-8,
         'ytick.labelsize':base-8}
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


def get_scaler_performance_metrices():
    """
    Plot violin plots given the target dir of the test trials. Get the agg level [true-lb, agg-probability]
    :param pool_len:
    :param task_id:
    :return:
    """
    postfix = ""
    data_dir = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-Inception"
    folders = find_folderes(data_dir, pattern="*both*meanx5*data5*-test-0.*")
    performance = {"ACC": np.empty((0,)), "patient_ACC": np.empty((0,)), "AUC": np.empty((0,)), "SEN": np.empty((0,)), "SPE": np.empty((0,))}

    for fd in folders:
        file = find_files(fd, pattern="AUC_curve_step*.csv")
        num_patient = find_files(fd, pattern="*prob_distri_of*.png")
        # rat_id = os.path.basename(fn).split("-")[-3]
        values = pd.read_csv(file[0], header=0).values

        true_labels = values[:, 0]   # assign true-lbs and probs in aggregation
        prob = values[:, 1]
        cutoff_thr, auc = find_optimal_cutoff(true_labels, prob)
        pred_lbs = (prob > cutoff_thr).astype(np.int)

        class0_inds = np.where(true_labels == 0.0)[0]
        class1_inds = np.where(true_labels == 1.0)[0]
        class0_prob = prob[class0_inds]
        class1_prob = prob[class1_inds]

        patient_acc = np.sum(["right" in name for name in num_patient]) / len(num_patient)
        acc = np.sum(pred_lbs == true_labels) / true_labels.size
        # sen = np.sum(pred_lbs[class1_inds] == true_labels[class1_inds]) / class1_inds.size
        sen = np.sum(pred_lbs[np.where(true_labels == 1.0)[0]] == 1.0) / len(np.where(true_labels == 1.0)[0])
        spe = np.sum(pred_lbs[class0_inds] == true_labels[class0_inds]) / class0_inds.size

        # scores["class0"] = np.append(scores["class0"],
        #                              class0_prob)
        # scores["class1"] = np.append(scores["class1"],
        #                              class1_prob)
        performance["ACC"] = np.append(performance["ACC"], acc)
        performance["SEN"] = np.append(performance["SEN"], sen)
        performance["SPE"] = np.append(performance["SPE"], spe)
        performance["AUC"] = np.append(performance["AUC"], auc)
        performance["patient_ACC"] = np.append(performance["patient_ACC"], patient_acc)


    ## Human performance
    # human_file = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_val_data5-2class-human-ratings.mat"
    # original = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_val_data5-2class_human_performance844_with_labels.mat"
    # values = scipy.io.loadmat(human_file)["data_ratings"]
    #
    # true_v = scipy.io.loadmat(original)["DATA"]
    # ids = true_v[:, 0]
    # pred_lbs = values[:, 0]
    # true_lbs = true_v[:, 1]
    # count = dict(Counter(list(ids)))
    # human_diagnosis = []
    # right_count = 0
    # for id in count.keys():
    #     id_inds = np.where(ids == id)[0]
    #     vote_label = (np.sum(true_lbs[id_inds]) * 1.0 / id_inds.size).astype(np.int)
    #     vote_pred = ((np.sum(pred_lbs[id_inds]) / id_inds.size) > 0.5).astype(np.int)
    #     right_count = right_count + 1 if vote_label==vote_pred else right_count
    #     human_diagnosis.append((id, vote_label, vote_pred))
    #
    # human_diagnosis = np.array(human_diagnosis)
    # sen = np.sum(human_diagnosis[:, 2][np.where(human_diagnosis[:, 1] == 1)[0]] == 1) / len(
    #     np.where(human_diagnosis[:, 1] == 1)[0])
    # spe = np.sum(human_diagnosis[:, 2][np.where(human_diagnosis[:, 1] == 0)[0]] == 0) / len(
    #     np.where(human_diagnosis[:, 1] == 0)[0])
    #
    # np.savetxt("/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/" + "human_patient_wise_diagnosis.csv", np.array(human_diagnosis), header="id,true,pred", fmt="%d", delimiter=",")


    print("ave sen", np.mean(performance["SEN"]), "std sen", np.std(performance["SEN"]), '\n', "min", performance["SEN"].min(), "max", performance["SEN"].max(), '\n'),
    print("ave spe", np.mean(performance["SPE"]), "std sen", np.std(performance["SPE"]), '\n', "min", performance["SPE"].min(), "max", performance["SPE"].max(), '\n'),
    print("ave auc", np.mean(performance["AUC"]), "std auc", np.std(performance["AUC"]), '\n', "min", performance["AUC"].min(), "max", performance["AUC"].max(), '\n'),
    print("ave acc", np.mean(performance["ACC"]), "std acc", np.std(performance["ACC"]), '\n', "min", performance["ACC"].min(), "max", performance["ACC"].max(), '\n'),
    print("patient acc", np.mean(performance["patient_ACC"]), "std acc", np.std(performance["patient_ACC"]), '\n', "min", performance["patient_ACC"].min(), "max", performance["patient_ACC"].max(), '\n')


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
# ------------------------------------------------


original = "../data/20190325/20190325-3class_lout40_val_data5-2class_human_performance844_with_labels.mat"

plot_name = "test_performance_with_different_data_aug_parameters"


if plot_name == "indi_rating_with_model":
    data_dir = "../data/20190325"

    model_res_with_aug = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-10-09T13-47-36-data-lout40-datas-1d-class2-Res_ECG_CAM-0.766certainEp3-aug_ops_meanx10-0.3-test-auc0.79/AUCs/AUC_curve_step_0.00-auc_0.7905-lout40-datas.csv"

    human_indi_rating = "../data/20190325/lout40-data5-doctor_ratings_individual.mat"
    # Get individual rater's prediction
    true_indi_lbs = {}
    true_indi_model_lbs = {}
    human_indi_lbs = {}
    model_indi_lbs = {}
    indi_mat = scipy.io.loadmat(human_indi_rating)['a']
    indi_ratings = np.array(indi_mat)

    true_data = scipy.io.loadmat("/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_val_data5-2class_human_performance844_with_labels.mat")["DATA"]
    true_label = true_data[:, 1].astype(np.int)

    # Get model's prediction
    model_auc_with_aug = pd.read_csv(model_res_with_aug, header=0).values
    true_model_label = model_auc_with_aug[:, 0].astype(np.int)
    pred_logits = model_auc_with_aug[:, 1]
    pred_lb = np.argmax(pred_logits, axis=0)
    model_fpr, model_tpr, _ = metrics.roc_curve(true_model_label, pred_logits)

    mean_fpr = []
    mean_tpr = []
    mean_score = []
    base_fpr = np.linspace(0, 1, 20)

    tpr_model = []

    start = 0
    plt.figure()
    colors = pylab.cm.cool(np.linspace(0, 1, 8))
    for i in range(indi_ratings.shape[1]):
        key = "{}".format(i)
        print(key)
        end = start + min(len(indi_ratings[0, i]), len(true_model_label) - start)
        true_indi_lbs[key] = true_label[start: start + len(indi_ratings[0, i])]
        true_indi_model_lbs[key] = true_model_label[start: end]

        human_indi_lbs[key] = indi_ratings[0, i][:, 0]
        model_indi_lbs[key] = pred_logits[start: end]
        start = end

        indi_fpr, indi_tpr, _ = metrics.roc_curve(true_indi_lbs[key], human_indi_lbs[key])
        indi_score = metrics.roc_auc_score(true_indi_lbs[key], human_indi_lbs[key])
        mean_fpr.append(indi_fpr)
        mean_tpr.append(indi_tpr)
        mean_score.append(indi_score)

        indi_model_fpr, indi_model_tpr, _ = metrics.roc_curve(true_indi_model_lbs[key], model_indi_lbs[key])
        indi_model_score = metrics.roc_auc_score(true_indi_model_lbs[key], model_indi_lbs[key])
        tpr_temp = interp(base_fpr, indi_model_fpr, indi_model_tpr)
        tpr_model.append(tpr_temp)

        plt.plot(indi_fpr[1], indi_tpr[1], color="r", marker="o", s=10, alpha=0.65)
        # plt.plot(indi_model_fpr, indi_model_tpr, color=colors[i], alpha=0.15, label='model {} AUC:  {:.2f}'.format(i+1, indi_model_score))

    mean_model_tpr = np.mean(np.array(tpr_model), axis=0)
    std_model_tpr = np.std(np.array(tpr_model), axis=0)
    mean_model_score = metrics.auc(base_fpr, mean_model_tpr)

    plt.plot(indi_fpr[1], indi_tpr[1], alpha=0.65, color="r", marker="o", s=10, label="individual radiologists")
    plt.plot(np.mean(np.array(mean_fpr)[:, 1]), np.mean(np.array(mean_tpr)[:, 1]), color="r", marker="d", s=16, label='human average AUC: {:.2f}'.format(np.mean(np.array(mean_score))))
    plt.plot(model_fpr, model_tpr, color="royalblue", linewidth=3.0, label='model AUC:  {:.2f}'.format(mean_model_score))

    plt.title("Receiver Operating Characteristic")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.legend(loc=4)
    plt.ylabel('true positive rate')
    plt.xlabel('false positive rate')
    plt.savefig(os.path.join(data_dir, "Model_with_human_rating_individual_on_certain_0.15_indi_roc.png"), format='png')
    plt.savefig(os.path.join(data_dir, "Model_with_human_rating_individual_on_certain_0.15_indi_roc.pdf"), format='pdf')
    plt.close()


elif plot_name == "human_whole_with_model":
    data_dir = "../data/20190325"
    human_rating = "../data/20190325/human-ratings-20190325-3class_lout40_val_data5-2class.mat"
    original = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_val_data5-2class_human_performance844_with_labels.mat"
    ori = scipy.io.loadmat(original)["DATA"]
    true_label = ori[:, 1]

    # Get model's prediction
    model_res_with_aug = "/home/epilepsy-data/data/metabolites/results/2019-10-09T13-47-36-data-lout40-datas-1d-class2-Res_ECG_CAM-0.766certainEp3-aug_ops_meanx10-0.3-test-auc0.79/AUCs/AUC_curve_step_0.00-auc_0.7905-lout40-datas.csv"
    model_res_wo_aug = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-10-30T16-00-14-data-lout40-datas-1d-class2-Res_ECG_CAM-0.900-aug_ops_meanx0-0-test-0.714/AUCs/AUC_curve_step_0.00-auc_0.7246-lout40-datas.csv"
    model_auc_with_aug = pd.read_csv(model_res_with_aug, header=0).values
    label_with_aug = model_auc_with_aug[:, 0].astype(np.int)
    pred_logits_with_aug = model_auc_with_aug[:, 1]

    model_auc_wo_aug = pd.read_csv(model_res_wo_aug, header=0).values
    label_wo_aug = model_auc_wo_aug[:, 0].astype(np.int)
    pred_logits_wo_aug = model_auc_wo_aug[:, 1]
    # pred_lb = np.argmax(pred_logits, axis=0)

    # Get human's total labels
    hum_whole = scipy.io.loadmat(human_rating)["data_ratings"]
    human_lb = hum_whole[:, 0]
    human_features = hum_whole[:, 1:]
    hum_fpr, hum_tpr, _ = metrics.roc_curve(true_label, human_lb)
    hum_score = metrics.roc_auc_score(true_label, human_lb)

    # PLot human average rating
    plt.figure(figsize=[10, 7])
    plt.plot(hum_fpr[1], hum_tpr[1], 'purple', marker="*", s=10, label='human AUC: {:.2f}'.format(hum_score))

    # Plot trained model prediction
    fpr_with_aug, tpr_with_aug, _ = metrics.roc_curve(label_with_aug, pred_logits_with_aug)
    fpr_wo_aug, tpr_wo_aug, _ = metrics.roc_curve(label_wo_aug, pred_logits_wo_aug)
    score_with_aug = metrics.roc_auc_score(label_with_aug, pred_logits_with_aug)
    score_wo_aug = metrics.roc_auc_score(label_wo_aug, pred_logits_wo_aug)

    plt.plot(fpr_with_aug, tpr_with_aug, 'royalblue', linestyle="-", linewidth=2, label='With aug. AUC: {:.2f}'.format(score_with_aug))
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
    from_dirs = False  # True
    if from_dirs:
        data_dir = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-Res_ECG_CAM"
        model = os.path.basename(data_dir).split("-")[-1]
        exp_mode = os.path.basename(data_dir).split("-")[-2]

        for data_source in ["data5", "data3", "data1", "data2"]:  #, "data7", "data9", "data1", "data3"
            # data_source = "data7"
            pattern = "*-{}-test-*".format(data_source)
            folders = find_folderes(data_dir, pattern=pattern)

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

            # for alpha in [0.05, 0.2, 0.35, 0.5]:
            #     for meth in ["same", "both", "ops"]:
            #         folders = find_folderes(data_dir, pattern=pattern)
            #         configs = [] # "aug_method": [], "aug_factor": [], "aug_fold": [], "from_epoch":
            #         coll_auc = 0
            #         coll_tprs = []
            #         base_fpr = np.linspace(0, 1, 20)
            #         for fn in folders:
            #             print(fn)
            #             splits = os.path.basename(fn).split("-")
            #             aug_name = splits[5]
            #             aug_fold = np.int(splits[6].split("x")[-1])
            #             aug_factor = np.float(splits[8])
            #             from_epoch = np.int(splits[11])
            #             test_auc = np.float(splits[-1])
            #             theta = np.float(splits[-6])
            #             configs.append((aug_name, aug_fold, aug_factor, from_epoch, theta, test_auc))
            #
            #
            #             saved_test_data = find_files(fn, pattern="Data*.txt")
            #             ff = open(saved_test_data[0], 'rb')
            #             data_dict = pickle.load(ff)
            #             labels = data_dict["output_data"]["test_labels"]
            #             logits = data_dict["output_data"]["test_logits"]
            #
            #             fpr, tpr, threshold = metrics.roc_curve(np.argmax(labels,axis=1),  logits[:,1])
            #             tpr_1 = np.interp(base_fpr, fpr, tpr)
            #             tpr_1[0] = 0.0
            #             coll_tprs.append(tpr_1)
            #             coll_auc += metrics.auc(base_fpr, tpr_1)
            #
            #         coll_tprs = np.array(coll_tprs)
            #         mean_tprs_1 = coll_tprs.mean(axis=0)
            #         std_1 = coll_tprs.std(axis=0)
            #         meanauc = coll_auc / len(coll_tprs)
            #
            #         plt.plot(base_fpr, mean_tprs_1, 'g',
            #                  label='AUC={:.3f}'.format(meanauc))
            #         plt.title("theta {}-method {}-ep1-alpha {}".format(0.9, meth, alpha))
            #         plt.savefig(os.path.join(data_dir, "theta {} method {}-auc-{:.3f}-ep1-alpha-{}.png".format(0.9, meth, meanauc, alpha)))
            #         plt.savefig(os.path.join(data_dir, "theta {} method {}-auc-{:.3f}-ep1-alpha-{}.pdf".format(0.9, meth, meanauc, alpha)), format="pdf")
            #         plt.close()
            ## plot same_mean aug, auc w.r.t.
            print("ok")
            configs = np.array(configs)
            aug_same = configs[np.where(configs[:, 0] == aug_name_encode["same"])[0]]
            aug_ops = configs[np.where(configs[:, 0] == aug_name_encode["ops"])[0]]
            aug_both = configs[np.where(configs[:, 0] == aug_name_encode["both"])[0]]

            factor_style = {0.05:"-", 0.2:"-.", 0.35:"--", 0.5:":"}
            meth_color = {"ops":"tab:orange", "same":"tab:blue", "both":"m"}
            markers = {1:"-d", 3:"-*", 5:"-o", 9:"-^"}
            styles = {1:":", 3:"-.", 5:"--", 9:"-"}
            
            # plot aug. method with error bar
            plt.figure(figsize=[12, 8])
            for res, method in zip([aug_same, aug_ops, aug_both], ["same", "ops", "both"]):
                for fold in [1,3,5,9]:
                    fd_configs = np.array(res[np.where(res[:,1] == fold)[0]])
                    if len(fd_configs) > 0:
                        plot_vl = []
                        for scale in [0.05, 0.2, 0.35, 0.5]:
                            print("{}, {}, {}".format(method, fold, scale))
                            if len(np.where(fd_configs[:,2] == scale)[0]) >= 1:
                                plot_vl.append([np.float(scale), np.mean(fd_configs[np.where(fd_configs[:,2] == scale)[0],-1])])

                        plt.plot(np.array(plot_vl)[:,0], np.array(plot_vl)[:,1], markers[fold], label="{}-fold{}-mean-{:.3f}".format(method, fold, np.mean(np.array(plot_vl)[:,1])), color=meth_color[method])
            plt.legend(),
            plt.title("\n".join(wrap("{} with {} on {}".format(exp_mode, model, data_source), 60)))
            plt.savefig(os.path.join(data_dir, "{}-with-{}-on-{}.png".format(exp_mode, model, data_source))),
            plt.savefig(os.path.join(data_dir, "{}-with-{}-on-{}.pdf".format(exp_mode, model, data_source)), format="pdf")
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


            np.savetxt(os.path.join(data_dir, 'model_{}_aug_same_entry_{}-{}+DA-with-{}-on-{}.txt'.format(model, len(aug_same),exp_mode, model, data_source)), aug_same, header="aug_name,aug_fold,aug_factor,cer_th,test_auc", delimiter=",", fmt="%s"),
            np.savetxt(os.path.join(data_dir, 'model_{}_aug_ops_entry_{}-{}+DA-with-{}-on-{}.txt'.format(model, len(aug_same),exp_mode, model, data_source)), aug_ops, header="aug_name,aug_fold,aug_factor,cer_th,test_auc", delimiter=",", fmt="%s"),
            np.savetxt(os.path.join(data_dir, 'model_{}_aug_both_entry_{}-{}+DA-with-{}-on-{}.txt'.format(model, len(aug_same),exp_mode, model, data_source)), aug_both, header="aug_name,aug_fold,aug_factor,cer_th,test_auc", delimiter=",", fmt="%s"),
            np.savetxt(os.path.join(data_dir, 'model_{}_all_different_config_theta{}-{}+DA-with-{}-on-{}.txt'.format(model, len(aug_same),exp_mode, model, data_source)), configs, header="aug_name,aug_fold,aug_factor,cer_th,test_auc", delimiter=",", fmt="%s")
    else:
        file_dir = "C:/Users/LDY/Desktop/1-all-experiment-results/metabolites/auc-func-as-augmentation-parameters/model_Res_ECG_CAM_all_different_config_theta5-DA+DA-with-Res_ECG_CAM-on-data5.txt"
        aug_meth = ["same", "ops", "both"]
        configs = pd.read_csv(file_dir, header=0).values
        aug_name_encode = {"same": 0, "ops": 1,"both": 2}

        configs = np.array(configs)
        aug_same = configs[np.where(configs[:, 0] == aug_name_encode["same"])[0]]
        aug_ops = configs[np.where(configs[:, 0] == aug_name_encode["ops"])[0]]
        aug_both = configs[np.where(configs[:, 0] == aug_name_encode["both"])[0]]

        factor_style = {0.05:"-", 0.2:"-.", 0.35:"--", 0.5:":"}
        meth_color = {"ops":"tab:orange", "same":"tab:blue", "both":"m"}
        markers = {1:"-d", 3:"-*", 5:"-o", 9:"-^"}
        styles = {1:":", 3:"-.", 5:"--", 9:"-"}
        
        # plot aug. method with error bar
        plt.figure(figsize=[12, 8])
        for res, method in zip([aug_same, aug_ops, aug_both], ["same", "ops", "both"]):
            for fold in [1,3,5,9]:
                fd_configs = np.array(res[np.where(res[:,1] == fold)[0]])
                if len(fd_configs) > 0:
                    plot_vl = []
                    for scale in [0.05, 0.2, 0.35, 0.5]:
                        print("{}, {}, {}".format(method, fold, scale))
                        if len(np.where(fd_configs[:,2] == scale)[0]) >= 1:
                            plot_vl.append([np.float(scale), np.mean(fd_configs[np.where(fd_configs[:,2] == scale)[0],-1])])

                    plt.plot(np.array(plot_vl)[:,0], np.array(plot_vl)[:,1], markers[fold], label="{}-fold{}-mean-{:.3f}".format(method, fold, np.mean(np.array(plot_vl)[:,1])), color=meth_color[method])
        plt.legend(),
        plt.title("\n".join(wrap("{} with {} on {}".format(exp_mode, model, data_source), 60)))
        plt.savefig(os.path.join(data_dir, "{}-with-{}-on-{}-best-same-fold3-scale0.5-both.png".format(exp_mode, model, data_source))),
        plt.savefig(os.path.join(data_dir, "{}-with-{}-on-{}-best-same-fold3-scale0.5-both.pdf".format(exp_mode, model, data_source)), format="pdf")
        plt.close()

        for vars in [[None, epoch, factor], [fold, None, factor], [fold, epoch, None]]:

            get_auc_as_factor(file_dir, fold=vars[0], epoch=vars[1], factor=vars[2], aug_meth=aug_meth, colors=colors)

    # for fold in [5, 10]:
    #     same = aug_same[np.where(aug_same[:, 1].astype(np.int) == fold)[0]]
    #     ops = aug_ops[np.where(aug_ops[:, 1].astype(np.int) == fold)[0]]
    #     both = aug_both[np.where(aug_both[:, 1].astype(np.int) == fold)[0]]
    #
    #     epoch_counter_same = list(Counter(list(same[:, 3])))
    #     epoch_counter_ops = list(Counter(list(ops[:, 3])))
    #     epoch_counter_both = list(Counter(list(both[:, 3])))
    #
    #     plt.figure(),
    #     for epo in [3, 4, 5, 6]:
    #         for data, name in zip([same, ops, both], ["same", "ops", "both"]):
    #             subdata = data[np.where(data[:, 3].astype(np.int) == epo)[0]]
    #             sortdata = np.array(sorted(zip(subdata[:, 2], subdata[:, 4], subdata[:, 1]), key=lambda x: x[0]))
    #
    #             if len(subdata) > 0:
    #                 plt.plot(sortdata[:, 0].astype(np.float), sortdata[:, 1].astype(np.float), label="{} class".format(name)),
    #         plt.title("Augment by {} fold from epoch {}".format(fold, epo))
    #         plt.legend(),
    #         plt.savefig(os.path.join(results, "model-{}-Augment by {} fold from epoch {}.png".format(model, fold, epo)))
    #         plt.close()


elif plot_name == "rename_test_folders":
    results = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-Res_ECG_CAM"
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


elif plot_name == "get_performance_metrices":
    get_scaler_performance_metrices()


elif plot_name == "move_folder":
    import shutil
    # target = "C:/\Users\\LDY\\Desktop\\metabolites-0301\\metabolites_tumour_classifier\\dest"
    target = "C:/Users/LDY/Desktop/metabolites-0301/metabolites_tumour_classifier/results/2019"
    need2move = [
        "C:/Users/LDY/Desktop/metabolites-0301/metabolites_tumour_classifier/results/2020-03-26T13-31-21-class2-Res_ECG_CAM-0.776-aug_ops_meanx5-0.8-train"
    ]
    for fd in need2move:
        # files = os.listdir(fd)
        print(fd)
        new_dest = os.path.join(target, os.path.basename(fd))
        if not os.path.isdir(new_dest):
            shutil.copytree(fd, new_dest)


elif plot_name == "certain_tsne_distillation":
    from scipy.io import loadmat as loadmat
    import scipy.io as io
    from scipy.stats import ks_2samp

    pattern = "full_summary-*.csv"
    data_dir = "C:\\Users\\LDY\\Desktop\\metabolites-0301\\metabolites_tumour_classifier\\data"
    # data_dir = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T22-07-37--Res_ECG_CAM-nonex0-factor-0-from-data5-certainFalse-theta-0-s989-100rns-train"
    # ori_data_dir = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/DATA.mat"
    ori_data_dir = "C:\\Users\\LDY\\Desktop\\metabolites-0301\\metabolites_tumour_classifier\\data\\20190325\\20190325-3class_lout40_train_test_data5.mat"

    # certain_fns = find_files(data_dir, pattern=pattern)
    # whole_data, distill_data = get_data_from_certain_ids(certain_fns[0], mat_file=ori_data_dir)
    whole_data = get_data_from_mat(ori_data_dir)
    data_source = "lout5"
    # data_source = data_dir.split("-")[-7]
    reduction_method = "tsne"
    if_save_data = False
    # data_dir = os.path.dirname(ori_data_dir)
    ## get the whole tsne projection
    if reduction_method == "tsne":
        if if_save_data:
            from bhtsne import tsne as TSNE
            reduced_proj_whole = TSNE(whole_data[:, 3:], dimensions=2)
            reduced_proj_distill= TSNE(distill_data[:, 3:], dimensions=2)
            # np.savetxt(os.path.join(os.path.dirname(ori_data_dir), "{}-whole_data-2d.csv".format(reduction_method)), reduced_proj_whole, fmt="%.5f", delimiter=","),
            np.savetxt(os.path.join(data_dir, "{}-whole_data-2d.csv".format(reduction_method)), reduced_proj_whole, fmt="%.5f", delimiter=","),
            np.savetxt(os.path.join(data_dir, "{}-distill_data-2d.csv".format(reduction_method)), reduced_proj_distill, fmt="%.5f", delimiter=",")
        else:
            filename_whole = os.path.join(data_dir, "{}-whole_data-lout5-2d.csv".format(reduction_method))
            reduced_proj_whole = pd.read_csv(filename_whole, header=None).values
            # filename_distill = os.path.join(data_dir, "distill_data_tsne-2d-from-whole-tsne.csv.csv".format(reduction_method))
            # reduced_proj_distill = pd.read_csv(filename_distill, header=None).values
    elif reduction_method == "UMAP":
        if if_save_data:
            import umap.umap_ as umap
            reduced_proj_whole = umap.UMAP(random_state=42).fit_transform(whole_data[:, 3:])
            reduced_proj_distill = umap.UMAP(random_state=42).fit_transform(distill_data[:, 3:])
            np.savetxt(os.path.join(data_dir, "{}-whole_data-2d.csv".format(reduction_method)), reduced_proj_whole, fmt="%.5f", delimiter=","),
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
            reduced_proj_whole = MDS_whole.fit_transform(whole_data[:, 3:])
            reduced_proj_distill = MDS_distill.fit_transform(distill_data[:, 3:])
            np.savetxt(os.path.join(data_dir, "{}-whole_data-2d.csv".format(reduction_method)), reduced_proj_whole, fmt="%.5f", delimiter=","),
            np.savetxt(os.path.join(data_dir, "{}-distill_data-2d-from-whole.csv".format(reduction_method)), reduced_proj_distill, fmt="%.5f", delimiter=",")
        else:
            filename_whole = os.path.join(data_dir, "{}-whole_data-2d.csv".format(reduction_method))
            filename_distill = os.path.join(data_dir, "{}-distill_data-2d-from-whole.csv".format(reduction_method))
            reduced_proj_whole = pd.read_csv(filename_whole, header=None).values
            reduced_proj_distill = pd.read_csv(filename_distill, header=None).values

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
    plt.legend(scatterpoints=4, loc=3)
    plt.title("{} of both classes (whole)".format(reduction_method))
    plt.xlabel("dimension #1 (p={:.2E})".format(p_x_whole)),
    plt.ylabel("dimension #2 (p={:.2E})".format(p_y_whole))
    plt.savefig(os.path.join(data_dir,  "{}-whole-{}.png".format(reduction_method, data_source))),
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


elif plot_name == "plot_metabolites":
    from scipy.io import loadmat as loadmat
    import scipy.io as io

    data_dir = "C:/Users/LDY/Desktop/testestestestest/DTC"
    file_patterns = "*.csv"

    # mat_file = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/DATA.mat"
    mat_file = "C:\\Users\\LDY\\Desktop\\metabolites-0301\\metabolites_tumour_classifier\\data\\DATA.mat"
    
    original_data = get_data_from_mat(mat_file)
    
    # get num of spectra distribution among patients
    stat_num_per_pat = Counter(original_data[:, 1])
    stat_num_per_pat.items()
    num_spectra = np.array([vl for _, vl in stat_num_per_pat.items()])
    plt.hist(num_spectra, bins=100)
    plt.title("Distribution of number of spectra in patients")
    plt.xlabel("number of voxels")
    plt.ylabel("patient counts")
    plt.vlines(np.percentile(num_spectra, 90), 0, 40, label="90th")
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(mat_file), "Distribution of number of spectra in patients-whole-data2.png")),
    plt.savefig(os.path.join(os.path.dirname(mat_file), "Distribution of number of spectra in patients-whole-data2.pdf"))
    plt.close()
    
    for pat_id, num in stat_num_per_pat.items():
        plt.figure()
        pat_inds = np.where(original_data[:, 1] == pat_id)[0]
        assert num == len(pat_inds)
        label = np.mean(original_data[pat_inds, 2])
        plt.plot(original_data[pat_inds, 3:].T)
        plt.title("Patient({})-lb{}-num({})".format(pat_id, label, num))
        plt.xlabel("metabolite index")
        plt.ylabel("norm. amplitude")
        plt.savefig(os.path.join(os.path.dirname(mat_file), "plot-spectra-patient-wise", "Spectra-Patient-{}-lb{}-({}).png".format(pat_id, label, num))),
        plt.savefig(os.path.join(os.path.dirname(mat_file), "plot-spectra-patient-wise", "Spectra-Patient-{}-lb{}-({}).pdf".format(pat_id, label, num))),
        plt.close()
    
    
    
    
    labels = original_data[:, 1]

    train_data = {}
    test_data = {}
    pat_ids = original_data[:, 0].astype(np.int)

    need_inds = np.where(pat_ids == 6)[0]
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
    class_dark = ["darkblue", "crimson"]

    files = find_files(data_dir, pattern=file_patterns)
    # certain_mat = np.empty((0, new_mat.shape[1]))
    certain_inds_tot = np.empty((0))
    for fn in files:
        certain = pd.read_csv(fn, header=0).values
        certain_inds = certain[:, 0].astype(np.int)
        certain_inds_tot = np.append(certain_inds_tot, certain_inds)
        print(os.path.basename(fn), len(certain_inds), "samples\n")

    uniq_inds = np.unique(certain_inds_tot).astype(np.int)
    certain_mat = new_mat[uniq_inds]

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
                pat_ids = values[:, 1].astype(np.int)
                lbs = values[:, 2]
                prob = values[:, 3:]
                if ind == 0:  #the first file to get the total number (3-class) of samples
                    total_num = num_smp_dataset[data_source]  #3-class samples id
                    ids_w_count = []
                    certain_w_count = []
                    certain_w_corr_count = []
                    dict_count = {key: 0 for key in np.arange(total_num)}  #total number 9243
                    dict_count_certain = {key: 0 for key in np.arange(total_num)}
                    dict_corr_count_certain = {key: 0 for key in np.arange(total_num)}
                    
                pred_lbs = np.argmax(prob, axis=1)
                right_inds = np.where(pred_lbs == lbs)[0]
                correct = np.unique(smp_ids[right_inds])
                ids_w_count += list(correct)
                
                # Get certain with differnt threshold
                if theta > 1:  #percentile
                    larger_prob = [pp.max() for pp in prob]
                    threshold = np.percentile(larger_prob, theta)
                    slc_ratio = 1 - theta / 100.
                else:  # absolute prob. threshold
                    threshold = theta
                    slc_ratio = 1 - theta
                ct_smp_ids = np.where([prob[i] > threshold for i in range(len(prob))])[0]
                ct_corr_inds = ct_smp_ids[np.where(lbs[ct_smp_ids] == pred_lbs[ct_smp_ids])[0]]
                certain_w_count += list(np.unique(smp_ids[ct_smp_ids]))
                certain_w_corr_count += list(np.unique(smp_ids[ct_corr_inds]))
                num_certain = len(ct_smp_ids)
    
            count_all = Counter(ids_w_count)
            dict_count.update(count_all)
    
            dict_count_certain.update(Counter(certain_w_count))
            dict_corr_count_certain.update(Counter(certain_w_corr_count))
            
            # if theta == 0.975:
            #     ipdb.set_trace()
            counter_array = np.array([[key, val] for (key, val) in dict_count.items()])
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
            plt.savefig(data_dir+"/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}-({}_theta-{}).png".format(os.path.basename(files[0]).split("_")[7], total_num, data_source, num_certain, theta)),
            plt.savefig(data_dir+"/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}-({}_theta-{}).pdf".format(os.path.basename(files[0]).split("_")[7], total_num, data_source, num_certain, theta), format="pdf")
            print("ok")
            plt.close()
            
            num2select = np.int(np.int(os.path.basename(files[0]).split("_")[7]) * slc_ratio)
            ct_concat_data = np.concatenate((np.array(sort_inds).reshape(-1,1)[-num2select:], rates.reshape(-1,1)[-num2select:], ct_sele_rates.reshape(-1, 1)[-num2select:], ct_corr_rates.reshape(-1, 1)[-num2select:]), axis=1)
            np.savetxt(data_dir+"/certain_{}_({}-{})-({}_theta-{}).csv".format(data_source, os.path.basename(files[0]).split("_")[7], total_num, num2select, theta), ct_concat_data, fmt="%.5f", delimiter=",", header="ori_sort_rate_id,ori_sort_rate,certain_sele_rate,certain_corr_rate")
    concat_data = np.concatenate((np.array(sort_inds).reshape(-1,1), rates.reshape(-1,1), ct_sele_rates.reshape(-1, 1), ct_corr_rates.reshape(-1, 1)), axis=1)
    np.savetxt(data_dir+"/full_summary-{}_100_runs_sort_inds_rate_({}-{}).csv".format(data_source, os.path.basename(files[0]).split("_")[7], total_num), concat_data, fmt="%.5f", delimiter=",", header="ori_sort_rate_id,ori_sort_rate,certain_sele_rate,certain_corr_rate")


elif plot_name == "100_single_ep_corr_classification_rate_mnist":
    """
    Get the correct classification rate with 100 runs of single-epoch-training
    """
    import ipdb
    from scipy.stats import spearmanr

    data_dirs = [
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-10T15-14-50-MLP-nonex0-factor-0-from-ep-0-from-lout40-mnist-theta-None-s129-100rns-noise-ratio0.2-train/certains",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-10T15-52-32-MLP-nonex0-factor-0-from-ep-0-from-lout40-mnist-theta-None-s129-100rns-noise-ratio0.8-train",
    ]
    num_smp_dataset = {"data0": 8357, "data1": 8326, "data2": 8566,
                       "data3": 8454, "data4": 8440, "data5": 8231,
                       "data6": 8371, "data7": 8357, "data8": 8384, "data9": 7701,
                       "mnist": 70000}

    for data_dir in data_dirs:
        files = find_files(data_dir, pattern="one_ep_data_train*.csv")

        spearmanr_rec = []
        # get correct count in 100 rauns
        data_source = os.path.basename(files[0]).split("_")[8]
        for ind in range(len(files)):
            fn = find_files(data_dir, pattern="one_ep_data_train_epoch_{}*.csv".format(ind))
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

        # if theta == 0.975:
        #     ipdb.set_trace()
        counter_array = np.array([[key, val] for (key, val) in dict_count.items()])
        noisy_inds_array = np.array([[key, val/len(files)] for (key, val) in noisy_lb_rec.items()])
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
                os.path.basename(files[0]).split("_")[7], total_num, data_source)),
        plt.savefig(
            data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}.pdf".format(
                os.path.basename(files[0]).split("_")[7], total_num, data_source), format="pdf")
        print("ok")
        plt.close()

    concat_data = np.concatenate((
                                 np.array(sort_inds).reshape(-1, 1), rates.reshape(-1, 1)), axis=1)
    np.savetxt(data_dir + "/full_summary-{}_100_runs_sort_inds_rate_({}-{}).csv".format(data_source, os.path.basename(
        files[0]).split("_")[7], total_num), concat_data, fmt="%.5f", delimiter=",",
               header="ori_sort_rate_id,ori_sort_rate,true_lbs,noisy_lbs")


elif plot_name == "100_single_ep_corr_classification_rate":
    """
    Get the correct classification rate with 100 runs of single-epoch-training
    """
    import ipdb
    from scipy.stats import spearmanr

    data_dirs = [
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2020-11-18T10-23-19--Inception-nonex0-factor-0-from-data5-certainFalse-theta-0-s789-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2020-11-18T10-23-20--Inception-nonex0-factor-0-from-data1-certainFalse-theta-0-s789-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2020-11-18T10-23-21--Inception-nonex0-factor-0-from-data2-certainFalse-theta-0-s789-100rns-train"
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
            pat_ids = values[:, 1].astype(np.int)
            lbs = values[:, 2]
            prob = values[:, 3:]
            if ind == 0:  # the first file to get the total number (3-class) of samples
                total_num = num_smp_dataset[data_source]  # 3-class samples id
                ids_w_count = []
                dict_count = {key: 0 for key in np.arange(total_num)}  #

            pred_lbs = np.argmax(prob, axis=1)
            right_inds = np.where(pred_lbs == lbs)[0]
            correct = np.unique(smp_ids[right_inds])
            ids_w_count += list(correct)

        count_all = Counter(ids_w_count)
        dict_count.update(count_all)

        # if theta == 0.975:
        #     ipdb.set_trace()
        counter_array = np.array([[key, val] for (key, val) in dict_count.items()])
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
        plt.savefig(
            os.path.dirname(data_dir) + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}.png".format(
                os.path.basename(files[0]).split("_")[7], total_num, data_source)),
        plt.savefig(
            os.path.dirname(data_dir) + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}.pdf".format(
                os.path.basename(files[0]).split("_")[7], total_num, data_source), format="pdf")
        print("ok")
        plt.close()

        concat_data = np.concatenate((
                                     np.array(sort_inds).reshape(-1, 1), rates.reshape(-1, 1)), axis=1)
        np.savetxt(os.path.dirname(data_dir) + "/full_summary-{}_100_runs_sort_inds_rate_({}-{}).csv".format(data_source, os.path.basename(
            files[0]).split("_")[7], total_num), concat_data, fmt="%.5f", delimiter=",",
                   header="sort_samp_ids,sort_corr_rate")


elif plot_name == "100_single_ep_patient_wise_rate":
    # load original data to get patient-wise statistics
    from scipy.io import loadmat as loadmat
    import scipy.io as io

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
    pat_ids = original_data[:, 0].astype(np.int)

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
    # Nenad validated the labels of these samples
    m_file = "C:/Users/LDY/Desktop/all-experiment-results/metabolites/20190325-certain-Validate.mat"
    mat = scipy.io.loadmat(m_file)["Validate"]  # [id, label, features]
    samp_ids = mat[:, 0]
    pat_ids = mat[:, 1]
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
    for sp_id, pat, lb, r_or_w in zip(samp_ids, pat_ids, labels, corr_or_wrong):
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
    src_data = ["data{}".format(jj) for jj in [0,1,2,3,4,6,7,8,9]]
    data_source_dirs = [
        "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_val_{}.mat".format(
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
    split_parts = {key: {} for key in range(5)}
    part_len = len(coll_mat) // 5
    for ii in range(4):
        split_parts[ii]["DATA"] = coll_mat[ii*part_len: (ii+1)*part_len]
        num_pat = len(Counter(split_parts[ii]["DATA"][:, 0]))
        scipy.io.savemat(
            os.path.join(os.path.dirname(dd),
                         "5_fold_20190325-3class[{}-{}-{}]_pat_{}_test_data{}.mat".format(
                             np.sum(split_parts[ii]["DATA"][:, 1] == 0),
                             np.sum(split_parts[ii]["DATA"][:, 1] == 1),
                             np.sum(split_parts[ii]["DATA"][:, 1] == 2), num_pat, ii)), split_parts[ii])

    split_parts[4]["DATA"] = coll_mat[4 * part_len: ]
    num_pat = len(Counter(split_parts[4]["DATA"][:, 0]))
    ii = 4
    scipy.io.savemat(
        os.path.join(os.path.dirname(dd),
                     "5_fold_20190325-3class[{}-{}-{}]_pat_{}_test_data{}.mat".format(
                         np.sum(split_parts[ii]["DATA"][:, 1] == 0),
                         np.sum(split_parts[ii]["DATA"][:, 1] == 1),
                         np.sum(split_parts[ii]["DATA"][:, 1] == 2), num_pat, ii)), split_parts[ii])


    # merge the other 4 sets to form train_validation set
    for jj in range(5):
        train_inds = list(np.arange(5))
        del train_inds[jj]
        train_coll = {"DATA": np.empty((0, 290))}
        for ii in train_inds:
            train_coll["DATA"] = np.vstack((train_coll["DATA"], split_parts[ii]["DATA"]))
        num_pat = len(Counter(train_coll["DATA"][:, 0]))
        print("ok")
        scipy.io.savemat(
            os.path.join(os.path.dirname(dd),
                         "5_fold_20190325-3class[{}-{}-{}]_pat_{}_train_val_data{}.mat".format(
                             np.sum(train_coll["DATA"][:, 1] == 0),
                             np.sum(train_coll["DATA"][:, 1] == 1),
                             np.sum(train_coll["DATA"][:, 1] == 2), num_pat, jj)), train_coll)



    








