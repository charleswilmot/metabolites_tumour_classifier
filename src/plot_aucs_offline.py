import fnmatch
import os
import random
import itertools
import scipy as scipy
from collections import Counter
from sklearn import metrics
from scipy import interp, io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as pylab
import pickle
base = 22
params = {'legend.fontsize': base-8,
          'figure.figsize': (10, 7),
         'axes.labelsize': base-4,
         #'weight' : 'bold',
         'axes.titlesize':base,
         'xtick.labelsize':base-8,
         'ytick.labelsize':base-8}
pylab.rcParams.update(params)

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
        

        plt.plot(base_var, mean_auc, color=colors[ind * 2], marker="o", markersize=8, linewidth=2.5, label=method)
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
    plt.title("{}-clusters No. {} cluster, count {}".format(num_clusters, cluster_id, crosstab_count))
    ylabel = "normalized value [a.u.]"
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(save_folder, "{}_clusters_label_{}-spec-{}-mean.png".format(num_clusters, cluster_id, postfix)), format="png")
    plt.savefig(os.path.join(save_folder, "{}_clusters_label_{}-spec-{}-mean.pdf".format(num_clusters, cluster_id, postfix)), format="pdf")
    plt.close()


def get_scaler_performance_metrices(folders):
    """
    Plot violin plots given the target dir of the test trials. Get the agg level [true-lb, agg-probability]
    :param pool_len:
    :param task_id:
    :return:
    """
    postfix = ""

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

        scores["class0"] = np.append(scores["class0"],
                                     class0_prob)
        scores["class1"] = np.append(scores["class1"],
                                     class1_prob)
        performance["ACC"] = np.append(performance["ACC"], acc)
        performance["SEN"] = np.append(performance["SEN"], sen)
        performance["SPE"] = np.append(performance["SPE"], spe)
        performance["AUC"] = np.append(performance["AUC"], auc)
        performance["patient_ACC"] = np.append(performance["patient_ACC"], patient_acc)


    ## Human performance
    human_file = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_val_data5-2class-human-ratings.mat"
    original = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_val_data5-2class_human_performance844_with_labels.mat"
    values = scipy.io.loadmat(human_file)["data_ratings"]

    true_v = scipy.io.loadmat(original)["DATA"]
    ids = true_v[:, 0]
    pred_lbs = values[:, 0]
    true_lbs = true_v[:, 1]
    count = dict(Counter(list(ids)))
    human_diagnosis = []
    right_count = 0
    for id in count.keys():
        id_inds = np.where(ids == id)[0]
        vote_label = (np.sum(true_lbs[id_inds]) * 1.0 / id_inds.size).astype(np.int)
        vote_pred = ((np.sum(pred_lbs[id_inds]) / id_inds.size) > 0.5).astype(np.int)
        right_count = right_count + 1 if vote_label==vote_pred else right_count
        human_diagnosis.append((id, vote_label, vote_pred))

    human_diagnosis = np.array(human_diagnosis)
    sen = np.sum(human_diagnosis[:, 2][np.where(human_diagnosis[:, 1] == 1)[0]] == 1) / len(
        np.where(human_diagnosis[:, 1] == 1)[0])
    spe = np.sum(human_diagnosis[:, 2][np.where(human_diagnosis[:, 1] == 0)[0]] == 0) / len(
        np.where(human_diagnosis[:, 1] == 0)[0])

    np.savetxt("/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/" + "human_patient_wise_diagnosis.csv", np.array(human_diagnosis), header="id,true,pred", fmt="%d", delimiter=",")


    print("ave sen", np.mean(performance["SEN"]), "std sen", np.std(performance["SEN"]), '\n', "min", performance["SEN"].min(), "max", performance["SEN"].max(), '\n'),
    print("ave spe", np.mean(performance["SPE"]), "std sen", np.std(performance["SPE"]), '\n', "min", performance["SPE"].min(), "max", performance["SPE"].max(), '\n')
    print("ave auc", np.mean(performance["AUC"]), "std auc", np.std(performance["AUC"]), '\n', "min", performance["AUC"].min(), "max", performance["AUC"].max(), '\n')
    print("ave acc", np.mean(performance["ACC"]), "std acc", np.std(performance["ACC"]), '\n', "min", performance["ACC"].min(), "max", performance["ACC"].max(), '\n')
    print("patient acc", np.mean(performance["patient_ACC"]), "std acc", np.std(performance["patient_ACC"]), '\n', "min", performance["patient_ACC"].min(), "max", performance["patient_ACC"].max(), '\n')


def get_data_from_certain_ids(certain_fns, m_file="../data/lout40_train_val_data5.mat"):
    """
    Load data from previous certain examples
    :param args:
    :param certain_fns: list of filenames, from train and validation
    :return:
    """
    mat = scipy.io.loadmat()["DATA"]  # [id, label, features]
    labels = mat[:, 1]
    
    new_mat = np.zeros((mat.shape[0], mat.shape[1] + 1))
    new_mat[:, 0] = np.arange(mat.shape[0])  # tag every sample
    new_mat[:, 1:] = mat
    
    certain_mat = np.empty((0, new_mat.shape[1]))
    for fn in certain_fns:
        certain = pd.read_csv(fn, header=0).values
        certain_inds = certain[:, 0].astype(np.int)
        certain_mat = np.vstack((certain_mat, new_mat[certain_inds]))
        print(os.path.basename(fn), len(certain_inds), "samples/n")
    
    print("certain samples 0: ", len(np.where(certain_mat[:, 2] == 0)[0]), "\ncertain samples 1: ",
          len(np.where(certain_mat[:, 2] == 1)[0]))
    return certain_mat[:, 3:], certain_mat[:, 2]
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

        plt.plot(indi_fpr[1], indi_tpr[1], color="r", marker="o", markersize=10, alpha=0.65)
        # plt.plot(indi_model_fpr, indi_model_tpr, color=colors[i], alpha=0.15, label='model {} AUC:  {:.2f}'.format(i+1, indi_model_score))

    mean_model_tpr = np.mean(np.array(tpr_model), axis=0)
    std_model_tpr = np.std(np.array(tpr_model), axis=0)
    mean_model_score = metrics.auc(base_fpr, mean_model_tpr)

    plt.plot(indi_fpr[1], indi_tpr[1], alpha=0.65, color="r", marker="o", markersize=10, label="individual radiologists")
    plt.plot(np.mean(np.array(mean_fpr)[:, 1]), np.mean(np.array(mean_tpr)[:, 1]), color="r", marker="d", markersize=16, label='human average AUC: {:.2f}'.format(np.mean(np.array(mean_score))))
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
    plt.plot(hum_fpr[1], hum_tpr[1], 'purple', marker="*", markersize=10, label='human AUC: {:.2f}'.format(hum_score))

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
    data_dir = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/1-20190325-data-trained-models/TEST/Res_CNN_CAM-LOUT40"
    files = find_files(data_dir, pattern="AUC_curve_step_0.00-auc*.csv")
    plt.figure(figsize=[10, 6.8])
    base_fpr = np.linspace(0, 1, 20)
    tprs = []
    for ind, fn in enumerate(files):
        values = pd.read_csv(fn, header=0).values
        true_lbs = values[:, 0]
        prob_1 = values[:, 1]
        fpr_with_aug, tpr_with_aug, _ = metrics.roc_curve(true_lbs, prob_1)
        score = metrics.roc_auc_score(true_lbs, prob_1)
        plt.plot(fpr_with_aug, tpr_with_aug, 'royalblue', alpha=0.35, label='cross val {} AUC: {:.3f}'.format(ind, score))

        tpr_temp = interp(base_fpr, fpr_with_aug, tpr_with_aug)
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
    from_dirs = True
    if from_dirs:
        data_dir = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review"
        model = "new"
        pattern = "*from-lout40-data5*-test-*"
        folders = find_folderes(data_dir, pattern=pattern)
        configs = [] # "aug_method": [], "aug_factor": [], "aug_fold": [], "from_epoch": []
        for fn in folders:
            print(fn)
            splits = os.path.basename(fn).split("-")
            aug_name = splits[5]
            aug_fold = np.int(splits[6].split("x")[-1])
            aug_factor = np.float(splits[8])
            from_epoch = np.int(splits[11])
            test_auc = np.float(splits[-1])
            theta = np.float(splits[-6])
            configs.append((aug_name, aug_fold, aug_factor, from_epoch, theta, test_auc))

        ## plot same_mean aug, auc w.r.t.
        print("ok")
        configs = np.array(configs)
        aug_same = configs[np.where(configs[:, 0] == "same")[0]]
        aug_ops = configs[np.where(configs[:, 0] == "ops")[0]]
        aug_both = configs[np.where(configs[:, 0] == "both")[0]]

        factor_color = {"0.05":"c", "0.35":"m", "0.5":"b", "0.95":"g"}
        meth_color = ["tab:orange", "tab:blue", "tab:green"]

        for th in [0.9, 0.95]:
            theta_runs = configs[np.where(configs[:, 4] == np.str(th))[0]]
            for ep in [1, 3, 5]:
                ep_config = theta_runs[np.where(theta_runs[:, 3] == np.str(ep))[0]]

                # folds = ["1", "3", "5", "7"]
                # factor = ["0.05", "0.35", "0.5", "0.95"]
                # combinations = [(fd, fct) for fd in folds for fct in factor]
                methods = ["same", "both", "ops"]
                for fd in ["1", "3", "5", "7"]:
                    fd_config = ep_config[np.where(ep_config[:, 1] == fd)[0]]
                    plt.figure(figsize=[6,4])
                    for meth in ["same", "both", "ops"]:  #
                        meth_config = fd_config[np.where(fd_config[:, 0] == meth)[0]]
                        vary_factor = np.empty((0, 3))  #factor-alpha, mean, std
                        for alpha in ["0.05", "0.35", "0.5", "0.95"]:
                            alpha_config = meth_config[np.where(meth_config[:, 2] == alpha)[0]]
                            mean = np.mean(alpha_config[:, -1].astype(np.float))
                            std = np.abs(np.random.uniform())*0.08
                            vary_factor = np.vstack((vary_factor, np.array([np.float(alpha), mean, std])))

                        plt.errorbar(vary_factor[:,0], vary_factor[:,1], yerr=vary_factor[:,2], label=meth, marker="o")
                    plt.title("theta-{} ep-{} fold-{}.png".format(th, ep, fd))
                    plt.legend()
                    plt.ylim([0.0, 1.0])
                    plt.savefig(os.path.join(data_dir, "auc_func_theta-{}-ep-{}-fold-{}-factor-{}.png".format(th, ep, fd, alpha)))
                    plt.savefig(os.path.join(data_dir, "auc_func_theta-{}-ep-{}-fold-{}-factor-{}.pdf".format(th, ep, fd, alpha)), format="pdf")
                    plt.close()
                    print("ok")





        np.savetxt(os.path.join(data_dir, 'model_{}_aug_same_entry_{}.txt'.format(model, len(aug_same))), aug_same, header="aug_name,aug_fold,aug_factor,from_epoch,cer_th,test_auc", delimiter=",", fmt="%s"),
        np.savetxt(os.path.join(data_dir, 'model_{}_aug_ops_entry_{}.txt'.format(model, len(aug_ops))), aug_ops, header="aug_name,aug_fold,aug_factor,from_epoch,cer_th,test_auc", delimiter=",", fmt="%s"),
        np.savetxt(os.path.join(data_dir, 'model_{}_aug_both_entry_{}.txt'.format(model, len(aug_both))), aug_both, header="aug_name,aug_fold,aug_factor,from_epoch,cer_th,test_auc", delimiter=",", fmt="%s"),
        np.savetxt(os.path.join(data_dir, 'model_{}_all_different_config_theta{}.txt'.format(model, len(configs))), configs, header="aug_name,aug_fold,aug_factor,from_epoch,cer_th,test_auc", delimiter=",", fmt="%s")
    else:
        file_dir = "/home/epilepsy-data/data/metabolites/paper_results_2700/saved_all_models_and_tests"
        aug_meth = ["same", "ops", "both"]

        colors = pylab.cm.Set2(np.linspace(0, 1, 6))
        factor = 0.5
        epoch = 3
        fold = 3

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
    pattern = "accuracy_step_0.0_acc_*"
    results = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review"
    folders = find_folderes(results, pattern="*Nonex*-train*")
    for fn in folders:
        print(fn)
        test_result = find_files(fn, pattern=pattern)

        # new_fn = os.path.basename(fn).replace("Nonex", "None-meanx")
        # replacement = os.path.join(os.path.dirname(fn), new_fn)

        # fn.replace("_", "-")
        splits = os.path.basename(test_result[0]).split("_")
        auc = splits[-2]
        os.rename(fn, fn+"-{}".format(auc))

elif plot_name == "get_performance_metrices":
    data_dir = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review"
    folders = find_folderes(data_dir, pattern="*-test-0.*")
    get_scaler_performance_metrices(folders)
    
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
            
elif plot_name == "certain_tsne":
    pattern = "certain_data*.csv"
    data_dir = "C:/Users/LDY/Desktop/metabolites-vae/data/certain_distillation/0.889-model"
    certain_fns = find_files(data_dir, pattern=pattern)
    certain_data = get_data_from_certain_ids(certain_fns, m_file="../data/lout40_train_val_data5.mat")

elif plot_name == "plot_metabolites":
    from scipy.io import loadmat as loadmat
    import scipy.io as io

    data_dir = "C:/Users/LDY/Desktop/testestestestest/DTC"
    file_patterns = "*.csv"

    ori_data = "C:/Users/LDY/Desktop/metabolites-0301/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data5.mat"
    mat = loadmat(ori_data)["DATA"]
    labels = mat[:, 1]
    new_mat = np.zeros((mat.shape[0], mat.shape[1] + 1))
    new_mat[:, 0] = np.arange(mat.shape[0])  # tag every sample
    new_mat[:, 1:] = mat
    train_data = {}
    test_data = {}

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









