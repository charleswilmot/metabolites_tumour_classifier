import fnmatch
import os
import random
import itertools
import scipy as scipy
from collections import Counter
from sklearn import metrics
from scipy import interp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as pylab
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


def get_auc_as_factor(epoch=5, factor=0.5, aug_meth=["same", "ops"], colors=pylab.cm.Set2(np.linspace(0, 1, 6)), fix_name="epoch"):
    """
    Get auc as a function of aug factor with three different aug methods
    header = np.array(["method", "fold", "factor", "epoch", "auc"])
    :param epoch:
    :param aug_meth:
    :return:
    """
    colors = ["royalblue", "paleturquoise", "orangered", "mistyrose", "limegreen", "palegreen"]
    if not epoch:
        base_var = np.linspace(1, 10, 9)
        var_name = "epoch"
        fix_value = factor
        fix_ind = 2
        var_ind = 3
        fold = 5
    elif not factor:
        base_var = np.linspace(0, 1.0, 10)
        var_name = "factor"
        fix_value = epoch
        fix_ind = 3
        var_ind = 2
        fold = 5
    plt.figure()
    for ind, method in enumerate(aug_meth):
        files = find_files(file_dir, pattern='*aug_{}*.txt'.format(method))

        sum_aucs = []

        for fn in files:
            print("Model {}, fix_name {}: {}".format(os.path.basename(fn).split("_")[1], fix_name, fix_value))
            data = pd.read_csv(fn, header=0).values
            need_inds = np.where(
                (data[:, fix_ind] == fix_value) &
                (data[:, 1] == fold))[0]
            new_data = data[need_inds]

            sort_data = sorted(new_data, key=lambda x: x[var_ind])
            var_values = np.array(sort_data)[:, var_ind].astype(np.float)
            auc = np.array(sort_data)[:, -1].astype(np.float)
            auc_interp = interp(base_var, var_values, auc)
            sum_aucs.append(auc_interp)
            print("ok")

        mean_auc = np.mean(np.array(sum_aucs), axis=0)
        std_auc = np.std(np.array(sum_aucs), axis=0)

        plt.plot(base_var, mean_auc, color=colors[ind * 2], linewidth=2.5, label=method)
        plt.fill_between(base_var, mean_auc - std_auc, mean_auc + std_auc, color=colors[ind * 2 + 1])

    plt.legend()
    plt.ylim([0.4, 0.82])
    plt.xlabel("{}".format(var_name))
    plt.ylabel("area under the curve")
    plt.savefig(os.path.dirname(file_dir) + "/auc_as_factor_all_fix_{}_var_{}.png".format(fix_name, var_name), format="png")
    plt.savefig(os.path.dirname(file_dir) + "/auc_as_factor_all_fix_{}_var_{}.pdf".format(fix_name, var_name), format="pdf")
    plt.close()

# ------------------------------------------------


original = "../data/20190325/20190325-3class_lout40_val_data5-2class_human_performance844_with_labels.mat"

plot_name = "test_aucs"


if plot_name == "indi_rating_with_model":
    data_dir = "../data/20190325"

    model_results = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-10-09T13-47-36-data-lout40-datas-1d-class2-Res_ECG_CAM-0.766certainEp3-aug_ops_meanx10-0.3-test-auc0.79/AUCs/AUC_curve_step_0.00-auc_0.7905-lout40-datas.csv"

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
    model_auc = pd.read_csv(model_results, header=0).values
    true_model_label = model_auc[:, 0].astype(np.int)
    pred_logits = model_auc[:, 1]
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
    model_results = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-10-09T13-47-36-data-lout40-datas-1d-class2-Res_ECG_CAM-0.766certainEp3-aug_ops_meanx10-0.3-test-auc0.79/AUCs/AUC_curve_step_0.00-auc_0.7905-lout40-datas.csv"
    model_auc = pd.read_csv(model_results, header=0).values
    label = model_auc[:, 0].astype(np.int)
    pred_logits = model_auc[:, 1]
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
    fpr, tpr, _ = metrics.roc_curve(label, pred_logits)
    score = metrics.roc_auc_score(label, pred_logits)

    plt.plot(hum_fpr[1], hum_tpr[1], 'purple', marker="*", markersize=10, label='human AUC: {:.2f}'.format(hum_score))
    plt.plot(fpr, tpr, 'royalblue', linewidth=2, label='model AUC: {:.2f}'.format(score))
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
elif plot_name == "certain":
    fn = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/saved_certain/2019-10-07T17-02-18-data-20190325-3class_lout40_train_test_data5-1d-class-2-Res_ECG_CAM-relu-aug_ops_meanx5-0.7-train--auc0.715/certains/certain_data_train_epoch_0_num660.csv"
elif plot_name == "test_aucs":
    from_dirs = True
    if from_dirs:
        results = "/home/epilepsy-data/data/metabolites/paper_results_2700/saved_all_models_and_tests"
        model = "0.776"
        pattern = "*exp{}*_train_test-*".format(model)
        folders = find_folderes(results, pattern=pattern)
        configs = [] # "aug_method": [], "aug_factor": [], "aug_fold": [], "from_epoch": []
        for fn in folders:
            print(fn)
            splits = os.path.basename(fn).split("_")
            aug_name = splits[2]
            aug_fold = splits[3].split("x")[-1]
            aug_factor = splits[5]
            from_epoch = splits[7]
            test_auc = splits[-1].split("-")[-1]
            configs.append((aug_name, aug_fold, aug_factor, from_epoch, test_auc))

        ## plot same_mean aug, auc w.r.t.
        print("ok")
        configs = np.array(configs)
        aug_same = configs[np.where(configs[:, 0] == "same")[0]]
        aug_ops = configs[np.where(configs[:, 0] == "ops")[0]]
        aug_both = configs[np.where(configs[:, 0] == "both")[0]]

        np.savetxt(os.path.join(results, 'model_{}_aug_same_entry_{}.txt'.format(model, len(aug_same))), aug_same, header="aug_name,aug_fold,aug_factor,from_epoch,test_auc", delimiter=",", fmt="%s")
        np.savetxt(os.path.join(results, 'model_{}_aug_ops_entry_{}.txt'.format(model, len(aug_ops))), aug_ops, header="aug_name,aug_fold,aug_factor,from_epoch,test_auc", delimiter=",", fmt="%s")
        np.savetxt(os.path.join(results, 'model_{}_aug_both_entry_{}.txt'.format(model, len(aug_both))), aug_both, header="aug_name,aug_fold,aug_factor,from_epoch,test_auc", delimiter=",", fmt="%s")
    else:
        file_dir = "/home/epilepsy-data/data/metabolites/paper_results_2700/Z-Summary-results"
        aug_meth = ["same", "ops", "both"]

        colors = pylab.cm.Set2(np.linspace(0, 1, 6))
        factor = 0.5
        get_auc_as_factor(epoch=None, factor=factor, aug_meth=aug_meth, colors=colors, fix_name="factor")

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
    results = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results"
    pattern = "*train_test"
    folders = find_folderes(results, pattern=pattern)
    for fn in folders:
        print(fn)
        test_result = find_files(fn, pattern="accuracy_step_0.0_acc_*")
        splits = os.path.basename(test_result[0]).split("_")
        auc = splits[-2]
        os.rename(fn, fn+'-{}'.format(auc))

    ## plot aug_method, from_epoch, mix_factor, aug_fold

















