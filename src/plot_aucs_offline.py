import fnmatch
from os import path as path
import itertools
import pickle


from scipy.io import loadmat, savemat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import scipy as scipy
from collections import Counter
from sklearn import metrics
from textwrap import wrap


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
          'figure.figsize': (8, 6),
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
            files.append(path.join(root, filename))

    return files


def find_optimal_cutoff(true_lbs, predicted):
    fpr, tpr, threshold = metrics.roc_curve(true_lbs, predicted)
    auc = metrics.roc_auc_score(true_lbs, predicted)
    ind = np.argsort(np.abs(tpr - (1-fpr)))[1]
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
            print("Model {}, fix_name1 {}: {}, fix_name2 {}: {}".format(path.basename(fn).split("_")[1], header[fix_ind1], fix_value1, header[fix_ind2], fix_value2))
            data = pd.read_csv(fn, header=0).values
            need_inds = np.where(
                (data[:, fix_ind1] == fix_value1) &
                (data[:, fix_ind2] == fix_value2))[0]
            new_data = data[need_inds]

            sort_data = sorted(new_data, key=lambda x: x[var_ind])
            var_values = np.array(sort_data)[:, var_ind].astype(np.float)
            auc = np.array(sort_data)[:, -1].astype(np.float)
            auc_interp = np.interp(base_var, var_values, auc)
            sum_aucs.append(auc_interp)
            print("ok")

        mean_auc = np.mean(np.array(sum_aucs), axis=0)
        std_auc = stats.sem(np.array(sum_aucs))
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
    plt.savefig(path.dirname(data_dir) + "/auc_as_factor_all_fix_{}-{}_and_{}-{}_var_{}-errbar.png".format(header[fix_ind1], fix_value1, header[fix_ind2], fix_value2, header[var_ind]), format="png")
    plt.savefig(path.dirname(data_dir) + "/auc_as_factor_all_fix_{}-{}_and_{}-{}_var_{}-errbar.pdf".format(header[fix_ind1], fix_value1, header[fix_ind2], fix_value2, header[var_ind]), format="pdf")
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
    plt.savefig(path.join(save_folder, "{}_clusters_label_{}-spec-{}-mean.png".format(num_clusters, cluster_id, postfix)), format="png")
    plt.savefig(path.join(save_folder, "{}_clusters_label_{}-spec-{}-mean.pdf".format(num_clusters, cluster_id, postfix)), format="pdf")
    plt.close()


def get_data_from_certain_ids(certain_fns, mat_file="../data/lout40_train_val_data5.mat"):
    """
    Load data from previous certain examples
    :param args:
    :param certain_fns: list of filenames, from train and validation
    :return:
    """
    mat = io.loadmat(mat_file)["DATA"]  # [id, label, features]
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
        print(path.basename(certain_fns), len(picked_ids), "samples\n")
    
    return sub_mat, new_mat[picked_ids]


def get_data_from_mat(mat_file):
    """
    Load data from previous certain examples
    :param args:
    :param certain_fns: list of filenames, from train and validation
    :return:
    """
    mat = io.loadmat(mat_file)["DATA"]  # [id, label, features]
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


def double_axis_in_one_plot(data1, data2, label1="correct clf. rate", label2="distilled selection rate"):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel("sample index (sorted)")
    ax1.set_ylabel("rate over 100 runs", color=color)
    ax1.plot(data1, label=label1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0,1.0])
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('rate (100 runs) ', color=color)  # we already handled the x-label with ax1
    ax2.plot(data2, label=label2, color=color)
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
        io.savemat(
            path.dirname(args.input_data) + '/20190325-{}class_lout{}_val_data{}.mat'.format(args.num_classes,
                                                                                                args.num_lout, i),
            val_mat)
        io.savemat(
            path.dirname(args.input_data) + '/20190325-{}class_lout{}_train_test_data{}.mat'.format(args.num_classes,
                                                                                                       args.num_lout,
                                                                                                       i),
            train_test_mat)



def plot_bokeh_interactive(x, y, indiv_id="1227", colormap=pylab.cm.jet,
                           hover_notions=[("pat_ids", np.arange(10))],
                           cmap_interval=10,
                           xlabel="xlabel", ylabel="ylabel",
                           title="Title", mode="tsne",
                           plot_func="scatter", postfix="postfix",
                           save_dir="../results"):
    """
    PLot Bokeh plot with given data
    :param x:
    :param y:
    :param indiv_id:
    :param colormap:
    :param hover_notions:
    :param plot_func:
    :param postfix:
    :return:
    """
    from bokeh.plotting import figure, output_file, show, ColumnDataSource, save
    from bokeh.sampledata.iris import flowers
    from bokeh.transform import linear_cmap
    from bokeh.palettes import Turbo256 as jet
    from bokeh.models import ColorBar, HoverTool
    import matplotlib.colors as clr
    # plot bokeh interactive
    tooltips = []
    data_dict = {"x": x,
                 "y": y,
                 "desc": np.arange(len(y)),
                 "colors": ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in
                            255 * colormap(clr.Normalize()(np.arange(len(x))))],
                 }
    for key, value in hover_notions:
        data_dict.update({key: value})
        tooltips.append((key, "@{}".format(key)))

    source_train = ColumnDataSource(data=data_dict)

    hovers = HoverTool(names=["train"],
                       tooltips=tooltips)
    p = figure(plot_width=650, plot_height=650, tools=[hovers, 'pan', 'wheel_zoom','box_select', 'lasso_select', 'poly_select', 'tap', 'reset'],
               title="{}-{}-{}".format(indiv_id, mode, postfix))
    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel
    if plot_func == "scatter":
        p.circle('x', 'y', size=10, fill_color='colors',
                 alpha=0.85, line_width=1, source=source_train, name="train")
    elif plot_func == "plot":
        p.line('x', 'y', source=source_train, name="train", line_width=2)
        p.circle('x', 'y', size=10, fill_color='white', line_width=1, source=source_train)
    mapper = linear_cmap(field_name='time', palette=jet, low=0, high=cmap_interval)
    color_bar = ColorBar(color_mapper=mapper['transform'], width=12, location=(0, 0))
    p.add_layout(color_bar, 'right')
    output_file(save_dir + '/2D-{}-on-{}2.html'.format(mode, postfix),
                title=title)
    save(p)

def interactive_bokeh_with_select(x, y, indiv_id="1227", colormap="Category10",
                                  hover_notions=[("pat_ids", np.arange(10))],
                                  colorby="label", cmap_interval=10,
                                  xlabel="xlabel", ylabel="ylabel",
                                  title="Title", mode="tsne",
                                  plot_func="scatter", postfix="postfix",
                                  save_dir="../results", scatter_size=3):
    """
        # Great! https://github.com/surfaceowl-ai/python_visualizations/blob/main/notebooks/bokeh_save_linked_plot_data.ipynb
            # Generate linked plots + TABLE displaying data + save button to export cvs of selected data
        :param x:
        :param y:
        :param indiv_id:
        :param colormap: bokeh palettes
        :param hover_notions:
        :param cmap_interval:
        :param xlabel:
        :param ylabel:
        :param title:
        :param mode:
        :param plot_func:
        :param postfix:
        :param save_dir:
        :return:
        """
    from bokeh.io import show
    from bokeh.layouts import row, grid
    from bokeh.models import CustomJS, ColumnDataSource, HoverTool, Button, ColorBar, FixedTicker
    from bokeh.events import ButtonClick  # for saving data
    from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
    from bokeh.plotting import figure, show, save, output_file
    from bokeh.transform import linear_cmap
    import matplotlib.colors as clr
    from bokeh.palettes import Category20, Category10, Viridis3, Viridis256, Category10_3, Set2
    from bokeh.transform import factor_cmap, factor_mark
    
    bokeh_color_palettes = {"Category10": Category10,
                            "Category20": Category20,
                            "Viridis256": Viridis256,
                            "Viridis3": Viridis3,
                            "Set2": Set2
                            }
    tooltips = []
    data_dict = {"x": x, "y": y, "desc": np.arange(len(y))}
    plot_width = 600
    plot_height = 600
    for key, value in hover_notions:
        data_dict.update({key: value})
        tooltips.append((key, "@{}".format(key)))
    if colorby == "pat_id":
        uniq_values, indices, counts = np.unique(data_dict[colorby], return_index=True, return_counts=True)
        indi_sort_order = np.sort(indices)


        # sort_uniq_pats[pat_index]
        replace_uniq_pats = np.arange(len(uniq_values))
        sort_pats_counts = np.append(indi_sort_order[1:] - indi_sort_order[0:-1], len(data_dict["pat_id"])-indi_sort_order[-1])
        replaced_all_pats = np.repeat(replace_uniq_pats, sort_pats_counts)
        cmap_colors = pylab.cm.viridis(np.linspace(0, 1, len(uniq_values)))
        data_dict.update({"colors": ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in
                                     255 * cmap_colors[replaced_all_pats]], })
        used_color_palette = "Viridis256"
    else:
        uniq_values, indices, counts = np.unique(data_dict[colorby], return_index=True, return_counts=True)
        cmap_colors = np.array(bokeh_color_palettes[colormap][3][0:len(uniq_values)])
        data_dict.update({"colors": cmap_colors[data_dict[colorby]]})
        used_color_palette = bokeh_color_palettes[colormap][3][0:len(uniq_values)]

        # cmap_colors = pylab.cm.tab10(np.linspace(0, 1, len(np.unique(data_dict[colorby]))))
        # data_dict.update({"colors": ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in
        #                              255 * cmap_colors[data_dict[colorby]]]})
        
    # data_dict.update({"colors": ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in
    #                              255 * colormap(clr.Normalize()(np.arange(len(x))))], })
    s1 = ColumnDataSource(data=data_dict)
    
    hovers = HoverTool(names=["train"], tooltips=tooltips)
    fig01 = figure(plot_width=plot_width, plot_height=plot_height,
                   tools=[hovers, 'pan', 'wheel_zoom', 'box_select', 'lasso_select', 'poly_select', 'tap', 'reset'],
                   title="\n".join(wrap("{}-{}-{}".format(indiv_id, mode, postfix), 60)))
    fig01.circle('x', 'y', size=scatter_size, fill_color='colors', line_color='colors', alpha=0.99, line_width=1, source=s1, name="train")
    mapper = linear_cmap(field_name='time', palette=used_color_palette, low=0, high=cmap_interval)  #"Viridis256", palette=bokeh_color_palettes[colormap]
    ticks = np.ceil(np.linspace(max(0, np.min(uniq_values)), cmap_interval, min(len(uniq_values), 5))).astype(np.int32)
    color_ticks = FixedTicker(ticks=ticks)
    color_bar = ColorBar(color_mapper=mapper['transform'], width=12, location=(0, 0), ticker=color_ticks)
    fig01.add_layout(color_bar, 'right')

    ## plot selected points
    s2 = ColumnDataSource(data=dict(x=[], y=[], pat_id=[], index=[], colors=[]))
    # demo smart error msg:  `box_zoom`, vs `BoxZoomTool`
    fig02 = figure(plot_width=500, plot_height=500, tools=["box_zoom", "wheel_zoom", "reset", "save"],
                   title="Selected are here", )  # x_range=(0, 1), y_range=(0, 1),
    # fig02.circle("x", "y", size=5, source=s2, alpha=0.5, color="firebrick")
    fig02.circle("x", "y", size=scatter_size+3, source=s2, alpha=0.99, fill_color='colors', line_color='colors')
    
    # create dynamic table of selected points
    columns = [TableColumn(field="pat_id", title="pat_id"), TableColumn(field="index", title="index"), TableColumn(field="label", title="label"), ]
    table = DataTable(source=s2, columns=columns, width=400, height=plot_height, sortable=True, selectable=True,
                      editable=True)
    
    # fancy javascript to link subplots
    # js pushes selected points into ColumnDataSource of 2nd plot
    # inspiration for this from a few sources:
    # credit: https://stackoverflow.com/users/1097752/iolsmit via: https://stackoverflow.com/questions/48982260/bokeh-lasso-select-to-table-update
    # credit: https://stackoverflow.com/users/8412027/joris via: https://stackoverflow.com/questions/34164587/get-selected-data-contained-within-box-select-tool-in-bokeh
    
    code = """
        var inds = cb_obj.indices;
        var d1 = s1.data;
        var d2 = s2.data;"""

    for key in data_dict.keys():
        code += f"d2['{key}'] = []; \n"
    
    code += "for (var i = 0; i < inds.length; i++) {\n"

    for key in data_dict.keys():
        code += f"d2['{key}'].push(d1['{key}'][inds[i]]);\n"

    code += """
        }
        s2.change.emit();
        table.change.emit();

        var inds = source_data.selected.indices;
        var data = source_data.data;
        var out = "x, y\\n";
        for (i = 0; i < inds.length; i++) {
        """
    code += "out+="
    for key in data_dict.keys():
        code += f"data['{key}'][inds[i]] + \",\" +"
    code = code[:-5] # remove last ","
    code += '"\\n"; \n'
    code += """
        }
        var file = new Blob([out], {type: 'text/plain'});
         """
    
    s1.selected.js_on_change("indices", CustomJS(args=dict(s1=s1, s2=s2, table=table), code=code))

    # create save button - saves selected datapoints to text file onbutton
    # inspriation for this code:
    # credit:  https://stackoverflow.com/questions/31824124/is-there-a-way-to-save-bokeh-data-table-content
    # note: savebutton line `var out = "x, y\\n";` defines the header of the exported file, helpful to have a header for downstream processing
    savebutton = Button(label="Save", button_type="success")
    savebutton.js_on_event(ButtonClick, CustomJS(args=dict(source_data=s1), code="""
                    var inds = source_data.selected.indices;
                    var data = source_data.data;
                    var out = "pat_id, index\\n";
                    for (var i = 0; i < inds.length; i++) {
                        out += data['pat_id'][inds[i]] + "," + data['index'][inds[i]] + "\\n";
                    }
                    var file = new Blob([out], {type: 'text/plain'});
                    var elem = window.document.createElement('a');
                    elem.href = window.URL.createObjectURL(file);
                    elem.download = 'selected-data.txt';
                    document.body.appendChild(elem);
                    elem.click();
                    document.body.removeChild(elem);
                    """))
    
    # add Hover tool
    # define what is displayed in the tooltip
    tooltips = [("pat_id", "@pat_id"), ("index", "@index"), ]
    source_train = ColumnDataSource(data=data_dict)
    fig02.add_tools(HoverTool(tooltips=tooltips))
    
    # display results
    layout = grid([fig01, fig02, table, savebutton], ncols=4)
    output_file(save_dir + '/2D-{}-on-{}-with-select-size{}.html'.format(mode, postfix, scatter_size), title=title)
    save(layout)
    # output_file(r"C:\Users\LDY\Desktop\1-all-experiment-results\test_bokeh2.html")
    # save(layout)
    print("ok")
    show(layout)



# ------------------------------------------------



plot_name = "first_impression_on_datasets_interactive_bokeh"


def load_selected_bokeh_points_plot(features, filename, region=[-5,-9], dataset_name="Dataset1"):
    saved_selected = r"C:\Users\LDY\Desktop\{}.txt".format(filename)
    saved_data = pd.read_csv(saved_selected, header=0).values
    saved_pat_ids = saved_data[:, 0]
    saved_sample_index = saved_data[:, 1]
    plt.figure()
    plt.plot(features[saved_sample_index].T)
    plt.xlabel("metabolite index")
    plt.ylabel("amplitude [a.u.]")
    plt.title("\n".join(wrap(f"{dataset_name}: region-{region}, filename-{filename}", 50)))
    plt.savefig(path.join(path.dirname(ori_data2),
                             f"plot_{dataset_name}_from_selected_{region}_{filename}.png"))
    plt.close()


if plot_name == "indi_rating_with_model":
    """
    Get performance of doctors, model_without_aug and model_with_aug on individual chuncks
    give individual doctors' ratings, model logits
    """
    print("Plot_name: ", plot_name)
    data_dir = "../data/20190325"
    # Get individual rater's prediction
    human_indi_rating = "../data/20190325/20190325-3class_lout40_test_data5-2class_doctor_ratings_individual_new.mat"
    indi_mat = loadmat(human_indi_rating)['a']
    indi_ratings = np.array(indi_mat)

    # Get model's prediction
    true_data = loadmat("../data/20190325/20190325-3class_lout40_test_data5-2class_human_performance844_with_labels.mat")["DATA"]
    true_label = true_data[:, 1].astype(np.int)
    
    model_res_wo_aug_fn = "C:/Users/LDY/Desktop/1-all-experiment-results/metabolites/auc-func-as-augmentation-parameters/with-randDA-AUC_curve_step_0.00-auc_0.7198-data5-test.csv"
    model_auc_wo_aug = pd.read_csv(model_res_wo_aug_fn, header=0).values
    true_model_label_wo_aug = model_auc_wo_aug[:, 0].astype(np.int)
    pred_logits_wo_aug = model_auc_wo_aug[:, 1]
    

    ## load data from saved fpr and tpr
    model_res_with_aug_fn = "C:/Users/LDY/Desktop/1-all-experiment-results/metabolites/auc-func-as-augmentation-parameters/with-DA-dist-AUC_curve_step_0.00-auc_0.7672-data5-test.csv"
    model_auc_with_aug = pd.read_csv(model_res_with_aug_fn, header=0).values
    true_model_label_with_aug = model_auc_with_aug[:, 0].astype(np.int)
    pred_logits_with_aug = model_auc_with_aug[:, 1]

    # Get human's cumulative labels
    human_rating = "../data/20190325/20190325-3class_lout40_test_data5-2class_human-ratings.mat"
    hum_whole = loadmat(human_rating)["data_ratings"]
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
    
    plt.savefig(path.join(data_dir, "Model_with_indi_human_rating_only_AUC.png"), format='png')
    plt.savefig(path.join(data_dir, "Model_with_indi_human_rating_only_AUC.pdf"), format='pdf')
    plt.close()
    

elif plot_name == "human_whole_with_model":
    print("Plot_name: ", plot_name)
    data_dir = "../data/20190325"
    original = "../data/20190325/20190325-3class_lout40_test_data5-2class_human_performance844_with_labels.mat"
    ori = loadmat(original)["DATA"]
    true_label = ori[:, 1]
    temp_count = Counter(ori[:, 0])
    patient_summary = {}
    for key in temp_count.keys():
        patient_summary[key] = np.append(temp_count[key], np.mean(true_label[ori[:, 0] == key]))

    # Get model's prediction
    
    # model_res_with_aug = [r"C:\1-study\FIAS\1-My-papers\1-11-submitted-2021.03 MLHC patient-wise-classification\results\results-Hatami\2021-05-08T01-38-12-Hatami2018_with_3pool-+DA-46spec-LOO"]
    # model_res_wo_aug = [r"C:\Users\LDY\Desktop\metabolites-0301\metabolites_tumour_classifier\data\20190325\test_prediction_Hatami2018_with_3pool-spec1-randshuffleCV-CV2_auc_0.719-len844-test_doc.csv"]
    model_res_wo_aug = [r"C:\1-study\FIAS\1-My-papers\1-6-submitted-IEEE-metabolites\IEEE-Medical imaging submission\Results\Datadata4_class2_modelRes_ECG_CAM_test_return_data_acc_0.611.txt"]
    model_res_with_aug = [r"C:\1-study\FIAS\1-My-papers\1-6-submitted-IEEE-metabolites\IEEE-Medical imaging submission\Results\Datadata2_class2_modelRes_ECG_CAM_test_return_data_acc_0.623.txt"]
    model_names = ["hatami-3Pool", "MLP-3Pool", "Inception-3Pool", "hatami-atten", "MLP-atten", "Inception-atten"]
    for model_with_aug, model_wo_aug in zip(model_res_with_aug, model_res_wo_aug):
        """
        certain_data = np.concatenate((output_data["test_certain_sample_ids"].reshape(-1, 1),
                                                   output_data["test_certain_labels"].reshape(-1, 1),
                                                   output_data["test_certain_logits"]), axis=1)"""
        ff = open(model_with_aug, 'rb')
        data_dict = pickle.load(ff)
        label_with_aug = data_dict["output_data"]["test_labels"]
        pred_logits_with_aug = data_dict["output_data"]["test_logits"]
        ff2 = open(model_wo_aug, 'rb')
        data_dict = pickle.load(ff2)
        label_wo_aug = data_dict["output_data"]["test_labels"]
        pred_logits_wo_aug = data_dict["output_data"]["test_logits"]
        
        ## load from csv file
        # model_auc_with_aug = pd.read_csv(model_res_with_aug, header=None).values
        # label_with_aug = model_auc_with_aug[:, 2].astype(np.int)
        # pred_logits_with_aug = model_auc_with_aug[:, -1]
        # model_auc_wo_aug = pd.read_csv(model_res_wo_aug, header=None).values
        # label_wo_aug = model_auc_wo_aug[:, 2].astype(np.int)
        # pred_logits_wo_aug = model_auc_wo_aug[:, -1]
    
        # Get human's total labels
        human_rating = "../data/20190325/20190325-3class_lout40_test_data5-2class_human-ratings.mat"
        hum_whole = loadmat(human_rating)["data_ratings"]
        
        human_lb = hum_whole[:, 0]
        human_features = hum_whole[:, 1:]
        hum_fpr, hum_tpr, _ = metrics.roc_curve(true_label, human_lb)
        hum_score = metrics.roc_auc_score(true_label, human_lb)

        # PLot human average rating
        plt.figure(figsize=[10, 7])
        plt.scatter(hum_fpr[1], hum_tpr[1], color='purple', marker="*", s=50,
                    label='cumulative performance'.format(hum_score))

        # Plot trained model prediction
        fpr_with_aug, tpr_with_aug, _ = metrics.roc_curve(label_with_aug[:,1], pred_logits_with_aug[:,1])
        fpr_wo_aug, tpr_wo_aug, _ = metrics.roc_curve(label_wo_aug[:,1], pred_logits_wo_aug[:,1])
        score_with_aug = metrics.roc_auc_score(label_with_aug[:,1], pred_logits_with_aug[:,1])
        score_wo_aug = metrics.roc_auc_score(label_wo_aug[:,1], pred_logits_wo_aug[:,1])

        plt.plot(fpr_with_aug, tpr_with_aug, 'royalblue', linestyle="-", linewidth=2,
                 label='With aug. AUC: {:.2f}'.format(
                     score_with_aug))  # , label='With aug. AUC: {:.2f}'.format(score_with_aug)
        plt.plot(fpr_wo_aug, tpr_wo_aug, 'violet', linestyle="-.", linewidth=2,
                 label='Without aug. AUC: {:.2f}'.format(score_wo_aug))
        plt.title("Receiver Operating Characteristic", fontsize=20)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.legend(loc=4)
        plt.ylabel('true positive rate', fontsize=18)
        plt.xlabel('false positive rate', fontsize=18)
        plt.savefig(path.join(data_dir, "model_with_human_rating_collectively_certain.pdf"), format='pdf')
        # plt.savefig(path.join(data_dir, "model_with_human_rating.eps"), format='eps')
        plt.savefig(path.join(data_dir, "model_with_human_rating_collectively_certain.png"), format='png')
        plt.close()
        
        ## Get four groups performance with model and humans: "agree_right", "agree_wrong", "disagree_human_correct", "disagree_human_wrong"
        agree_all_right = np.where((label_with_aug==human_lb) & (human_lb==true_label))[0]
        agree_all_wrong = np.where((label_with_aug==human_lb) & (human_lb!=true_label))[0]
        disagree_human_right = np.where((label_with_aug!=human_lb) & (human_lb==true_label))[0]
        disagree_human_wrong = np.where((label_with_aug!=human_lb) & (human_lb!=true_label))[0]
        
        for subfd in ["agree_all_right", "agree_all_wrong", "disagree_human_right", "disagree_human_wrong"]:
            path = path.join(path.dirname(model_res_with_aug), subfd)
            if not path.exists(path):
                os.mkdir(path)
        
        group_compare_sum = {"agree_all_right":[], "agree_all_wrong":[], "disagree_human_right":[], "disagree_human_wrong":[]}
        for name, inds, folder in zip(["all agree right", "all agree wrong", "disagree human right", "disagree human wrong"], [agree_all_right, agree_all_wrong, disagree_human_right, disagree_human_wrong], ["agree_all_right", "agree_all_wrong", "disagree_human_right", "disagree_human_wrong"]):
            current_spec = ori[inds]
            current_pat_count = np.array(sorted([ele for ele in Counter(current_spec[:, 0]).items()], key=lambda x : x[0]))
            
            for pat in current_pat_count[:, 0]:
                pat_inds = np.where(current_spec[:,0] == pat)[0]
                pat_spec = current_spec[pat_inds]
                pat_tot_num = patient_summary[pat][0]
                group_compare_sum[folder].append([len(pat_inds), pat_tot_num, pat, pat_spec[0,1]])  #number of spec, tot_number, patID, patLabel
                
                ## for each patient, plot the coresponding spectra
                # plt.figure()
                # plt.plot(pat_spec[:,2:].T)
                # plt.ylabel("normalized value")
                # plt.xlabel("index")
                # plt.title("Patient {}, {}/{} {} (true {})".format(pat, len(pat_inds), pat_tot_num, name, pat_spec[0,1]))
                # plt.savefig(path.join(data_dir, folder, "{}-out-{}-{}-patient-{}.png".format(len(pat_inds), pat_tot_num, name, pat)), format='png')
                # plt.savefig(path.join(data_dir, folder,"{}-out-{}-{}-patient-{}.pdf".format(len(pat_inds), pat_tot_num, name, pat)), format='pdf')
                # plt.close()
        # preprocess the summary matrix, assign 0 to non-existing pat-id
        for pat in patient_summary.keys():
            for folder in group_compare_sum.keys():
                pat_ind = np.where(np.array(group_compare_sum[folder])[:,-2]==pat)[0]
                if len(pat_ind) < 1:
                    group_compare_sum[folder].append([0, patient_summary[pat][0], pat, patient_summary[pat][1]])
                    
                    
        rank_key = "agree_all_right"
        xticks = np.arange(len(group_compare_sum[rank_key]))  # the x locations for the groups
        width = 0.8
        sorted_summary = np.array(sorted(group_compare_sum[rank_key], key=lambda x: x[0]))
        rank_patients = sorted_summary[:,-2]
        all_sorted_summaries = {key: 0 for key in group_compare_sum.keys()}
        colors = ['#1D2F6F', '#8390FA', '#6EAF46', '#FAC748']
        bar_bottom = np.zeros(len(xticks))
        
        fig = plt.figure(figsize=[12,6])
        for ind, folder in enumerate(group_compare_sum.keys()):
            curren_summary = np.array(group_compare_sum[folder])
            # get the counts with the predefined patient ranking
            temp_summary = np.empty((0, curren_summary.shape[1]))
            for jj in rank_patients:
                temp_summary = np.vstack((temp_summary, curren_summary[curren_summary[:,2]==jj]))
            plt.bar(xticks, np.array(temp_summary[:,0]), width, bottom=bar_bottom, label=folder, color=colors[ind])
            bar_bottom = np.add(bar_bottom, np.array(temp_summary[:,0])).tolist()
        
        for ind in xticks:
            plt.text(ind, bar_bottom[ind], np.str(np.int32(patient_summary[rank_patients[ind]][1])), horizontalalignment="center")
        plt.legend()
        plt.ylabel("number of spectra")
        plt.xlabel("patient IDs")
        plt.xticks(xticks, temp_summary[:,2].astype(np.int32), rotation=90)
        plt.title("Summary of model-human agreeness")
        plt.tight_layout()
        plt.savefig(path.join(data_dir, folder,
                                 "summary-{}.png".format( folder)),
                    format='png')
        plt.savefig(path.join(data_dir, folder,
                                 "summary-{}.pdf".format( folder)),
                    format='pdf')
        plt.close()
    
    
        wrong_inds = np.where(true_label!=human_lb)[0]
        wrong_spec = ori[wrong_inds]
        wrong_pat_count = sorted([ele for ele in Counter(wrong_spec[:, 0]).items()],
                                 key=lambda x: x[0])
        

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
    plt.savefig(path.join(data_dir, "All ROC curves in cross validation test-lout40-validation.png"), format='png')
    plt.savefig(path.join(data_dir, "All ROC curves in cross validation test-lout40-validation.pdf"), format='pdf')
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
    from_dirs = False  #True   #
    if from_dirs:
        data_dir = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-with-distillation-MLP"
        model = path.basename(data_dir).split("-")[-1]
        exp_mode = path.basename(data_dir).split("-")[-2]

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
                splits = path.basename(fn).split("-")
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
            plt.savefig(path.join(data_dir, "new-{}-with-{}-on-{}.png".format(exp_mode, model, data_source))),
            plt.savefig(path.join(data_dir, "new-{}-with-{}-on-{}.pdf".format(exp_mode, model, data_source)), format="pdf")
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
            # plt.savefig(path.join(data_dir, "{}-with-{}-on-{}-best-same-fold3-scale0.5-both.png".format(exp_mode, model, data_source))),
            # plt.savefig(path.join(data_dir, "{}-with-{}-on-{}-best-same-fold3-scale0.5-both.pdf".format(exp_mode, model, data_source)), format="pdf")
            # plt.close()
            #

            np.savetxt(path.join(data_dir, 'new-model_{}_all_different_config_theta{}-{}+DA-with-{}-on-{}.txt'.format(model, len(aug_same), exp_mode, model, data_source)), configs, header="aug_name,aug_fold,aug_factor,cer_th,test_auc", delimiter=",", fmt="%s")
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
        data_source = "all-CVs"  #path.basename(fn).split("-")[-1].split(".")[0]
        
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
            plt.savefig(path.join(path.dirname(fn), "{}-all-methods-in-one-{}-with-{}-on-{}.png".format(plot_style, method, model_name, data_source))),
            plt.savefig(path.join(path.dirname(fn), "{}-all-methods-in-one-{}-with-{}-on-{}.pdf".format(plot_style, method, model_name, data_source)), format="pdf")
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
                    
                im = axes[md_case].imshow(matrix_values, interpolation='none', vmin=0.43, vmax=0.74, aspect='equal', cmap="Blues")
                axes[md_case].set_xlabel(r"mixing weight $\alpha$")
                axes[md_case].set_ylabel(r"augmentation factor $\Phi$")
                divider = make_axes_locatable(axes[md_case])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                clb = fig.colorbar(im, cax=cax)   #, orientation='horizontal'
                clb.set_label('AUC', labelpad=-40, y=1.05, rotation=0)

                axes[md_case].set_xticks(np.arange(0, 4, 1), [0.05, 0.2, 0.35, 0.5]),
                axes[md_case].set_yticks(np.arange(0, 4, 1), [1,3,5,9])
                
                # threshold = np.mean(matrix_values)
                threshold = 0.6
                for scl_ind in range(4):
                    for fd_inds in range(4):
                        color = "black" if matrix_values[scl_ind, fd_inds] < threshold else "white"
                        axes[md_case].text(fd_inds, scl_ind, r'${:.3f}$'.format(matrix_values[scl_ind, fd_inds]), color=color, horizontalalignment='center', fontsize=17)
                        axes[md_case].text(fd_inds, scl_ind+0.20, r'$\pm {:.3f}$'.format(matrix_stds[scl_ind, fd_inds]), color=color, horizontalalignment='center', fontsize=15)
                md = "other" if method=="ops" else method
                axes[md_case].set_title("\n".join(wrap("Aug-with-{}".format(md), 60)))
                print("oki")
                
                
            plt.tight_layout()
            plt.setp(axes, xticks=np.arange(0, 4, 1), xticklabels=[0.05, 0.2, 0.35, 0.5],
                    yticks=np.arange(0, 4, 1), yticklabels=[1,3,5,9])

            plt.savefig(path.join(path.dirname(fn), "{}-3-in-1-method_{}-in-one-with-{}-on-{}-same-color-scale.png".format(plot_style, md, model_name, data_source)), bbox_inches='tight'),
            plt.savefig(path.join(path.dirname(fn), "{}-3-in-1-method_{}-in-one-with-{}-on-{}-same-color-scale.pdf".format(plot_style, md, model_name, data_source)), format="pdf")
            plt.close()


elif plot_name == "get_performance_metrices":
    #Get overall performance metrices across different cross-validation sets
    print("Plot_name: ", plot_name)
    postfix = ""
    data_dirs = [
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-with-distillation-Inception"]
    for data_source, prefix in zip(["", "data5"], ["all-CVs", "data5"]):
        for data_dir in data_dirs:
            for method in ["both"]:  # "None","same", "both", "ops"
                for fold in [9, 5, 0, 1, 3]:  #
                    for alpha in [0.5, 0, 0.05, 0.2, 0.35]:
                        folders = find_folderes(data_dir,
                                                pattern="*{}*x{}-factor-{}-from-{}*-test-0.*".format(method, fold,
                                                                                                     alpha,
                                                                                                     data_source))
                        if len(folders) > 0:
                            print(path.basename(data_dir), "{}x{}-{}-[{}]!".format(method, fold, alpha, [
                                path.basename(fdn).split("-")[-3] for fdn in folders]))
                            performance = {"ACC": np.empty((0,)), "patient_ACC": np.empty((0,)), "AUC": np.empty((0,)),
                                           "SEN": np.empty((0,)), "SPE": np.empty((0,)), "F1_score": np.empty((0,)),
                                           "MCC": np.empty((0,))}
                            performance_summary = []
                            data_names = []
                            for fd in folders:
                                file = find_files(fd, pattern="AUC_curve_step*.csv")
                                num_patient = find_files(fd, pattern="*prob_distri_of*.png")
                                # rat_id = path.basename(fn).split("-")[-3]
                                data_names.append("{}-{}-{}\n".format(path.basename(fd).split("-")[-3],
                                                                      path.basename(fd).split("-")[-5],
                                                                      path.basename(fd).split("-")[-1]))
                                if len(file) > 0:
                                    values = pd.read_csv(file[0], header=0).values
                                    true_labels = values[:, 0]  # assign true-lbs and probs in aggregation
                                    pred_logits = values[:, 1]
                                    
                                    patient_auc = np.sum(["right" in name for name in num_patient]) / len(num_patient)
                                    # get summary of model performance's metrics
                                    accuracy, sensitivity, specificity, precision, F1_score, auc, fpr, tpr, mcc = get_scalar_performance_matrices_2classes(
                                        true_labels, pred_logits, if_with_logits=True)
                                    
                                    performance["ACC"] = np.append(performance["ACC"], accuracy)
                                    performance["SEN"] = np.append(performance["SEN"], sensitivity)
                                    performance["SPE"] = np.append(performance["SPE"], specificity)
                                    performance["AUC"] = np.append(performance["AUC"], auc)
                                    performance["F1_score"] = np.append(performance["F1_score"], F1_score)
                                    performance["MCC"] = np.append(performance["MCC"], mcc)
                                    performance["patient_ACC"] = np.append(performance["patient_ACC"], patient_auc)
                                    
                                    performance_summary.append(["{}-{}x{}-{}-data[{}]\n".format(
                                        path.basename(data_dir), method, fold, alpha, data_names),
                                                                "Sensitivity: mean-{:.3f}, std-{:.3f}\n".format(
                                                                    np.mean(performance["SEN"]), np.std(performance[
                                                                                                            "SEN"])) + "specificity: mean-{:.3f}, std-{:.3f}\n".format(
                                                                    np.mean(performance["SPE"]), np.std(performance[
                                                                                                            "SPE"])) + "AUC: mean-{:.3f}, std-{:.3f}\n".format(
                                                                    np.mean(performance["AUC"]), np.std(performance[
                                                                                                            "AUC"])) + "patient acc: mean-{:.3f}, std-{:.3f}\n".format(
                                                                    np.mean(performance["patient_ACC"]), np.std(
                                                                        performance[
                                                                            "patient_ACC"])) + "F1-score: mean-{:.3f}, std-{:.3f}\n".format(
                                                                    np.mean(performance["F1_score"]), np.std(
                                                                        performance[
                                                                            "F1_score"])) + "MCC: mean-{:.3f}, std-{:.3f}\n".format(
                                                                    np.mean(performance["MCC"]), np.std(performance[
                                                                                                            "MCC"])) + "ACC: mean-{:.3f}, std-{:.3f}\n".format(
                                                                    np.mean(performance["ACC"]),
                                                                    np.std(performance["ACC"]))
                                    
                                                                ])
                            np.savetxt(path.join(data_dir,
                                                    "{}-AUC-{:.4f}-performance-summarries-of-{}x{}-{}-num{}-CVs.csv".format(
                                                        prefix, np.mean(performance["AUC"]), method, fold, alpha,
                                                        len(folders))), np.array(performance_summary), fmt="%s",
                                       delimiter=",")
                            
                            print("{}-{}x{}-{}-data[{}]\n".format(path.basename(data_dir), method, fold, alpha,
                                                                  data_names))
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
                            print(path.basename(data_dir), "{}x{}-{}-No data!".format(method, fold, alpha))


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
    # data_dir = path.dirname(ori_data_dir)
    if_get_distilldata = False
    ## get the whole tsne projection
    if reduction_method == "tsne":
        if if_save_data:
            from bhtsne import tsne as TSNE
            reduced_proj_whole = TSNE(whole_data[:, feature_start_id:], dimensions=2)
            np.savetxt(path.join(data_dir, "{}-whole_data-2d.csv".format(reduction_method)), reduced_proj_whole, fmt="%.5f", delimiter=","),
            if if_get_distilldata:
                reduced_proj_distill= TSNE(distill_data[:, feature_start_id:], dimensions=2)
                np.savetxt(path.join(data_dir, "{}-distill_data-2d.csv".format(reduction_method)), reduced_proj_distill, fmt="%.5f", delimiter=",")
        else:
            filename_whole = path.join(data_dir, "{}-whole_data-lout5-2d.csv".format(reduction_method))
            reduced_proj_whole = pd.read_csv(filename_whole, header=None).values
            # filename_distill = path.join(data_dir, "distill_data_tsne-2d-from-whole-tsne.csv.csv".format(reduction_method))
            # reduced_proj_distill = pd.read_csv(filename_distill, header=None).values
    elif reduction_method == "UMAP":
        if if_save_data:
            import umap.umap_ as umap
            reduced_proj_whole = umap.UMAP(random_state=42).fit_transform(whole_data[:, feature_start_id:])
            np.savetxt(path.join(data_dir, "{}-whole_data-2d.csv".format(reduction_method)), reduced_proj_whole, fmt="%.5f", delimiter=","),
            if if_get_distilldata:
                reduced_proj_distill = umap.UMAP(random_state=42).fit_transform(distill_data[:, feature_start_id:])
                np.savetxt(path.join(data_dir, "{}-distill_data-2d.csv".format(reduction_method)), reduced_proj_distill, fmt="%.5f", delimiter=",")
        else:
            filename_whole = path.join(data_dir, "{}-whole_data-2d.csv".format(reduction_method))
            reduced_proj_whole = pd.read_csv(filename_whole, header=None).values
            filename_distill = path.join(data_dir, "{}-distill_data-2d-from-whole.csv".format(reduction_method))
            reduced_proj_distill = pd.read_csv(filename_distill, header=None).values
    elif reduction_method == "MDS":
        if if_save_data:
            from sklearn.manifold import MDS
            MDS_whole = MDS(n_components=2, random_state=199)
            MDS_distill = MDS(n_components=2, random_state=199)
            reduced_proj_whole = MDS_whole.fit_transform(whole_data[:, feature_start_id:])
            if if_get_distilldata:
                reduced_proj_distill = MDS_distill.fit_transform(distill_data[:, feature_start_id:])
                np.savetxt(path.join(data_dir, "{}-distill_data-2d-from-whole.csv".format(reduction_method)), reduced_proj_distill, fmt="%.5f", delimiter=",")
            np.savetxt(path.join(data_dir, "{}-whole_data-2d.csv".format(reduction_method)), reduced_proj_whole, fmt="%.5f", delimiter=","),
        else:
            filename_whole = path.join(data_dir, "{}-whole_data-2d.csv".format(reduction_method))
            filename_distill = path.join(data_dir, "{}-distill_data-2d-from-whole.csv".format(reduction_method))
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
    plt.savefig(path.join(data_dir, "{}-whole-{}.png".format(reduction_method, data_source))),
    plt.savefig(path.join(data_dir, "{}-whole-{}.pdf".format(reduction_method, data_source)), format="pdf")
    plt.close()

    # plot the whole set w.r.t patients
    new_pat_ids = np.empty((0))
    new_order = np.empty((0))
    for pat in np.unique(whole_data[:, 1]):
        pat_inds = np.where(whole_data[:, 1] == pat)[0]
        temp_count = whole_data[pat_inds, 1]
        label = np.mean(whole_data[pat_inds, 2])
        if label == 1:
            temp_count = temp_count + 5000
        new_pat_ids = np.append(new_pat_ids, temp_count)
        new_order = np.append(new_order, pat_inds).astype(np.int)
    plt.figure(figsize=[10, 7])
    # plt.scatter(reduced_proj_whole[:, 0], reduced_proj_whole[:, 1], c=new_pat_ids.astype(np.int), s=15, cmap="jet", facecolor=None)
    plt.scatter(reduced_proj_whole[new_order, 0], reduced_proj_whole[new_order, 1], c=new_pat_ids, s=15, cmap="jet", facecolor=None),
    plt.colorbar(),
    plt.title("\n".join(wrap("Grouped by the patients", 40))),
    plt.xlabel("dimension #1"),
    plt.ylabel("dimension #2")
    plt.savefig(path.join(data_dir, "grouped-by-samples-{}-whole-{}.png".format(reduction_method, data_source))),
    plt.savefig(path.join(data_dir, "grouped-by-samples-{}-whole-{}.pdf".format(reduction_method, data_source)), format="pdf")
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
    plt.savefig(path.join(data_dir, "umap_visualization", "tsne-distill-{}.png".format(data_source)))
    plt.savefig(path.join(data_dir, "umap_visualization", "tsne-distill-{}.pdf".format(data_source)), format="pdf")
    plt.savefig(path.join(data_dir, "Distilled tumor samples-{}-from-{}.png".format(reduction_method, data_source))),
    plt.savefig(path.join(data_dir, "Distilled tumor samples-{}-from-{}.pdf".format(reduction_method, data_source)), format="pdf")
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
    plt.savefig(path.join(data_dir, "umap_visualization", "{}-subset-the-same-number-as-distill-from-whole-{}.png".format(reduction_method, data_source))),
    plt.savefig(path.join(data_dir, "umap_visualization", "{}-subset-the-same-number-as-distill-from-whole-{}.pdf".format(reduction_method, data_source)),
                format="pdf")
    plt.close()

    # plt.figure(figsize=[8, 6.5])
    # plt.scatter(tsne_distill[:, 0], tsne_distill[:, 1], c=distill_data[:,1], cmap="jet", facecolor=None)
    # plt.colorbar()
    # plt.title("Patients from the distilled set  (healthy<1000, tumor>1000)")
    # plt.xlabel("dimension #1"),
    # plt.ylabel("dimension #2")
    # plt.savefig(path.join(data_dir, "grouped-by-patients-{}-distill-{}.png".format(reduction_method, data_source))),
    # plt.savefig(path.join(data_dir, "grouped-by-patients-{}-distill-{}.pdf".format(reduction_method, data_source)), format="pdf")
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
    plt.savefig(path.join(path.dirname(mat_file), "Distribution-of-number-of-spectra-in-patients-whole2.png")),
    plt.savefig(path.join(path.dirname(mat_file), "Distribution-of-number-of-spectra-in-patients-whole2.pdf"))
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
    #     plt.savefig(path.join(path.dirname(mat_file), "plot-spectra-patient-wise", "Spectra-Patient-{}-lb{}-({}).png".format(pat_id, label, num))),
    #     plt.savefig(path.join(path.dirname(mat_file), "plot-spectra-patient-wise", "Spectra-Patient-{}-lb{}-({}).pdf".format(pat_id, label, num))),
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
        print(path.basename(fn), len(certain_inds), "samples\n")

    uniq_inds = np.unique(certain_inds_tot).astype(np.int)
    certain_mat = original_data[uniq_inds]

    ###################################################################
    # plot the certain samples' PCA
    # np.savetxt(path.join(data_dir, "certain_samples_lout40_fold5[smp_id,pat_id,label,meta]_class(0-1)=({}-{}).csv".format(len(np.where(certain_mat[:,2]==0)[0]), len(np.where(certain_mat[:,2]==1)[0]))), certain_mat, delimiter=",", fmt="%.5f")
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
    #     plt.savefig(path.join(data_dir, "certain_samples_class{}.png".format(c)))
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
                path.join(data_dir, "certain_samples_class{}_fig_{}.png".format(c, ii)))
            plt.close()


elif plot_name == "first_impression_on_datasets_interactive_bokeh":
    from scipy.io import loadmat as loadmat

    """
        Dataset1
        1= voxel from affected hemisphere of tumor patient;
        2= voxel from healthy hemisphere of tumor patient;
        0= voxel from both hemisphere of the patient suffering from smth. else.
        """
    ori_data1 = r"C:\Users\LDY\Desktop\metabolites-0301\metabolites_tumour_classifier\data\20190325\2019.03.25-DATA.mat"
    ### load the second version of data
    """
	Dataset2
	% cluster=1 --> Tumor Progress
	% cluster=0 --> Pseudo Progress
	% class_label= 1 --> 'voxel in tumor'
	% class_label= 0 -->'voxel in healthy part of the brain'
	"""
    ori_data2 = r"C:\Users\LDY\Desktop\metabolites-0301\metabolites_tumour_classifier\data\2022.02.01\new_saved_Data_TT_vs_TP2.mat"
    
    ##############################################################################
    dict_pat_ids = {}
    dict_vox_labels = {}
    dict_clusters = {}
    dict_features = {}
    dict_dataset_inds = {}
    
    ## load two datasets and combine them
    for data_dir, data_name in zip([ori_data1, ori_data2], ["dataset1", "dataset2"]):
        ### load the first version of data
        mat = loadmat(data_dir)["DATA"]
        pat_ids = mat[:, 0].astype(np.int32)

        if data_name == "dataset1":
            labels = mat[:, 1].astype(np.int32)
            dict_clusters[data_name] = 2 * np.ones(len(labels))  # there is no cluster for dataset1, so assign 1
            features = mat[:, 2:]
        else:
            clusters = mat[:, 1].astype(np.int32)
            labels = mat[:, 2].astype(np.int32)
            features = mat[:, 3:]
            dict_clusters[data_name] = clusters
        dict_pat_ids[data_name] = pat_ids
        dict_vox_labels[data_name] = labels
        dict_features[data_name] = features

    # presaved_prj_file = r"C:\Users\LDY\Desktop\metabolites-0301\metabolites_tumour_classifier\data\2022.02.01\pacmap_of_both_datasets_[pats,cluster,lb,proj].csv"
    presaved_prj_file = r"C:\Users\LDY\Desktop\metabolites-0301\metabolites_tumour_classifier\data\2022.02.01\pacmap_of_both_datasets_[name,pats,cluster,lb,proj].csv"
    # compute individual pacmap proj, and combined projection
    if not presaved_prj_file:
        from pacmap import PaCMAP  # it is very slow
        combined_dataset_name = np.append(np.repeat("dataset1", len(dict_pat_ids["dataset1"])),
                                          np.repeat("dataset2", len(dict_pat_ids["dataset2"])))
        combined_pat_ids = np.append(dict_pat_ids["dataset1"], dict_pat_ids["dataset2"])
        combined_vox_labels = np.append(dict_vox_labels["dataset1"] + 10, dict_vox_labels["dataset2"])
        combined_clusters = np.append(dict_clusters["dataset1"],
                                      dict_clusters["dataset2"])  # dataset1 patients has cluster 2
    
        
        combined_features = np.vstack((dict_features["dataset1"],
                                       dict_features["dataset2"]))
        # fit the data (The index of transformed data corresponds to the index of the original data)
        for data_name in [ "dataset2"]:  #"dataset1",
            embedding = PaCMAP(n_dims=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
            X_transformed = embedding.fit_transform(dict_features[data_name], init="pca")
            
            np.savetxt(path.join(path.dirname(ori_data2), f"pacmap_of_{data_name}_[name,pats,cluster,lb,proj].csv"),
                                              np.concatenate((np.repeat(data_name, len(dict_pat_ids[data_name])).reshape(-1, 1),
                                                              dict_pat_ids[data_name].reshape(-1, 1),
                                                              dict_clusters[data_name].reshape(-1, 1),
                                                              dict_vox_labels[data_name].reshape(-1, 1), X_transformed), axis=1), delimiter=",",
                                              fmt="%s")
        ## save combined projection
        embedding = PaCMAP(n_dims=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
        X_transformed_all = embedding.fit_transform(combined_features, init="pca")
        np.savetxt(path.join(path.dirname(ori_data2), "pacmap_of_both_datasets_[name,pats,cluster,lb,proj].csv"),
                   np.concatenate((combined_dataset_name.reshape(-1,1),
                                   combined_pat_ids.reshape(-1, 1),
                                   combined_clusters.reshape(-1, 1),
                                   combined_vox_labels.reshape(-1, 1), X_transformed_all), axis=1), delimiter=",", fmt="%s")
    else:
        if "both_datasets" in path.basename(presaved_prj_file):
            load_info = pd.read_csv(presaved_prj_file, header=None).values
            load_combined_dataset_names = load_info[:, 0]
            load_combined_pat_ids = load_info[:, 1].astype(np.int32)
            load_combined_clusters = load_info[:, 2].astype(np.int32)
            load_combined_vox_labels = load_info[:, 3].astype(np.int32)
            X_transformed = load_info[:, 4:].astype(np.float32)
            assert np.sum(load_combined_pat_ids==np.append(dict_pat_ids["dataset1"], dict_pat_ids["dataset2"])) == len(load_combined_pat_ids), "order of data is missed up at pat_ids!"
            assert np.sum(load_combined_vox_labels==np.append(dict_vox_labels["dataset1"]+10, dict_vox_labels["dataset2"])) == len(load_combined_vox_labels), "order of data is missed up at vox_labels!"
            assert np.sum(load_combined_clusters==np.append(dict_clusters["dataset1"], dict_clusters["dataset2"])) == len(load_combined_clusters), "order of data is missed up at clusters!"
    
            dataset_inds1 = np.where(load_combined_dataset_names == "dataset1")[0]
            dataset_inds2 = np.where(load_combined_dataset_names == "dataset2")[0]
            dict_dataset_inds["dataset1"] = dataset_inds1
            dict_dataset_inds["dataset2"] = dataset_inds2
            color_palettes = {0:"Category10", 1:"Category10_3", 2:"Viridis256", 3: "Set2"}
            # visualize the embedding
            for data_name in ["dataset1", "dataset2"]:
                for colorby, colorby_name, colorby_data, colormap in zip(["pat_id", "label", "cluster"],
                                                               ["patient IDs", "voxel labels", "patient clusters"],
                                                                [dict_pat_ids, dict_vox_labels, dict_clusters],
                                                                         [color_palettes[2], color_palettes[0], color_palettes[3]]):
                    if colorby == "cluster" and data_name == "dataset1":
                        interactive_bokeh_with_select(X_transformed[:, 0], X_transformed[:, 1],
                                                      indiv_id="Both-datasets", colormap=colormap,
                                                      hover_notions=[("pat_id", load_combined_pat_ids),
                                                                     ("label", load_combined_vox_labels),
                                                                     ("index", np.arange(len(load_combined_vox_labels))),
                                                                     ("cluster", load_combined_clusters.astype(np.int32))],
                                                      colorby=colorby, cmap_interval=np.max(load_combined_clusters),
                                                      xlabel="dimension #1", ylabel="dimension #1", title="Title",
                                                      mode="pacmap", plot_func="scatter",
                                                      postfix="colored by patient clusters of both datasets (0-Pseudo Progress, 1-true progress, 2-dataset1)",
                                                      save_dir=path.dirname(ori_data2), scatter_size=0.5)#, scatter_size=3
                    else:
                        interactive_bokeh_with_select(X_transformed[dict_dataset_inds[data_name], 0],
                                                      X_transformed[dict_dataset_inds[data_name], 1],
                                                      indiv_id=data_name, colormap=colormap,
                                                      hover_notions=[("pat_id", dict_pat_ids[data_name]),
                                                                     ("label", dict_vox_labels[data_name]),
                                                                     ("index", np.arange(len(dict_vox_labels[data_name]))),
                                                                     ("cluster", dict_clusters[data_name].astype(np.int32))],
                                                      colorby=colorby,
                                                      cmap_interval=np.max(colorby_data[data_name]), xlabel="dimension #1", ylabel="dimension #1",
                                                      title="Title", mode="pacmap", plot_func="scatter",
                                                      postfix=f"colored by {colorby_name} - {data_name}",
                                                      save_dir=path.dirname(ori_data2), scatter_size=0.5)
                    
                
        else:
            if "dataset1" in path.basename(presaved_prj_file):
                pat_ids = pat_ids1
                clusters = clusters1
                labels = labels1
                dataset_name = "Dataset1"
            elif "dataset2" in path.basename(presaved_prj_file):
                pat_ids = pat_ids2
                clusters = clusters2
                labels = labels2
                dataset_name = "Dataset2"
                
            load_info = pd.read_csv(presaved_prj_file, header=None).values
            load_dataset_names = load_info[:, 0]
            load_pat_ids = load_info[:, 1]
            load_clusters = load_info[:, 2]
            load_labels = load_info[:, 3]
            X_transformed = load_info[:, 4:]

            assert np.sum(load_pat_ids == pat_ids) == len(
                combined_pat_ids), "order of data is missed up!"
            assert np.sum(load_labels == labels) == len(
                combined_vox_labels), "order of data is missed up!"
            assert np.sum(load_clusters == clusters) == len(
                combined_clusters), "order of data is missed up!"

            for colorby, colorby_name, colorby_data in zip(["pat_id", "label", "cluster"],
                                                           ["patient IDs", "voxel labels", "patient clusters",
                                                            load_pat_ids, load_labels, load_clusters]):
                postfix = "(0-Pseudo Progress, 1-true progress, 2-dataset1)"
                interactive_bokeh_with_select(X_transformed[:, 0], X_transformed[:, 1], indiv_id=dataset_name,
                                              colormap="Category10",
                                              hover_notions=[("pat_id", pat_ids), ("label", labels),
                                                             ("index", np.arange(len(labels))),
                                                             ("cluster", clusters.astype(np.int32))], colorby=colorby,
                                              cmap_interval=np.max(colorby_data), xlabel="dimension #1",
                                              ylabel="dimension #1", title="Title", mode="pacmap", plot_func="scatter",
                                              postfix=f"colored by {colorby_name} of {colorby_name}",
                                              save_dir=path.dirname(ori_data2))
                ## plot rightaway with selected points  # load_selected_bokeh_points_plot(features1, "selected-data (6)", region=[-5,-9], dataset_name="Dataset1")
        

elif plot_name == "certain_samples":
    from scipy.io import loadmat as loadmat
    import scipy.io as io
    
    ### load the first version of data
    """
        1= voxel from affected hemisphere of tumor patient;
        2= voxel from healthy hemisphere of tumor patient;
        0= voxel from both hemisphere of the patient suffering from smth. else.
        """
    ori_data1 = r"C:\Users\LDY\Desktop\metabolites-0301\metabolites_tumour_classifier\data\20190325\2019.03.25-DATA.mat"
    mat1 = loadmat(ori_data1)["DATA"]
    pat_ids1 = mat1[:, 0].astype(np.int)
    labels1 = mat1[:, 1].astype(np.int)
    features1 = mat1[:, 2:]
    new_mat1 = np.zeros((mat1.shape[0], mat1.shape[1] + 1))
    new_mat1[:, 0] = np.arange(mat1.shape[0])  # tag every sample
    new_mat1[:, 1:] = mat1
    train_data = {}
    test_data = {}
    
    class_names = ["healthy", "tumor"]
    class_colors = ["lightblue", "violet"]
    class_dark = ["darkblue", "crimson"]
    
    files = find_files(data_dir, pattern=file_patterns)
    # certain_mat = np.empty((0, new_mat.shape[1]))
    certain_inds_tot = np.empty((0))
    for data_fn in files:
        certain = pd.read_csv(data_fn, header=0).values
        certain_inds = certain[:, 0].astype(np.int32)
        certain_inds_tot = np.append(certain_inds_tot, certain_inds)
        print(path.basename(data_fn), len(certain_inds), "samples\n")
    
    uniq_inds = np.unique(certain_inds_tot).astype(np.int32)
    certain_mat = new_mat[uniq_inds]
    
    # np.savetxt(path.join(data_dir, "certain_samples_lout40_fold5[smp_id,pat_id,label,meta]_class(0-1)=({}-{}).csv".format(len(np.where(certain_mat[:,2]==0)[0]), len(np.where(certain_mat[:,2]==1)[0]))), certain_mat, delimiter=",", fmt="%.5f")
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
    #     plt.savefig(path.join(data_dir, "certain_samples_class{}.png".format(c)))
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
                plt.setp(axs[j // 5, np.mod(j, 5)].get_yticklabels(),
                         visible=False)
            
            f.text(0.5, 0.05, 'index'),
            f.text(0.02, 0.5, 'Normalized amplitude', rotation=90,
                   verticalalignment='center'),
            f.subplots_adjust(hspace=0, wspace=0)
            plt.savefig(
                path.join(data_dir,
                             "certain_samples_class{}_fig_{}.png".format(c, ii)))
            plt.close()


elif plot_name == "distill_valid_labels":
    from scipy import stats

    print("Plot_name: ", plot_name)
    # Nenad validated the labels of these samples
    m_file = "C:/Users/LDY/Desktop/all-experiment-results/metabolites/20190325-certain-Validate.mat"
    mat = loadmat(m_file)["Validate"]  # [id, label, features]
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
    plt.savefig(path.join(path.dirname(ccr_summary), "optimal-theta-with-validated-labels-ccr-{:.4f}-theta{:.2f}.png".format(rocDF["threshold"].values[0], scipy.stats.percentileofscore(ccr_samp_ccr, threshold))))
    plt.savefig(path.join(path.dirname(ccr_summary), "optimal-theta-with-validated-labels-ccr-{}-theta{:.2f}.pdf".format(rocDF["threshold"].values[0], scipy.stats.percentileofscore(ccr_samp_ccr, threshold))),  format="pdf")
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
    plt.savefig(path.join("C:/Users/LDY/Desktop", "CCR-distribution-of-NP-validated-samples-p{:.2E}.png".format(p_rank)))
    plt.savefig(path.join("C:/Users/LDY/Desktop", "CCR-distribution-of-NP-validated-samples-p{:.2E}.pdf".format(p_rank)), format="pdf")
    plt.close()
    print("ok")


elif plot_name == "re_split_data_0_9_except_5":
    # shuffle all data and split. Not patient specific
    print("Plot_name: ", plot_name)
    src_data = ["data{}".format(jj) for jj in [0,1,2,3,4]]
    data_source_dirs = [
        "../data/20190325/5_fold_pat_split_20190325-2class_test_{}.mat".format(
            src_dir) for src_dir in src_data]

    # each cross_validation set
    coll_mat = np.empty((0, 290))
    for dd in data_source_dirs:
        ## load original .mat data and split train_val
        mat = loadmat(dd)["DATA"]
        coll_mat = np.vstack((coll_mat, mat[mat[:,1]!=2]))
    np.random.shuffle(coll_mat)
    print("ok")
    for cc in range(3):
        print("class ", cc, np.sum(coll_mat[:, 1] == cc))
    total_pat_ids = Counter(coll_mat[:, 0])
    np.random.shuffle(coll_mat)
    test_part = {key: {} for key in range(5)}
    part_len = len(coll_mat) // 5
    
    for ii in range(4):
        test_part[ii]["DATA"] = coll_mat[ii * part_len: (ii + 1) * part_len]
        num_pat = len(Counter(test_part[ii]["DATA"][:, 0]))
        savemat(
            path.join(path.dirname(dd),
                         "5_fold_randshuffle-2class_test_data{}.mat".format(ii)), test_part[ii])
        Count_test = Counter(test_part[ii]["DATA"][:, 0])
        np.savetxt(path.join(path.dirname(dd), "5_fold_randshuffle-2class[{}-{}]_test_data{}_summary_pat_{}.csv".format(
                                     np.sum(test_part[ii]["DATA"][:, 1] == 0),
                                     np.sum(test_part[ii]["DATA"][:, 1] == 1), ii, num_pat)), np.array([[pat, Count_test[pat]] for pat in Count_test.keys()]), delimiter=",", fmt="%d")
    ii = 4
    test_part[ii]["DATA"] = coll_mat[ii * part_len:]
    Count_test = Counter(test_part[ii]["DATA"][:, 0])
    num_pat = len(Count_test)
    np.savetxt(path.join(path.dirname(dd),
                            "5_fold_randshuffle-2class[{}-{}]_test_data{}_summary_pat_{}.csv".format(
                                np.sum(test_part[ii]["DATA"][:, 1] == 0),
                                np.sum(test_part[ii]["DATA"][:, 1] == 1), ii,
                                num_pat)),
               np.array([[pat, Count_test[pat]] for pat in Count_test.keys()]),
               delimiter=",", fmt="%d")
    savemat(
        path.join(path.dirname(dd),
                     "5_fold_randshuffle-2class_test_data{}.mat".format( ii)), test_part[ii])


    # merge the other 4 sets to form train_validation set
    train_val_folds = {key: {} for key in range(5)}

    for jj in range(5):   # for each fold, combine other folds to get train
        current_train_folds = list(np.arange(5))
        current_train_folds.remove(jj)

        curren_train_coll = {"DATA": np.empty((0, 290))}
        for ind in current_train_folds:
            curren_train_coll["DATA"] = np.vstack((curren_train_coll["DATA"], test_part[ind]["DATA"]))

        savemat(
            path.join(path.dirname(dd),
                         "5_fold_randshuffle-2class_train_val_data{}.mat".format(jj)), curren_train_coll)

        Count_train = Counter(curren_train_coll["DATA"][:, 0])
        num_pat = len(Count_train)
        np.savetxt(path.join(path.dirname(dd),
                                "5_fold_randshuffle-2class[{}-{}]_train_val_data{}_summary_pat_{}.csv".format(
                                    np.sum(curren_train_coll["DATA"][:, 1] == 0),
                                    np.sum(curren_train_coll["DATA"][:, 1] == 1), jj,
                                    num_pat)),
                   np.array(
                       [[pat, Count_train[pat]] for pat in Count_train.keys()]),
                   delimiter=",", fmt="%d")


elif plot_name == "re_split_data_0_9_except_5_patient_wise_get_data_statistics":
    print("Plot_name: ", plot_name)
    src_data = ["data{}".format(jj) for jj in [0,1,2,3,4,6,7,8,9]]
    data_dir_root = "../data/20190325"
    data_source_dirs = [path.join(data_dir_root, "20190325-3class_lout40_test_{}.mat".format(
                src_dir)) for src_dir in src_data]

    # each cross_validation set
    coll_mat = np.empty((0, 290))
    for dd in data_source_dirs:
        ## load original .mat data and split train_val
        mat = loadmat(dd)["DATA"]
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
        np.savetxt(path.join(data_dir_root,"{}_fold_pat_split_20190325-2class_train_val_data{}_patient_ids_{}.csv".format(n_cv_folds, cv_fold, len(trian_val_pat_ids))), trian_val_pat_ids, header="pat_id,count,percentile{}".format(n_percentile_folds), fmt="%.3f", delimiter=",")
        np.savetxt(path.join(data_dir_root,"{}_fold_pat_split_20190325-2class_test_data{}_patient_ids_{}.csv".format(n_cv_folds, cv_fold, len(test_pat_ids))), test_pat_ids, header="pat_id,count,percentile{}".format(n_percentile_folds), fmt="%.3f", delimiter=",")
        
        new_CV_fold_test_data = {"DATA": np.empty((0, 290))}  #np.empty((0, 290))
        for pat_id in test_pat_ids[:, 0]:
            new_CV_fold_test_data["DATA"] = np.vstack((new_CV_fold_test_data["DATA"], coll_mat[per_pat_spectra_inds[pat_id]]))

        savemat(path.join(data_dir_root,"{}_fold_pat_split_20190325-2class_train_val_data{}.mat".format(n_cv_folds, cv_fold)), new_CV_fold_train_val_data)
        savemat(path.join(data_dir_root, "{}_fold_pat_split_20190325-2class_test_data{}.mat".format(n_cv_folds, cv_fold)), new_CV_fold_test_data)
        print(path.join(data_dir_root, "5_fold_pat_split_20190325-2class_test_data{}.mat".format(cv_fold)))

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
    ori = loadmat(original)["DATA"]
    true_label = ori[:, 1]
    
    # Get human's total labels
    human_rating = "../data/20190325/human-ratings-20190325-3class_lout40_val_data5-2class.mat"
    hum_whole = loadmat(human_rating)["data_ratings"]
    human_lb = hum_whole[:, 0]
    human_features = hum_whole[:, 1:]
    hum_fpr, hum_tpr, _ = metrics.roc_curve(true_label, human_lb)
    hum_auc = metrics.roc_auc_score(true_label, human_lb)
    hum_score = metrics.roc_auc_score(true_label, human_lb)
    
    tpr = hum_tpr[1]
    fpr = hum_fpr[1]

    get_auc_from_d_prime(tpr, fpr)


#
#
# elif plot_name == "100_single_ep_corr_classification_rate_mnist_old":
#     """
#     Get the correct classification rate with 100 runs of single-epoch-training
#     """
#     import ipdb
#     from scipy.stats import spearmanr
#
#     data_dirs = [
#         "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/9-train-with-MNIST-MLP/2021-02-19T17-42-25--MLP-both_meanx0-factor-0-from-mnist-ctFalse-theta-1-s9311-100rns-train-trainOnTrue", ]
#     num_smp_dataset = {"data0": 8357, "data1": 8326, "data2": 8566, "data3": 8454, "data4": 8440, "data5": 8231,
#                        "data6": 8371, "data7": 8357, "data8": 8384, "data9": 7701, "mnist": 60000}
#
#     for data_dir in data_dirs:
#         files = find_files(data_dir, pattern="one*.csv")
#
#         spearmanr_rec = []
#         # get correct count in 100 rauns
#         data_source = path.basename(files[0]).split("_")[-4]
#         for ind in tqdm(range(len(files))):
#             values = pd.read_csv(files[ind], header=0).values
#             smp_ids = values[:, 0].astype(np.int)
#             pat_ids = values[:, 1].astype(np.int)
#             lbs = values[:, 2]
#             prob = values[:, 3:]
#             if ind == 0:  # the first file to get the total number (3-class) of samples
#                 total_num = num_smp_dataset[data_source]  # 3-class samples id
#                 ids_w_count = []
#                 noisy_lb_counts = []
#                 dict_count = {key: 0 for key in np.arange(total_num)}  # total number 9243
#                 noisy_lb_rec = {key: 0 for key in np.arange(total_num)}  # total number 9243
#                 pre_rank = np.arange(total_num)  # indices
#
#             pred_lbs = np.argmax(prob, axis=1)
#             right_inds = np.where(pred_lbs == lbs)[0]
#             # right_inds = np.where(pred_lbs == pat_ids)[0]
#             correct = np.unique(smp_ids[right_inds])
#             ids_w_count += list(correct)
#             # noisy_lb_counts += list(np.unique(smp_ids[pat_ids != lbs]))  #  it should be the same for every file
#
#
#             if ind % 10 == 0:
#                 count_all = Counter(ids_w_count)
#                 dict_count.update(count_all)
#                 curr_count_array = np.array([[key, val] for (key, val) in dict_count.items()])
#                 curr_rank = curr_count_array[np.argsort(curr_count_array[:, 1]), 0]
#                 # spearmanr_rec.append([ind, np.sum(curr_rank==pre_rank)])
#                 spearmanr_rec.append([ind, spearmanr(pre_rank, curr_rank)[0]])
#                 pre_rank = curr_rank.copy()
#
#         count_all = Counter(ids_w_count)
#         dict_count.update(count_all)
#         # noisy_lb_rec.update(Counter(noisy_lb_counts))
#         noisy_lb_counts = list(np.unique(smp_ids[pat_ids != lbs]))  # it should be the same for every file
#
#         noisy_lb_rec.update(Counter(noisy_lb_counts))
#
#         counter_array = np.array([[key, dict_count[key]] for key in np.arange(total_num)])
#         noisy_inds_array = np.array([[key, noisy_lb_rec[key]] for key in np.arange(total_num)])
#         ipdb.set_trace()
#         # noisy_inds_array = np.array([[key, val] for (key, val) in noisy_lb_rec.items()])
#         sort_inds = np.argsort(counter_array[:, 1])
#         sample_ids_key = counter_array[sort_inds][:, 0]
#         # rates = counter_array[sort_inds, 1]/counter_array[:, 1].max()
#         rates = counter_array[sort_inds][:, 1] / len(files)
#         noisy_lb_rate = noisy_inds_array[sort_inds][:, 1]
#
#         assert np.sum(counter_array[sort_inds, 0] == noisy_inds_array[sort_inds, 0]), "sorted sample indices mismatch"
#
#         fig, ax1 = plt.subplots()
#         ax1.set_xlabel("sample sorted by the correct clf. rate"),
#         ax1.set_ylabel("correct clf. rate (over 100 runs)"),
#         ax1.plot(rates, label="original data set"),
#         ax1.tick_params(axis='y'),
#         ax1.set_ylim([0, 1.0])
#         ax1.legend(loc="upper left")
#
#         ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#         ax2.set_ylabel('counts'),  # we already handled the x-label with ax1
#         ax2.plot(noisy_lb_rate.cumsum(), "m", label="accum. # of noisy labels"),
#         ax2.plot(np.ones(total_num).cumsum(), "c", label="accum. # of all samples")
#         ax2.set_ylim([0, total_num])
#         ax2.tick_params(axis='y')
#         ax2.legend(loc="upper right")
#         plt.title("distillation effect-{}.png".format(data_source))
#         plt.savefig(
#             data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}.png".format(
#                 path.basename(files[0]).split("_")[-6], total_num, data_source)),
#         plt.savefig(
#             data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}.pdf".format(
#                 path.basename(files[0]).split("_")[-6], total_num, data_source), format="pdf")
#         print("ok")
#         plt.close()
#
#     concat_data = np.concatenate((np.array(sort_inds).reshape(-1, 1), rates.reshape(-1, 1)), axis=1)
#     np.savetxt(data_dir + "/full_summary-{}_100_runs_sort_inds_rate_({}-{}).csv".format(data_source, path.basename(
#         files[0]).split("_")[7], total_num), concat_data, fmt="%.5f", delimiter=",",
#                header="ori_sort_rate_id,ori_sort_rate,true_lbs,noisy_lbs")
#
#
# elif plot_name == "100_single_ep_corr_classification_rate_mnist_old2":
#     """
#     Get the correct classification rate with 100 runs of single-epoch-training
#     """
#     import ipdb
#     from scipy.stats import spearmanr
#
#     data_dirs = [
#         "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/9-train-with-MNIST-MLP/2021-02-10T13-20-28--MLP-both_meanx0-factor-0-from-mnist-certainFalse-theta-1-s379-100rns-train"]
#     num_smp_dataset = {"data0": 8357, "data1": 8326, "data2": 8566, "data3": 8454, "data4": 8440, "data5": 8231,
#                        "data6": 8371, "data7": 8357, "data8": 8384, "data9": 7701, "mnist": 60000}
#
#     for data_dir in data_dirs:
#         files = find_files(data_dir, pattern="one*.csv")
#
#         spearmanr_rec = []
#         print("number of files: ", len(files))
#         data_source = path.basename(files[0]).split("_")[-4]
#         for ind in tqdm(range(len(files))):
#             fn = find_files(data_dir, pattern="one_{}*.csv".format(ind))
#             values = pd.read_csv(fn[0], header=0).values
#             smp_ids = values[:, 0].astype(np.int)
#             pat_ids = values[:, 1].astype(np.int)
#             lbs = values[:, 2]
#             prob = values[:, 3:]
#             if ind == 0:  # the first file to get the total number (3-class) of samples
#                 total_num = num_smp_dataset[data_source]  # 3-class samples id
#                 ids_w_count = []
#                 noisy_lb_counts = []
#                 dict_count = {key: 0 for key in np.arange(total_num)}  # total number 9243
#                 noisy_lb_rec = {key: 0 for key in np.arange(total_num)}  # total number 9243
#                 pre_rank = np.arange(total_num)  # indices
#
#             pred_lbs = np.argmax(prob, axis=1)
#             right_inds = np.where(pred_lbs == pat_ids)[0]
#             correct = np.unique(smp_ids[right_inds])
#             ids_w_count += list(correct)
#             noisy_lb_counts += list(smp_ids[pat_ids != lbs])  # sample ids that with noisy labels
#
#             if ind % 10 == 0:
#                 count_all = Counter(ids_w_count)
#                 dict_count.update(count_all)
#                 curr_count_array = np.array([[key, val] for (key, val) in dict_count.items()])
#                 curr_rank = curr_count_array[np.argsort(curr_count_array[:, 1]), 0]
#                 # spearmanr_rec.append([ind, np.sum(curr_rank==pre_rank)])
#                 spearmanr_rec.append([ind, spearmanr(pre_rank, curr_rank)[0]])
#                 pre_rank = curr_rank.copy()
#
#         count_all = Counter(ids_w_count)
#         dict_count.update(count_all)
#         noisy_lb_rec.update(Counter(noisy_lb_counts))
#
#         counter_array = np.array([[key, val] for (key, val) in dict_count.items()])
#         noisy_inds_array = np.array([[key, val / len(files)] for (key, val) in noisy_lb_rec.items()])
#         sort_inds = np.argsort(counter_array[:, 1])
#         sample_ids_key = counter_array[sort_inds, 0]
#         rates = counter_array[sort_inds, 1] / len(files)
#         noisy_lb_rate = noisy_inds_array[sort_inds, 1]
#
#         ipdb.set_trace()
#         assert np.sum(counter_array[sort_inds, 0] == noisy_inds_array[
#             sort_inds, 0]) == total_num, "sorted sample indices mismatch"
#
#         fig, ax1 = plt.subplots()
#         ax1.set_xlabel("sample sorted by the correct clf. rate"),
#         ax1.set_ylabel("correct clf. rate (over 100 runs)"),
#         ax1.plot(rates, label="original data set"),
#         ax1.tick_params(axis='y'),
#         ax1.set_ylim([0, 1.0])
#         ax1.legend(loc="upper left")
#
#         ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#         ax2.set_ylabel('counts'),  # we already handled the x-label with ax1
#         ax2.plot(noisy_lb_rate.cumsum(), "m", label="accum. # of noisy labels"),
#         ax2.plot(np.ones(total_num).cumsum(), "c", label="accum. # of all samples")
#         ax2.set_ylim([0, total_num])
#         ax2.tick_params(axis='y')
#         ax2.legend(loc="upper right")
#         plt.title("distillation effect-{}.png".format(data_source))
#         plt.savefig(
#             data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}.png".format(
#                 path.basename(files[0]).split("_")[-5], total_num, data_source)),
#         plt.savefig(
#             data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}.pdf".format(
#                 path.basename(files[0]).split("_")[-5], total_num, data_source), format="pdf")
#         print("ok")
#         plt.close()
#
#     ipdb.set_trace()
#     concat_data = np.concatenate((np.array(sort_inds).reshape(-1, 1), rates.reshape(-1, 1)), axis=1)
#     np.savetxt(data_dir + "/full_summary-{}_100_runs_sort_inds_rate_({}-{}).csv".format(data_source, path.basename(
#         files[0]).split("_")[-4], total_num), concat_data, fmt="%.5f", delimiter=",",
#                header="ori_sort_rate_id,ori_sort_rate,true_lbs,noisy_lbs")
