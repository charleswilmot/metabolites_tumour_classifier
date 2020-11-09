# Random Forest Classifier

# Importing the libraries
import os
import datetime
import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from sklearn.ensemble import RandomForestClassifier
from scipy.io import loadmat
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.model_selection import RandomizedSearchCV

from dataio import get_data
import utils

class Configure(object):
    def __init__(self):
        self.input_data = "../data/20190325/20190325-3class_lout40_train_test_data5.mat"
        self.test_data = "../data/20190325/20190325-3class_lout40_val_data5.mat"
        self.num_classes = 2

        self.train_or_test = "train"
        self.num_classes = 2
        self.batch_size = 32
        self.test_bs = 128
        self.rand_seed = 988
        self.aug_method = "same-mean"
        self.aug_scale = 0
        self.aug_folds = 0
        self.from_epoch = 0
        self.theta_thr = 1
        self.output_path = None
        self.certain_dir = "C:\\Users\\LDY\\Desktop\\metabolites-0301\\metabolites_tumour_classifier\\rsults\\2020-09-01T21-35-28-None-meanx0-factor-0-from-ep-0-from-lout40-data5-theta-0.95-train\\certains",
        self.resume_training = False
        self.output_root = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review"
        self.input_data = "../data/20190325/20190325-3class_lout40_train_test_data5.mat"
        self.model_save_dir = None
        self.data_source = "data5"
        self.data_mode = "metabolites"
        self.data_dim = "1d"
        self.if_single_runs = False
        self.from_clusterpy = False

        self.model_name = "MLP"
        self.new_folder = "1-Pure"
        self.optimizer_type = "adam"
        self.loss_type = "softmax_ce"
        self.learning_rate = 0.0002
        self.test_ratio = 0.25
        self.postfix = "none"
        self.if_from_certain = False
        self.if_save_certain = False
        self.distill_old = False



def augment_with_batch_mean(args, spec, true_labels, train):
    """
    Augment the original spectra with the mini-mini-same-class-batch mean
    :param labels_aug:
    :param spec:
    :param spec_aug:
    :param true_labels:
    :return:
    """
    num2average = 1
    for class_id in range(args.num_classes):
        # find all the samples from this class
        if args.aug_method == "ops_mean":
            inds = np.where(true_labels == args.num_classes-1-class_id)[0]
        elif args.aug_method == "same_mean":
            inds = np.where(true_labels == class_id)[0]

        # randomly select 100 groups of 100 samples each and get mean
        aug_inds = np.random.choice(inds, inds.size*num2average, replace=True).reshape(-1, num2average)  # random pick 10000 samples and take mean every 2 samples
        mean_batch = np.mean(spec[aug_inds], axis=1)   # get a batch of spectra to get the mean for aug

        new_mean = (mean_batch - np.mean(mean_batch, axis=1)[:, np.newaxis]) / np.std(mean_batch, axis=1)[:, np.newaxis]
        # new_mean = (mean_batch - np.max(mean_batch, axis=1)[:, np.newaxis]) / (np.max(mean_batch, axis=1) - np.min(mean_batch, axis=1))[:, np.newaxis]
        # rec_inds = np.random.choice(inds, aug_folds * 100, replace=True).reshape(aug_folds, 100)  # aug receiver inds

        for fold in range(args.aug_folds):
            aug_zspec = (1 - args.aug_scale) * spec[inds] + new_mean[np.random.choice(new_mean.shape[0], inds.size)] * args.aug_scale
            train["spectra"] = np.vstack((train["spectra"], aug_zspec))
            train["labels"] = np.append(train["labels"], true_labels[inds])
            # ids_aug = np.append(ids_aug, true_ids[inds])
        print("original spec shape class : ", class_id, spec.shape, "augment spec shape: ", train["spectra"].shape)

    return train


# Importing the datasets
# def get_data(config):
#     """
#     Load self_saved data. A dict, data["features"], data["labels"]. See the save function in split_data_for_val()
#     # First time preprocess data functions are needed: split_data_for_val(args),split_data_for_lout_val(args)
#     :param config: Param object with path to the data
#     :return:
#     """
#     mat = loadmat(config.train_dir)["DATA"]
#     spectra = mat[:, 2:]
#     labels = mat[:, 1]
#     ids = mat[:, 0]
#     train_data = {}
#     val_data = {}
#     spectra = zscore(spectra, axis=1)
#     ## following code is to get only label 0 and 1 data from the file. TODO: to make this more easy and clear
#     if config.num_classes - 1 < np.max(labels):
#         need_inds = np.empty((0))
#         for class_id in range(config.num_classes):
#             need_inds = np.append(need_inds, np.where(labels == class_id)[0])
#         need_inds = need_inds.astype(np.int32)
#         spectra = spectra[need_inds]
#         labels = labels[need_inds]
#         ids = ids[need_inds]
#
#     assert config.num_classes != np.max(labels), "The number of class doesn't match the data!"
#
#     X_train, X_val, Y_train, Y_val = train_test_split(spectra, labels, test_size=config.val_ratio, random_state=132)
#
#     train_data["spectra"] = X_train.astype(np.float32)
#     train_data["labels"] = Y_train.astype(np.int32)
#     # train_data["ids"] = np.squeeze(ids_rand[0:config.num_train]).astype(np.int32)
#
#     print("num val samples: ", len(Y_val))
#
#     ## oversample the minority samples ONLY in training data
#     if config.aug_folds > 0:
#         train_data = augment_with_batch_mean(config, train_data["spectra"], train_data["labels"], train_data)
#         print("After augmentation--num of train class 0: ", len(np.where(train_data["labels"] == 0)[0]),
#               "num of train class 1: ",
#               len(np.where(train_data["labels"] == 1)[0]))
#     train_data = oversample_train(train_data, config.num_classes)
#     print("After oversampling--num of train class 0: ", len(np.where(train_data["labels"] == 0)[0]),
#           "num of train class 1: ", len(np.where(train_data["labels"] == 1)[0]))
#     train_data["num_samples"] = len(train_data["labels"])
#     return train_data["spectra"], train_data["labels"], X_val, Y_val
#

def oversample_train(train_data, num_classes):
    """
    Oversample the minority samples
    :param train_data:"spectra", 2d array, "labels", 1d array
    :return:
    """
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=34)
    X_resampled, y_resampled = ros.fit_resample(train_data["spectra"], train_data["labels"])
    train_data["spectra"] = X_resampled
    train_data["labels"] = y_resampled

    return train_data


def get_test_data(config):
    mat = loadmat(config.test_dir)["DATA"]
    spectra = mat[:, 2:]
    labels = mat[:, 1]
    ids = mat[:, 0]
    train_data = {}
    val_data = {}
    spectra = zscore(spectra, axis=1)
    ## following code is to get only label 0 and 1 data from the file. TODO: to make this more easy and clear
    if config.num_classes - 1 < np.max(labels):
        need_inds = np.empty((0))
        for class_id in range(config.num_classes):
            need_inds = np.append(need_inds, np.where(labels == class_id)[0])
        need_inds = need_inds.astype(np.int32)
        spectra = spectra[need_inds]
        labels = labels[need_inds]
        ids = ids[need_inds]
    return spectra, labels


def plot_confusion_matrix(conf, num_classes=2, ifnormalize=False, postfix="test", save_dir="results/"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    :param confm: confusion matrix
    :param num_classes: int, the number of classes
    :param normalize: boolean, whether normalize to (0,1)
    :return:
    """
    TN = conf[0][0]
    FN = conf[1][0]
    TP = conf[1][1]
    FP = conf[0][1]
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    if ifnormalize:
        cm = (conf * 1.0 / conf.sum(axis=1)[:, np.newaxis])*1.0
    else:
        cm = conf
    f = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)
    plt.colorbar()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] != 0:
            plt.text(j, i, np.int(cm[i, j]*100)/100.0, horizontalalignment="center", color="darkorange", size=20)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title("Confusion matrix on {} set, SEN-{}, SPE-{}".format(postfix, TPR, TNR))
    f.savefig(save_dir + '/confusion_matrix_{}.png'.format(postfix))
    plt.close()


def plot_importance(model):
    """
    PLot color coded importance
    :param model:
    :return:
    """
    importance = model.feature_importances_
    import_inds = np.argsort(importance)
    fig = plt.figure(figsize=[12, 10])
    ax = fig.add_subplot(111)
    colors = pylab.cm.jet(np.linspace(0, 1, 288))
    lines = []
    for ind, idx in enumerate(import_inds):
        ax.scatter(import_inds[ind], importance[import_inds[ind]], color=colors[ind]),
        ax.plot(importance, '-.')
    plt.xlabel("importance")
    plt.xlabel("metabolite indices")
    plt.savefig(os.path.join(save_dir, "RF_importance.png"))
    plt.close()


def visualize_top_2_RF(X_Set, Y_Set, model, num_classes=2, postfix="val", save_dir="results/"):
    """
    plot the decision map with the two variables
    :param X_Set:
    :param Y_Set:
    :param model:
    :param postfix:
    :param save_dir:
    :return:
    """
    X1, X2 = np.meshgrid(
        np.arange(start=X_Set[:, 0].min() - 1, stop=X_Set[:, 0].max() + 1, step=0.01),
        np.arange(start=X_Set[:, 1].min() - 1, stop=X_Set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.25,
                 cmap=ListedColormap(('royalblue', 'violet')))
    plt.xlim(X1.min(), X1.max()),
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.arange(num_classes)):
        plt.scatter(X_Set[X_Set == j, 0], X_Set[Y_Set == j, 1], c=ListedColormap(('darkblue', 'm'))(i), label=j)
    plt.title('Random Forest Classifier (training set)'),
    plt.xlabel('#1 important feature')
    plt.ylabel('#2 important feature')
    plt.legend(),
    plt.savefig(os.path.join(save_dir, "RF_exaple_{}.png".format(postfix)))
    plt.close()
    

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
##########################################################################

if __name__ == "__main__":
    config = Configure()

    root = "../Random_forest_results"
    time_str = '{0:%Y-%m-%dT%H-%M-%S-}'.format(datetime.datetime.now())
    config.output_path = os.path.join(root, time_str+'lout40-5')
    subdirs = ["model"]
    hyper_search = True
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
        for subdir in subdirs:
            os.makedirs(config.output_path + '/{}'.format(subdir))
            
    # Splitting the dataset into the training set and val set
    train_data, test_data = get_data(config)
    
    if hyper_search:
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        print(random_grid)
        
        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestClassifier()
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        # Fit the random search model
        rf_random.fit(train_data["spectra"], train_data["labels"])
        
        base_model = RandomForestClassifier(n_estimators = 300, random_state = 42)
        base_model.fit(train_data["spectra"], train_data["labels"])
        base_accuracy = evaluate(base_model, test_data["spectra"], test_data["labels"])
        
        best_random = rf_random.best_estimator_
        random_accuracy = evaluate(best_random, test_data["spectra"], test_data["labels"])
        print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
    else:
    
        # Fitting the classifier into the training set
        classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
        classifier.fit(train_data["spectra"], train_data["labels"])
    
        # Predicting the val set results
        Y_Pred = classifier.predict(test_data["spectra"])
    
        # Making the Confusion Matrix
        cm = confusion_matrix(test_data["labels"], Y_Pred)
        plot_confusion_matrix(cm, num_classes=2, ifnormalize=False, postfix="val", save_dir=config.output_path)
    
        # plot feature importance
        plot_importance(classifier)
    
        # ind0, ind1 = 171, 65
        # new_train = np.vstack((X_train[:, ind0], X_train[:, ind1])).T
        # clf = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
        # clf.fit(new_train, Y_train)
        #
        # # Visualising the training set results
        # X_Set_train, Y_Set_train = new_train, Y_train
        # visualize_top_2_RF(X_Set_train, Y_Set_train, clf, postfix="train", save_dir=save_dir)
        #
    
        # # Visualising the val set results
        # new_val = np.vstack((X_val[:, ind0], X_val[:, ind1])).T
        # X_Set_val, Y_Set_val = new_val, Y_val
        # visualize_top_2_RF(X_Set_val, Y_Set_val, clf, postfix="val", save_dir=save_dir)
        # new_test = np.vstack((X_test[:, ind0], X_test[:, ind1])).T
        # X_Set_test, Y_Set_test = new_test, Y_test
        # visualize_top_2_RF(X_Set_test, Y_Set_test, clf, postfix="test", save_dir=save_dir)
    
        # Test SET
        X_test, Y_test = get_test_data(config)
        Y_pred = classifier.predict(X_test)
        cm_test = confusion_matrix(Y_test, Y_pred)
        plot_confusion_matrix(cm_test, num_classes=2, ifnormalize=False, postfix="test", save_dir=config.output_path)
    
        print("ok")
        
        
