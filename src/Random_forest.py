# Random Forest Classifier

# Importing the libraries
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy.io import loadmat
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap


class Configure(object):
    def __init__(self):
        self.train_dir = "../data/20190325/20190325-3class_lout40_train_test_data5.mat"
        self.test_dir = "../data/20190325/20190325-3class_lout40_val_data5.mat"
        self.num_classes = 2
        self.val_ratio = 0.25
        self.aug_method = "ops_mean"
        self.aug_folds = 1
        self.aug_scale = 0.35


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
        elif args.aug_method == "mean":
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
def get_data(config):
    """
    Load self_saved data. A dict, data["features"], data["labels"]. See the save function in split_data_for_val()
    # First time preprocess data functions are needed: split_data_for_val(args),split_data_for_lout_val(args)
    :param config: Param object with path to the data
    :return:
    """
    mat = loadmat(config.train_dir)["DATA"]
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

    # if config.test_or_train == 'train':
    #     temp_rand = np.arange(len(labels))
    #     np.random.shuffle(temp_rand)
    #     spectra_rand = spectra[temp_rand]
    #     labels_rand = labels[temp_rand]
    #     ids_rand = ids[temp_rand]
    # elif config.test_or_train == 'test':  # In test, don't shuffle
    #     spectra_rand = spectra
    #     labels_rand = labels
    #     ids_rand = ids
    #     print("data labels: ", labels_rand)
    assert config.num_classes != np.max(labels), "The number of class doesn't match the data!"

    X_train, X_val, Y_train, Y_val = train_test_split(spectra, labels, test_size=config.val_ratio, random_state=132)

    train_data["spectra"] = X_train.astype(np.float32)
    train_data["labels"] = Y_train.astype(np.int32)
    # train_data["ids"] = np.squeeze(ids_rand[0:config.num_train]).astype(np.int32)

    print("num_samples: ", len(Y_val))

    ## oversample the minority samples ONLY in training data
    train_data = augment_with_batch_mean(config, train_data["spectra"], train_data["labels"], train_data)
    print("After augmentation--num of train class 0: ", len(np.where(train_data["labels"] == 0)[0]),
          "num of train class 1: ",
          len(np.where(train_data["labels"] == 1)[0]))
    train_data = oversample_train(train_data, config.num_classes)
    print("After oversampling--num of train class 0: ", len(np.where(train_data["labels"] == 0)[0]),
          "num of train class 1: ", len(np.where(train_data["labels"] == 1)[0]))
    train_data["num_samples"] = len(train_data["labels"])
    return train_data["spectra"], train_data["labels"], X_val, Y_val


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


##########################################################################
config = Configure()
root = "../Random_forest_results"
time_str = '{0:%Y-%m-%dT%H-%M-%S-}'.format(datetime.datetime.now())
save_dir = os.path.join(root, time_str+'lout40-5')
subdirs = ["model"]
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    for subdir in subdirs:
        os.makedirs(save_dir + '/{}'.format(subdir))

# Splitting the dataset into the training set and val set
X_train, Y_train, X_val, Y_val = get_data(config)

# datasets = pd.read_csv('../data/20ind10325/social_network_ads.csv')
# X = datasets.iloc[:, [2,3]].values
# Y = datasets.iloc[:, 4].values
# X_train, X_val, Y_train, Y_val = train_test_split(X, Y, val_size = 0.25, random_state = 0)
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_val = sc_X.transform(X_val)

# Fitting the classifier into the training set
classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the val set results
Y_Pred = classifier.predict(X_val)

# Making the Confusion Matrix

cm = confusion_matrix(Y_val, Y_Pred)

# plot feature importance
importance = classifier.feature_importances_
import_inds = np.argsort(importance)
plt.figure()
plt.plot(importance, label="importance")
plt.savefig(os.path.join(save_dir, "RF_importance.png"))
plt.close()

ind0, ind1 = 171, 65
new_train = np.vstack((X_train[:, ind0], X_train[:, ind1])).T
new_val = np.vstack((X_val[:, ind0], X_val[:, ind1])).T
clf = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
clf.fit(new_train, Y_train)
# Visualising the training set results

X_Set, Y_Set = new_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01), np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.25, cmap = ListedColormap(('royalblue', 'violet')))
plt.xlim(X1.min(), X1.max()),
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1], c = ListedColormap(('darkblue', 'm'))(i), label = j)
plt.title('Random Forest Classifier (training set)'),
plt.xlabel('#1 important feature')
plt.ylabel('#2 important feature')
plt.legend(),
plt.savefig(os.path.join(save_dir, "RF_exaple_train.png"))
plt.close()


fig = plt.figure()
ax = fig.add_subplot(111)
jet = colors.Colormap('jet')
cNorm  = colors.Normalize(vmin=0, vmax=287)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
lines = []
for ind, idx in enumerate(import_inds):
    ax.scatter(import_inds[ind], importance[import_inds[ind]], color=scalarMap.to_rgba(ind))
plt.savefig(os.path.join(save_dir, "RF_importance.png")),

# Visualising the val set results

X_Set, Y_Set = new_val, Y_val
X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.20, cmap = ListedColormap(('royalblue', 'violet')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
                c = ListedColormap(('darkblue', 'm'))(i), label = j)
plt.title('Random Forest Classifier (Validation set)')
plt.xlabel('#1 important feature')
plt.ylabel('#2 important feature')
plt.legend()
plt.savefig(os.path.join(save_dir, "RF_exaple_val.png"))
plt.close()


# Validation SET
X_test, Y_test = get_test_data(config)
Y_pred = classifier.predict(X_test)
cm_test = confusion_matrix(Y_test, Y_pred)

new_test = np.vstack((X_test[:, ind0], X_test[:, ind1])).T
X_Set, Y_Set = new_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.20, cmap = ListedColormap(('royalblue', 'violet')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
                c = ListedColormap(('darkblue', 'm'))(i), label = j)
plt.title('Random Forest Classifier (Test set)')
plt.xlabel('#1 important feature')
plt.ylabel('#2 important feature')
plt.legend()
plt.savefig(os.path.join(save_dir, "RF_exaple_test.png"))
plt.close()
new = np.vstack((Y_test, Y_pred)).T
