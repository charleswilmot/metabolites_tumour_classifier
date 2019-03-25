'''
define Hparams space
create model with the Hparams
save names with the values of the Hparams
save the best model with the Hparams
'''

import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
# import math
import os
import fnmatch
import scipy.io
import pandas as pd

# from tf.keras.models import Sequential  # This does not work!
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
# from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
# from tensorflow.python.keras.models import load_model

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
# from skopt.plots import plot_convergence
# from skopt.plots import plot_objective, plot_evaluations
# from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args

# from tensorflow.examples.tutorials.mnist import input_data

dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
                             name='learning_rate')
dim_num_dense_layers = Integer(low=1, high=4, name='num_dense_layers')
dim_num_dense_nodes = Integer(low=50, high=512, name='num_dense_nodes')
dim_activation = Categorical(categories=['relu', 'lrelu'],
                             name='activation')
dim_kernel_size = Integer(low=2, high=100, name='kernel_size')
dim_num_filters_cnn1 = Integer(low=8, high=128, name='num_filters_cnn1')
dim_num_filters_cnn2 = Integer(low=8, high=128, name='num_filters_cnn2')
dim_num_filters_cnn3 = Integer(low=8, high=128, name='num_filters_cnn3')
dim_drop_rate = Real(low=0.2, high=0.85, name='drop_rate')


dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_dense_nodes,
              dim_activation,
              dim_kernel_size,
              dim_num_filters_cnn1,
              dim_num_filters_cnn2,
              dim_num_filters_cnn3,
              dim_drop_rate]
default_parameters = [1e-5, 1, 50, 'relu', 9, 8, 16, 32, 0.5]



def find_files(data_dir, file_pattern='*.csv'):
    """
    Find all the files in one directory with pattern in the filenames and perform train_test_split, and save file names seperately.
    :param args.data_dir: str, the directory of the files
    :param args.class_mode: str, "1EPG (BL-EPG)", "3EPG (BL-earlyEPG-middleEPG-lateEPG)"
    :param args.test_ratio: the ratio of whole data used for testing
    :param args.num_hours_per_class: how many hours to choose for training
    :param save_dir:
    :return: test_files, list all the testing files
    :return: train_files, list all the training files
    """
    train_files_labels = []
    test_files_labels = []
    ## get the number of files in folders
    for root, dirnames, fnames in os.walk(data_dir):
        if os.path.basename(root) == "BL":
            fnames = fnmatch.filter(fnames, file_pattern)
            train_files_labels, test_files_labels = get_train_test_files_split(root, fnames, 0.2,
                                                                               train_files_labels, test_files_labels,
                                                                               label=0,
                                                                               num2use=8)
        elif os.path.basename(root) == "EPG":
            fnames = fnmatch.filter(fnames, file_pattern)
            train_files_labels, test_files_labels = get_train_test_files_split(root, fnames, 0.2, train_files_labels,
                                                                               test_files_labels, label=1, num2use=8)
        else:
            print("No file is found under mode")
    
    np.random.shuffle(train_files_labels)
    np.random.shuffle(test_files_labels)
    np.savetxt(os.path.join("Optimize", "test_files.txt"), np.array(test_files_labels), fmt="%s", delimiter=",")
    np.savetxt(os.path.join("Optimize", "train_files.txt"), np.array(train_files_labels), fmt="%s", delimiter=",")
    return train_files_labels, test_files_labels


def get_train_test_files_split(root, fns, ratio, train_list, test_list, label=0, num2use=100):
    """
    Get equal number of files for testing from each folder
    :param fns: list, all file names from the folder
    :param ratio: float, the test file ratio.
    :param train_list: the list for training files
    :param test_list: the list for testing files
    :param label: int, the label need to be assigned to the file
    :param num2use: int, the number of files that you want to use(randomize file selection)
    :return: lists, editted train and test file lists
    """
    np.random.shuffle(fns)
    if num2use > len(fns):
        num_files = len(fns)
    else:
        num_files = num2use

    num_test_files = np.ceil(ratio * num_files).astype(np.int)

    train_within_folder = []
    for ind, f in enumerate(fns[0:num_files]):
        if ind < num_test_files:
            test_list.append((os.path.join(root, f), label))
        else:
            train_list.append((os.path.join(root, f), label))
            train_within_folder.append((os.path.join(root, f), label))

    if ratio != 1.0:
        if num2use > len(fns):  # only oversampling training data
            theo_num_train = np.int(num2use * (1 - ratio))
            repeat_times = theo_num_train // len(train_within_folder) - 1 # already have one round of those files
            train_within_folder = train_within_folder * repeat_times + train_within_folder[0: theo_num_train - (len(train_within_folder) * repeat_times )]
            for ind, fn in enumerate(train_within_folder):
                train_list.append((fn[0], label) )

    return train_list, test_list


def log_dir_name(learning_rate, num_dense_layers,
                 num_dense_nodes, activation, kernel_size, num_filters_cnn1, num_filters_cnn2, num_filters_cnn3, drop_rate):

    # The dir-name for the TensorBoard log-dir.
    s = "-lr_{0:.0e}-layers_{1}-nodes_{2}-act_{3}-kernel_{4}_-filterCNN1_{5}-filterCNN2_{6}-filterCNN3_{7}-drop_{8}-acc-"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate,
                       num_dense_layers,
                       num_dense_nodes,
                       activation,
                       kernel_size,
                       num_filters_cnn1,
                       num_filters_cnn2,
                       num_filters_cnn3,
                       drop_rate)

    return log_dir


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_example_errors(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Boolean array whether the predicted class is incorrect.
    incorrect = (cls_pred != data.test.cls)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def get_data():
    """
    Load self_saved data. A dict, data["features"], data["labels"]. See the save function in split_data_for_val()
    # First time preprocess data functions are needed: split_data_for_val(args),split_data_for_lout_val(args)
    :param args: Param object with path to the data
    :return:
    """
    data_dir = "../data/20190301-3class_lout30_train_test_data10.mat"
    mat = scipy.io.loadmat(data_dir)["DATA"]
    spectra = mat[:, 2:]
    labels = mat[:, 1]
    ids = mat[:, 0]
    train_data = {}
    test_data = {}
    num_classes = 2
    ## following code is to get only label 0 and 1 data from the file. TODO: to make this more easy and clear
    if num_classes - 1 < np.max(labels):
        need_inds = np.empty((0))
        for class_id in range(num_classes):
            need_inds = np.append(need_inds, np.where(labels == class_id)[0])
        need_inds = need_inds.astype(np.int32)
        spectra = spectra[need_inds]
        labels = labels[need_inds]
        ids = ids[need_inds]

    test_ratio = 10
    temp_rand = np.arange(len(labels))
    np.random.shuffle(temp_rand)
    spectra_rand = spectra[temp_rand]
    labels_rand = labels[temp_rand]
    ids_rand = ids[temp_rand]
    assert num_classes != np.max(labels), "The number of class doesn't match the data!"
    num_train = int(((100 - test_ratio) * spectra_rand.shape[0]) // 100)

    test_data["spectra"] = spectra_rand[num_train:].astype(np.float32)
    lb_int = np.squeeze(labels_rand[num_train:]).astype(np.int32)
    test_data["labels"] = np.eye(num_classes)[lb_int]
    test_data["ids"] = np.squeeze(ids_rand[num_train:]).astype(np.int32)

    ## oversample the minority samples ONLY in training data
    train_data["spectra"] = spectra_rand[0:num_train].astype(np.float32)
    train_data["labels"] = np.squeeze(labels_rand[0:num_train]).astype(np.int32)
    train_data = oversample_train(train_data, num_classes)

    # train_data["labels"] = np.eye(num_classes)[np.squeeze(train_data["labels"][0:num_train]).astype(np.int32)]

    return train_data, test_data


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
    train_data["labels"] = np.eye(num_classes)[np.squeeze(y_resampled).astype(np.int32)]


    return train_data


def create_model(learning_rate, num_dense_layers,
                 num_dense_nodes, activation, kernel_size, num_filters_cnn1, num_filters_cnn2, num_filters_cnn3, drop_rate):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """
    
    # Start construction of a Keras Sequential model.
    model = Sequential()

    # Add an input layer which is similar to a feed_dict in TensorFlow.
    # Note that the input-shape must be a tuple containing the image-size.
    model.add(InputLayer(input_shape=(img_size_flat,)))

    # The input from MNIST is a flattened array with 784 elements,
    # but the convolutional layers expect images with shape (28, 28, 1)
    model.add(Reshape(img_shape_full))

    # Add the CNN layer to the model.
    # This has two hyper-parameters we want to optimize:
    # The number of nodes and the activation function.
    model.add(Conv2D(kernel_size=(kernel_size, 1), strides=(4, 1), filters=num_filters_cnn1, padding='same', activation=activation, name='conv1'))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(drop_rate))
    # There are many hyper-parameters in this layer, but we only
    # want to optimize the activation-function in this example.
    model.add(Conv2D(kernel_size=(kernel_size, 1), strides=(2, 1), filters=num_filters_cnn2, padding='same', activation=activation, name='conv2'))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(drop_rate))
    # Second convolutional layer.
    # Again, we only want to optimize the activation-function here.
    model.add(Conv2D(kernel_size=(kernel_size, 1), strides=(2, 1), filters=num_filters_cnn3, padding='same',  activation=activation, name='conv3'))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(drop_rate))

    # Flatten the 4-rank output of the convolutional layers
    # to 2-rank that can be input to a fully-connected / dense layer.
    model.add(Flatten())

    # Add fully-connected / dense layers.
    # The number of layers is a hyper-parameter we want to optimize.
    for i in range(num_dense_layers):
        # Name of the layer. This is not really necessary
        # because Keras should give them unique names.
        name = 'layer_dense_{0}'.format(i+1)

        # Add the dense / fully-connected layer to the model.
        # This has two hyper-parameters we want to optimize:
        # The number of nodes and the activation function.
        model.add(Dense(num_dense_nodes,
                        activation=activation,
                        name=name))

    # Last fully-connected / dense layer with softmax-activation
    # for use in classification.
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    
    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = Adam(lr=learning_rate)
    
    # In Keras we need to compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    
    return model


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers,
            num_dense_nodes, activation, kernel_size, num_filters_cnn1, num_filters_cnn2, num_filters_cnn3, drop_rate):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """

    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_dense_layers:', num_dense_layers)
    print('num_dense_nodes:', num_dense_nodes)
    print('activation:', activation)
    print('kernel_size:', kernel_size)
    print('num_filters_cnn1:', num_filters_cnn1)
    print('num_filters_cnn2:', num_filters_cnn2)
    print('num_filters_cnn3:', num_filters_cnn3)
    print('drop_rate:', drop_rate)
    print()

    
    # Create the neural network with these hyper-parameters.
    model = create_model(learning_rate=learning_rate,
                         num_dense_layers=num_dense_layers,
                         num_dense_nodes=num_dense_nodes,
                         activation=activation,
                         kernel_size=kernel_size,
                         num_filters_cnn1=num_filters_cnn1,
                         num_filters_cnn2=num_filters_cnn2,
                         num_filters_cnn3=num_filters_cnn3,
                         drop_rate=drop_rate,
                         )

    # Dir-name for the TensorBoard log-files.
    log_dir = log_dir_name(learning_rate, num_dense_layers,
                           num_dense_nodes, activation, kernel_size, num_filters_cnn1, num_filters_cnn2, num_filters_cnn3, drop_rate)
    
    # Create a callback-function for Keras which will be
    # run after each epoch has ended during training.
    # This saves the log-files for TensorBoard.
    # Note that there are complications when histogram_freq=1.
    # It might give strange errors and it also does not properly
    # support Keras data-generators for the validation-set.
    # callback_log = TensorBoard(
    #     log_dir=log_dir,
    #     histogram_freq=0,
    #     batch_size=32,
    #     write_graph=True,
    #     write_grads=False,
    #     write_images=False)
   
    # Use Keras to train the model.
    history = model.fit(x=train_data["spectra"],
                        y=train_data["labels"],
                        epochs=20,
                        batch_size=64,
                        validation_data=validation_data)

    # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    accuracy = history.history['val_acc'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()

    # Save the model if it improves on the best-found performance.
    # We use the global keyword so we update the variable outside
    # of this function.
    global best_accuracy

    # If the classification accuracy of the saved model is improved ...
    if accuracy > best_accuracy:
        # Save the new model to harddisk.
        # model.save(path_best_model)
        model.summary(print_fn=myprint)
        np.savetxt("Optiresults/" + log_dir + "{}".format(accuracy), np.arange(10).reshape(5, 2), header="-lr_{0:.0e}-layers_{}-nodes_{}-act_{}-kernel_{}_-CNN1_{}-CNN2_{}-CNN3_{}-drop_{}-acc-".format(learning_rate,
                       num_dense_layers,
                       num_dense_nodes,
                       activation,
                       kernel_size,
                       num_filters_cnn1,
                       num_filters_cnn2,
                       num_filters_cnn3,
                       drop_rate, accuracy))
        # Update the classification accuracy.
        best_accuracy = accuracy

    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    
    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.
    return -accuracy


if __name__ == "__main__":
    best_accuracy = 0.0

    # Number of classes, one class for each of 10 digits.
    num_classes = 2

    train_data, test_data = get_data()
    # data = input_data.read_data_sets('data/MNIST/', one_hot=True)
    print("Size of:")
    print("- Training-set:\t\t{}".format(len(train_data["labels"])))
    print("- Test-set:\t\t{}".format(len(test_data["labels"])))
    validation_data = (test_data["spectra"], test_data["labels"])

    # We know that MNIST images are 28 pixels in each dimension.
    img_size = 288

    # Images are stored in one-dimensional arrays of this length.
    img_size_flat = 288

    # Tuple with height and width of images used to reshape arrays.
    # This is used for plotting the images.
    img_shape = (img_size, 1)

    # Tuple with height, width and depth used to reshape arrays.
    # This is used for reshaping in Keras.
    img_shape_full = (img_size, 1, 1)

    # Number of colour channels for the images: 1 channel for gray-scale.
    num_channels = 1

    # fitness(x=default_parameters)
    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='EI',  # Expected Improvement.
                                n_calls=500,
                                x0=default_parameters)