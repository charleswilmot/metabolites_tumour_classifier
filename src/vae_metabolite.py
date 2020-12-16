#### This is a script treat the eeg data(2250 clips) as sequence data and apply VAE to encode feateures
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as scipyio
from scipy.stats import zscore

import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from tqdm import tqdm

import ipdb
import datetime


### Hyperparams
class Config():
    def __init__(self):
        self.height, self.width = 288, 1
        self.hid_dim1 = 256
        self.hid_dim2 = 64
        self.latent_dim = 2  #0.98
        self.batch_size = 64
        self.num_epochs = 101
        self.num_classes = 2


def plot_prior(sess, args, load_model=False, save_name='save_name'):
    #if load_model:
        #saver.restore(sess, os.path.join(os.getcwd(), logdir + '/' + "{}".format(model_No)))
    nx = ny = 4
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)
    canvas = np.empty((args.height * ny, args.width * nx))
    noise = tf.random_normal([1, args.latent_dim])
    z = tf.placeholder(tf.float32, [1, args.latent_dim], 'dec_input')
    reconstruction = decoder(z)
    latent = np.random.randn(1, args.latent_dim)
    #sess2 = tf.Session()
    #sess2.run(tf.global_variables_initializer())
    for ii, yi in enumerate(x_values):
      for j, xi in enumerate(y_values):
        latent[0, 0:2] = xi, yi  #sess.run(reconstruction, {z_2: np_z, X: np_x_fixed})
        x_reconstruction = sess.run(reconstruction, feed_dict={z: latent})
        canvas[(nx - ii - 1) * args.height:(nx - ii) * args.height, j *
                                                               args.width:(j + 1) * args.width] = x_reconstruction.reshape(args.height, args.width)
    plt.savefig(save_name, format="jpg")   # canvas
                        
def upsample(inputs, name='depool', factor=[2,2]):
    size = [int(inputs.shape[1] * factor[0]), int(inputs.shape[2] * factor[1])]
    with tf.name_scope(name):
        out = tf.image.resize_bilinear(inputs, size=size, align_corners=None, name=None)
    return out


def plot_test(original, reconstruction, load_model = False, save_name="save"):
    # Here, we plot the reconstructed image on test set images.
    #if load_model:
        #saver.restore(sess, os.path.join(os.getcwd(), logdir + '/' + "{}".format(model_No)))
    num_pairs = 10
    for pair in range(num_pairs):
        #reshaping to show original test image
        x_image = np.reshape(original[pair, :], (28,28))
        index = (1 + pair) * 2
        ax1 = plt.subplot(5,4,index - 1)  # arrange in 5*4 layout
        plt.imshow(x_image, aspect="auto")
        if pair == 0 or pair == 1:
            plt.title("Original")
        plt.xlim([0, 27])
        plt.ylim([27, 0])

        x_reconstruction_image = np.reshape(reconstruction[pair, :], (28,28))
        ax2 = plt.subplot(5,4,index, sharex = ax1, sharey=ax1)
        plt.imshow(x_reconstruction_image, aspect="auto")
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.xlim([0, 27])
        plt.ylim([27, 0])
        plt.tight_layout()
        if pair == 0 or pair == 1:
            plt.title("Reconstruct")
            
    plt.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.95,
                wspace=0.30, hspace=0.22)
    plt.savefig(save_name + "samples.png", format="png")
    plt.close()


def load_data(data_dir, args):
    mat = scipyio.loadmat(data_dir)["DATA"]
    spectra = mat[:, 2:]
    labels = mat[:, 1]
    ids = mat[:, 0]
    spectra = zscore(spectra, axis=1)
    ## following code is to get only label 0 and 1 data from the file. TODO: to make this more easy and clear
    if args.num_classes - 1 < np.max(labels):
        need_inds = np.empty((0))
        for class_id in range(args.num_classes):
            need_inds = np.append(need_inds, np.where(labels == class_id)[0])
        need_inds = need_inds.astype(np.int32)
        spectra = spectra[need_inds]
        labels = labels[need_inds]
        ids = ids[need_inds]

    return spectra, labels

############################ Encoder ############################

def encoder(inputs_enc, config, num_filters=[32, 64, 64, 64], kernel_size=[3, 3], pool_size=[2, 2], scope=None):
    """parameters from
    https://www.kaggle.com/rvislaywade/visualizing-mnist-using-a-variational-autoencoder
    def encoder_net(x, latent_dim, h_dim):
    Construct an inference network parametrizing a Gaussian.
    Args:
    x: A batch of real data (MNIST digits).
    latent_dim: The latent dimensionality.
    hidden_size: The size of the neural net hidden layers.
    Returns:
    mu: Mean parameters for the variational family Normal
    sigma: Standard deviation parameters for the variational family Normal
    often in convolution padding = 'same', in max_pooling padding = 'valid'
    """

    with tf.variable_scope(scope, 'encoder'):
        inputs_enc = tf.reshape(inputs_enc, [-1,  config.height, config.width, 1])
        net = inputs_enc
    # Convolutional Layer 
        for layer_id, num_outputs in enumerate(num_filters):   ## avoid the code repetation
            with tf.variable_scope('block_{}'.format(layer_id)) as layer_scope:
                net = tf.layers.conv2d(
                                        inputs = net,
                                        filters = num_outputs,
                                        kernel_size = kernel_size,
                                        padding='SAME',
                                        activation=tf.nn.relu)
                #net = tf.layers.max_pooling2d(inputs=net, pool_size=pool_size, padding='SAME', strides=2)
                net = tf.compat.v1.layers.BatchNormalization()
                print(net.shape)
                
        ### dense layer
        with tf.name_scope("dense"):
            net = tf.compat.v1.layers.dropout(inputs=net, rate=0.75)
            net = tf.reshape(net, [-1,  net.shape[1]*net.shape[2]*net.shape[3]])
            net = tf.compat.v1.layers.dense(inputs=net, units=config.hid_dim2, activation=tf.nn.relu)
        print(net.shape)
        ### Get mu
        mu_1 = tf.compat.v1.layers.dense(net, config.latent_dim, activation_fn=None)
        # layer 2   Output mean and std of the latent variable distribution
        sigma_1 = tf.compat.v1.layers.dense(net, config.latent_dim, activation_fn=None)
        # Reparameterize import Randomness
        noise = tf.random_normal([1, config.latent_dim])
        # z_1 is the fisrt leverl output(latent variable) of our Encoder
        z_1 = mu_1 + tf.multiply(noise, tf.exp(0.5*sigma_1))
        print(z_1.shape)
        return mu_1, sigma_1, z_1     #dense1
                                   #

def decoder(inputs_dec, config, num_filters=[25, 1], kernel_size=5, scope=None):
    """Build a generative network parametrizing the likelihood of the data
    Args:
    inputs_dec: Samples of latent variables with size latent_dim_2
    hidden_size: Size of the hidden state of the neural net
    Returns:
    reconstruction: logits for the Bernoulli likelihood of the data
    """
    net = inputs_dec
    print(net.shape)
    with tf.variable_scope(scope, 'dec'):
        with tf.name_scope('dec_fc_dropout'):
            net = tf.layers.dense(inputs=net, units=config.hid_dim2, activation=tf.nn.relu)
            net = tf.layers.dropout(inputs=net, rate=0.75, name='dec_dropout1')
            net = tf.layers.dense(inputs=net, units=14 * 14 * 25, activation=tf.nn.relu)
            net = tf.layers.dropout(inputs=net, rate=0.75, name='dec_dropout2')
            net = tf.reshape(net, [-1, 14, 14, num_filters[0]])
            print(net.shape)
            ########### deconvolution layer
            net = upsample(net)
            for layer_id, num_outputs in enumerate(num_filters):
                with tf.variable_scope('block_{}'.format(layer_id)) as layer_scope:
                    net = tf.layers.conv2d(
                                                        inputs = net,
                                                        filters = num_outputs,
                                                        kernel_size = kernel_size,
                                                        padding='SAME',
                                                        activation=tf.nn.sigmoid)

                    print(net.shape)
            #assert len(shape) == len(output_dim), 'shape mismatch'
            #### reconstruction activ = sigmoid
            reconstruction  = tf.reshape(net, [-1, config.height * config.width])
            
            return reconstruction


def train(input_enc):
    args = Config()
    version = "vae_cnn"

    root = "results_vae"
    time_str = '{0:%Y-%m-%dT%H-%M-%S-}'.format(datetime.datetime.now())
    save_dir = os.path.join(root, time_str + 'lout40-5-500samples-' + version)
    logdir = os.path.join(save_dir, "model")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)



    with tf.name_scope("Data"):
        ### Get data
        train_data, train_data = load_data("/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data5.mat")
        test_data, test_labels = load_data("/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_val_data5.mat")

        # create TensorFlow Dataset objects
        dataset_train = tf.data.Dataset.from_tensor_slices((train_data, train_data)).repeat().batch(args.batch_size).shuffle(buffer_size=5000)
        dataset_test = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).repeat().batch(args.batch_size).shuffle(buffer_size=5000)
        iter = dataset_train.make_initializable_iterator()
        iter_test = dataset_test.make_initializable_iterator()
        ele = iter.get_next()   #you get the filename
        ele_test = iter_test.get_next()   #you get the filename

    ### Graph
    mu_1, sigma_1, z = encoder(inputs_enc, args)
    reconstruction = decoder(z, args)

    # Loss function = reconstruction error + regularization(similar image's latent representation close)
    with tf.name_scope('loss'):
        Log_loss = tf.reduce_sum(inputs_enc  * tf.log(reconstruction + 1e-7) + (1 - inputs_enc ) * tf.log(1 - reconstruction + 1e-7), reduction_indices=1)
        KL_loss = -0.5 * tf.reduce_sum(1 + 2*sigma_1 - tf.pow(mu_1, 2) - tf.exp(2 * sigma_1), reduction_indices=1)
        VAE_loss = tf.reduce_mean(Log_loss - KL_loss)

    #Outputs a Summary protocol buffer containing a single scalar value.
    # tf.summary.histogram("vae/mu", mu_1)
    # tf.summary.histogram("vae/sigma", sigma_1)
    # tf.summary.scalar('VAE_loss', VAE_loss)
    # tf.summary.scalar('KL_loss1', tf.reduce_mean(KL_loss))
    # tf.summary.scalar('Log_loss1', tf.reduce_mean(Log_loss))
    test_loss = tf.Variable(0.0)
    # test_loss_sum = tf.summary.scalar('test_loss', test_loss)

    optimizer = tf.compat.v1.train.AdadeltaOptimizer().minimize(-VAE_loss)

    #################### Set up logging for TensorBoard.
    # Training  init all variables and start the session!
    init = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session()
    sess.run(init)
    sess.run(iter.initializer)
    sess.run(iter_test.initializer)
    ## Add ops to save and restore all the variables.
    saver = tf.compat.v1.train.Saver()

    #store value for these 3 terms so we can plot them later
    vae_loss_array = []
    test_vae_array = []
    log_loss_array = []
    KL_loss_array = []
    #iteration_array = [i*recording_interval for i in range(num_iterations/recording_interval)]

    ### get the real data
    num_classes = 2
    for ep in tqdm(range(args.epochs)):

        # save_name = config.results_dir + '/' + "_step{}_".format( batch)
        data_train, labels_train =  sess.run(ele)   # names, 1s/0s
        hot_labels_train =  np.eye((num_classes))[labels_train.astype(int)]   # get one-hot lable

        #run our optimizer on our data
        sess.run([optimizer], feed_dict={inputs_enc: data_train})
        ### test
        if (ep % 10 == 0):
            ##################### test #######################################
            data_test, labels_test = sess.run(ele_test)  # names, 1s/0s
            hot_labels_test =  np.eye((num_classes))[labels_test.astype(int)]   # get one-hot lable
                
            test_temp = VAE_loss.eval({input_enc : data_test})
            test_vae_array = np.append(test_vae_array, test_temp)

            reconstruction_test = reconstruction.eval({input_enc: test_data[0:10]})
            #every 1K iterations record these values
            temp_vae = VAE_loss.eval(feed_dict={inputs_enc: data_test})
            temp_log = np.mean(Log_loss.eval(feed_dict={inputs_enc: data_test}))
            temp_KL = np.mean(KL_loss.eval(feed_dict={inputs_enc: data_test}))
            vae_loss_array.append(temp_vae )
            KL_loss_array.append(temp_KL)
            log_loss_array.append( temp_log)
            print("Iteration: {}, Loss: {}, log_loss: {}, KL_term {}".format(ep, temp_vae, temp_log, temp_KL ))

            plot_test(test_data[0:10], reconstruction_test, save_name=save_dir + '/test_reconstruction.png')
            ########### plot prior
            nx = ny = 8
            x_values = np.linspace(-1, 1, nx)
            y_values = np.linspace(-1, 1, ny)
            canvas = np.zeros((args.height * ny, args.width * nx))
            z_sample = tf.placeholder(tf.float32)
            for ii, yi in enumerate(x_values):
                for jj, xi in enumerate(y_values):
                    latent = np.array([[xi, yi]])  # sess.run(reconstruction, {z_2: np_z, X: np_x_fixed})
                    x_reconstruction = sess.run(reconstruction, feed_dict={z: latent})
                    canvas[ii * args.height:(ii + 1) * args.height,
                    jj * args.width:(jj + 1) * args.width] = x_reconstruction.reshape(args.height, args.width)
            plt.savefig(save_dir + 'prior.png', format="jpg")  # canvas


            # Save model
            saver.save(sess, logdir + '/epoch-' + str(ep) + '-model.ckpt')

            # Plot all losses
            plt.figure()
            plt.plot(np.arange(len(vae_loss_array)), vae_loss_array, color = 'orchid', label='vae_los')
            plt.plot(np.arange(len(vae_loss_array)),  KL_loss_array, color = 'c', label='KL_loss')
            plt.plot(np.arange(len(vae_loss_array)),  log_loss_array, color = 'b', label='log_likelihood')
            plt.xlabel("training ")
            plt.legend(loc="best")
            plt.savefig(save_dir+"losses_iter{}.png".format(ep), format="png")
            plt.title('Loss during training')
            plt.close()
            
            plt.figure()
            plt.plot(np.arange(len(test_vae_array)),  test_vae_array, color = 'darkcyan')
            plt.title('Loss in test')
            plt.savefig(save_dir+"test_loss_iter{}.png".format(ep), format="png")
            plt.close()
            #func.plot_learning_curve(test_vae_array, test_vae_array, num_trial=1, save_name=resultdir + "/learning_curve.png")
            #func.plot_learning_curve(np.array(vae_loss_array), save_name=results_dir + "/loss_in_training.png")

        
if __name__ == "__main__":
    #### input
    with tf.name_scope("input"):
        ## the real data from database
        inputs_enc = tf.placeholder(tf.float32, [None, 288 * 1], name='inputs_enc')

    train(inputs_enc)
