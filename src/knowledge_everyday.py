17.08.2017
# short time fourier transform
https://kevinsprojects.wordpress.com/2014/12/13/short-time-fourier-transform-using-python-and-numpy/
STFT Algorithm:

So, we understand what we’re trying to make – now we have to figure out how to make it.  The data flow we have to achieve is pretty simple, as we only need to do the following steps:

    Pick out a short segment of data from the overall signal
    Multiply that segment against a half-cosine function
    Pad the end of the segment with zeros
    Take the Fourier transform of that segment and normalize it into positive and negative frequencies
    Combine the energy from the positive and negative frequencies together, and display the one-sided spectrum
    Scale the resulting spectrum into dB for easier viewing
    Clip the signal to remove noise past the noise floor which we don’t care about


# activation functions and their derivatives
def nonlin(self, z, deriv=False, nonlinearFun='relu'):
    #prime = 1000 * self.sigmoid_like(x) * (1 - self.sigmoid_like(x))
    if nonlinearFun=='relu':
        if deriv:
            return 1. * (z - 0 > 0)
        else:
            return z * (z - 0 > 0)
    if nonlinearFun=='leakyrelu':
        leaky = 0.01
        if deriv:
            return 1. * (z - 0 > 0) + leaky*(z-0<0)
        else:
            return z * (z - 0 > 0) + leaky*(z-0<0)
    if nonlinearFun=='sigmoid':
        if deriv:
            return 1 / (1+np.exp(-z)) * (1 - 1 / (1+np.exp(-z)))
        else:
            return 1 / (1+np.exp(-z))
    if nonlinearFun=='softmax':
        if deriv:
            return np.exp(z) / np.sum(np.exp(z)) - (np.exp(z) / (np.sum(np.exp(z))))**2
        else:
            return np.exp(z) / np.sum(np.exp(z), axis=0)
    if nonlinearFun=='tanh':
        if deriv:
            return (1-z**2)
        else:
            return np.tanh(z)

# matrix derivatives
Y = A * X
dYdX = A.T

Y = X * A
dYdX = A

Y = A.T * X * B
DYDX = A*B.T

Y = A.T * X.T * B
DYDX = B * A.T

D(X.T*A) = (DX.T/DX)*A + X.T(DA/DX) = I*A + X.T*0 = A

# what is the betst way to make a copy as new set of parameters?????

# make bptt_step safer from no-initialization
T = 5
bptt_truncate_step = 3
for t in np.arange(T)[::-1]:
    for bptt in np.arange(max(0, t-bptt_truncate_step), t+1)[::-1]:
        print("step {}, bptt step {}".format(t, bptt))
'''step 4, bptt step 4
step 4, bptt step 3
step 4, bptt step 2
step 4, bptt step 1
step 3, bptt step 3
step 3, bptt step 2
step 3, bptt step 1
step 3, bptt step 0
step 2, bptt step 2
step 2, bptt step 1
step 2, bptt step 0
step 1, bptt step 1
step 1, bptt step 0
step 0, bptt step 0
'''

# to_one_hot
def to_one_hot(num_classes, labels):
    '''Make int label into one-hot encoding
    Param:
    num_classes: int, number of classes
    labels: 1D array'''
    ret = np.eye(num_classes)[labels]

    return ret

23.10.2017
nice-plots: 3.5. Validation curves: plotting scores to evaluate models
http://scikit-learn.org/stable/modules/learning_curve.html

24.10.2017
1. Replacing spaces in the file names
    >>rename -n "s/ /_/g" *
2. plot with repeated xlabel:
    plt.figure()
    plt.xticks(range(100), np.tile(np.arange(10), 10))

27.10.2017
plt spectrogram
def plot_specgram(self, frames, sampFreq, title="Spectrogram", save_name="spectrogram", ifsave=False):
    plt.figure()
    cmap = plt.get_cmap('viridis') # this may fail on older versions of matplotlib
    vmin = -40  # hide anything below -40 dB
    cmap.set_under(color='k', alpha=None)

    sampFreq, frames = wavfile.read("song.wav")
    fig, ax = plt.subplots()
    if len(frames.shape) != 1:
        frames = frames[:, 0]     # first channel
    pxx, freq, t, cax = ax.specgram(frames,
                                    Fs=sampFreq,      # to get frequency axis in Hz
                                    cmap=cmap, vmin=vmin)
    cbar = fig.colorbar(cax)
    cbar.set_label('Intensity dB')
    ax.axis("tight")

    # Prettify
    import matplotlib
    import datetime

    ax.set_xlabel('time h:mm:ss')
    ax.set_ylabel('frequency kHz')

    scale = 1e3                     # KHz
    ticks = matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))
    ax.yaxis.set_major_formatter(ticks)

    def timeTicks(x, pos):
        d = datetime.timedelta(seconds=x)
        return str(d)
    formatter = matplotlib.ticker.FuncFormatter(timeTicks)
    ax.xaxis.set_major_formatter(formatter)

Paper:
    FOrmal models of language learning----nice open ended learning chart

2017.11.13
count occerrence of values in an array
>>np.unique([1, 1, 2, 2, 3, 3, 4])
array([1, 2, 3, 4])

Launching TensorBoard
To run TensorBoard, use the following command (alternatively >>python -m tensorboard.main)
>>tensorboard --logdir=path/to/log-directory
TensorBoard 0.1.8 at http://digda:6006 (Press CTRL+C to quit)


2017.11.14
deal with the first nonvalid step
# Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=args.max_checkpoints)

    try:
        saved_global_step = load(saver, sess, restore_from)
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1

    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)

    step = None
    last_saved_step = saved_global_step
    try:
        for step in range(saved_global_step + 1, args.num_steps):
            ...
    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()

2017.11.16

'''plot like animation'''
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)   # plot original
plt.ion()
plt.show()
while in training:
    try:   # in case it is the first time, you can't do the operation
        ax.lines.remove(lines[0])
    except:
        pass

    lines = ax.plt(x_data, prediction, 'r-')
    plt.pause(0.1)

optimizer:
    SGD:
        W += -learning_rate * dx   # zig zag to the mountain feet
    Momentum:
        m = b1 * m - learning_rate * dx  # inertia go downwards
        W += m
    AdaGrad:
        v += dx^2
        W += -learning_rate * dx / sqrt(v)   # bad shoe, so go straight line
    RMSProp:   # combine momentum and AdaGrad
        v = b1 * v - (1 - b1) * dx^2
        W += -learning_rate * dx / sqrt(v)
    Adam: # add the -learning_rate*dx part from momentum in RMSProp
        m = b1 * m - (1 - b1) * dx -----------Momentum
        v = b2 * v - (1 - b2) * dx^2 ---------AdaGrad
        W += -learning_rate * m / sqrt(v)

tensorboard:
    sess = tf.Session()
    merged = tf.merge_all)summaries()
    write = tf.train.SummaryWriter("logs/", sess.graph)

    for step....:

        if step % 50 == 0:
            result = sess.run(merged, feed_dict={x: x_data, y: y_data})
            writer.add_summary(result, step)

2017.11.21 deal with out of memory problem

SLURM:
    >>srun -p x-men -c 10 python3 train.py --data_dir=corpus --gc_channels=32 --restore_from logdir/train/2017-11-20T11-05-18

Commands
'salloc' is used to allocate resources for a job in real time. Typically this is used to allocate resources and spawn a shell. The shell is then used to execute srun commands to launch parallel tasks.
'sbatch' is used to submit a job script for later execution. The script will typically contain one or more srun commands to launch parallel tasks.
'scancel #JOB number' is used to cancel a pending or running job or job step. It can also be used to send an arbitrary signal to all processes associated with a running job or job step.
'sinfo' reports the state of partitions and nodes managed by Slurm. It has a wide variety of filtering, sorting, and formatting options.
'smap' reports state information for jobs, fftpartitions, and nodes managed by Slurm, but graphically displays the information to reflect network topology.
'squeue' reports the state of jobs or job steps. It has a wide variety of filtering, sorting, and formatting options. By default, it reports the running jobs in priority order and then the pending jobs in priority order.
'srun' is used to submit a job for execution or initiate job steps in real time. srun has a wide variety of options to specify resource requirements, including: minimum and maximum node count, processor count, specific nodes to use or not use, and specific node characteristics (so much memory, disk space, certain required features, etc.). A job can contain multiple job steps executing sequentially or in parallel on independent or shared resources within the job's node allocation.

2017-11-23
train wavenet
Generate wavenet:
    'with no global condition'
    >>python3 generate.py --num_samples 16000  --wav_out_path results/generate/2017-11-15T13-46-47_speaker228_001_noGlobalCOndition.wav logdir/train/2017-11-15T13-46-47/model.ckpt-88
    'with no condition on OLLO'
    >>python3 generate.py --num_samples 32000  --wav_out_path results/generate/epoch125000_OLLO_NOconditioned.wav logdir/train/2017-12-18T11-31-30_OLLO_125000/model.ckpt-27500 --gc_channels=32 --gc_id=404 --gc_cardinality=411
    'with global condition'
    >>python3 generate.py --num_samples 32000  --wav_out_path results/generate/epoch125000_NOconditioned.wav logdir/train/2017-12-18T11-31-30_OLLO_125000/model.ckpt-27500 --gc_id=311 --gc_cardinality=377 --gc_channels=32 --wav_seed=rest_corpus/p226/p226_003.wav
    'with global condition on OLLO'
    >>python3 generate.py --num_samples 32000  --wav_out_path results/generate/epoch125000_OLLO_NOconditioned.wav logdir/train/2017-12-18T11-31-30_OLLO_125000/model.ckpt-27500 --gc_id=311 --gc_cardinality=377 --gc_channels=32 --wav_seed=rest_corpus/p226/p226_003.wav
Train:
    >>tensorboard --logdir=logdir/generate/2017-12-08T16-39-18
    'ON rest_corpus'
    >>python3 train.py --data_dir=rest_corpus --gc_channels=32 --restore_from logdir/train/2017-12-04T13-48-11_12000   #only give the dir
    'ON OLLO'
    >>python3 train.py --data_dir= --gc_channels=32 --restore_from logdir/train/2017-12-04T13-48-11_12000   #only give the dir

    >>srun -p x-men python3 train.py --gc_channels=32 --data_dir=rest_corpus --restore_from logdir/train/2017-11-21T14-12-14_42500


rename file "replace space in the file name with _"
>>for file in *; do mv "$file" ${file// /_}; done

run with cluster:
    srun -p x-men --mem 10GB -c 10 python hello.py


2017.11.29
clean git:
    everyday:
        while not going home:
            1. go to your master branch and pull from origin >>git pull
            2. work on an issue A, open a branch "issueA"  >>git checkout -b issueA
            3. constantly go back to your master and keep your master uptodate   >>git checkout master
            4. go back to your branch "issueA", >>git rebase

        >>git status
        "nothing is new!!"

prediction error:
    learn to make the movements so that it will minimize the error between
predict the change of bases function given the movement


20017.11.30
### Git Notes from Alex
# Interactive file adding
https://alblue.bandlem.com/2011/10/git-tip-of-week-interactive-adding.html

# Undo commit
https://stackoverflow.com/questions/927358/undo-the-last-git-commit

# Amend your last commit
$ git add .
$ git commit --amend

# List all files currently being tracked under the branch master
$ git ls-tree -r master --name-only

# List all branches of remote
$ git ls-remote <remote>
$ git remote show <remote>

# List all branches with last modification date being tracked locally
$ for k in `git branch -r | perl -pe 's/^..(.*?)( ->.*)?$/\1/'`; do echo -e `git
show --pretty=format:"%Cgreen%ci %Cblue%cr%Creset" $k -- | head -n 1`\\t$k; done |
sort -r

# Push a new local branch to a remote Git repository and track it too
$ git checkout -b <feature_branch_name>
... edit files, add and commit ...
$ git push -u origin <feature_branch_name>

# Push a local Git branch to master branch in the remote
$ git push <remote> <local_branch_name>:<remote_branch_to_push_into>
$ git push origin develop:master

# Delete a Git branch on remote
$ git push origin --delete <branch_name>

# Delete a Git branch locally
$ git branch -D <branch_name>

# Ignore files in Git
https://git-scm.com/docs/gitignore

# Gitignore is not working
https://stackoverflow.com/questions/11451535/gitignore-is-not-working

## More Advanced commands, handle with care
# Make a patch
https://ariejan.net/2009/10/26/how-to-create-and-apply-a-patch-with-git/

# Rewrite Git history with rebase
http://git-scm.com/book/en/Git-Tools-Rewriting-History


2017.12.13

def reconstruct_audio_from_spec( data):
    '''
    Param:
    data: array-like data, wav data'''

    dt = 0.1  #  define a time increment (seconds per sample)
    N = len(data)

    Nyquist = 1/(2*dt)  #  Define Nyquist frequency
    df = 1 / (N*dt)  #  Define the frequency increment

    G = np.fft.fftshift(np.fft.fft(data))  #  Convert "data" into frequency domain and shift to frequency range below
    f = np.arange(-Nyquist, Nyquist-df, df) #  define frequency range for "G"

    if len(G) != len(f):
        length = min(len(G), len(f))
    G_new = G[:length]*(1j*2*np.pi*f[:length])

    data_rec = np.abs(np.fft.ifft(np.fft.ifftshift(G_new)))

    plt.figure()
    plt.plot(data, 'b-', label='original')
    plt.hold(True)
    plt.plot(data_rec, 'm-', label='reconstruction', alpha=0.6)
    plt.legend(loc="best")
    plt.xlabel("frames")

2017.12.22
-split audio in SORN_WithFSD_LU
-Audio reconstruction on phase:  The most often-used phase reconstruction technique comes from  Griffin and Lim [1984] in ibrosa library
-DCGAN and spectrogram: http://deepsound.io/dcgan_spectrograms.html


2018.1.08
make gifs
# Make the gifs -- vae repo
if FLAGS.latent_dim == 2:
os.system(
    'convert -delay 15 -loop 0 {0}/posterior_predictive_map_frame*png {0}/posterior_predictive.gif'
    .format(FLAGS.logdir))

2018.01.09
multiprocessing make parallel plots
"""
# define what to do with each data pair ( p=[3,5] ), example: calculate product
def myfunc(p):
    #product_of_list = np.prod(p)

    xx = p[0]
    yy =  p[1]
    plt.plot(1, 2)
    plt.savefig("{}.png".format(yy))

def multi():
    xx = [2,1,8,9]
    yy = [3,4,5,7]
    data_pairs = map(lambda x,y: [x, y], xx, yy)

    pool = Pool()
    print data_pairs
    pool.map(myfunc, data_pairs)

multi()

"""

2018.01.12   spectrogram to audio_dir

# whole process: load audio--get spectrogram--scale to pixels--
#                from pixel values--scale up to (0, -4) spectrogram--recover audio
# functions are in usefull.../utils/spectrogramer.py
rate, data = wav.read('OLLO2.wav')
IPython.display.Audio(data=data, rate=rate)
wav_spectrogram = pretty_spectrogram(data.astype('float64'), fft_size = fft_size,
                                   step_size = step_size, log = True, thresh = spec_thresh)

# t, f, ss = signal.
scale_spec = scale_data(np.transpose(wav_spectrogram))   # scale up to image data
scale_spec = np.round(scale_spec)

crop_spec = crop_center(scale_spec,128,0)    # cut middle 128*128

BACK = scale_data(np.transpose(scale_spec), new_max=0, new_min=-4)   # scale back to wav spectrogram


recovered_audio_orig = invert_pretty_spectrogram(BACK, fft_size = fft_size,
                                            step_size = step_size, log = True, n_iter = 10)
IPython.display.Audio(data=recovered_audio_orig, rate=rate) # play the audio
# scale_spec, wav_spectrogram, BACK

inverted_spectrogram = pretty_spectrogram(recovered_audio_orig.astype('float64'), fft_size = fft_size,
                                   step_size = step_size, log = True, thresh = spec_thresh)
fig, ax = plt.subplots(nrows=1,ncols=1)
ax.matshow(scale_spec, interpolation='nearest', aspect='auto', cmap="gray_r", origin='lower')

2018.01.18
# tensorflow mask out zero values
import numpy as np
import tensorflow as tf
input = np.array([[1,0,3,5,0,8,6]])
X = tf.placeholder(tf.int32,[None,7])
zeros = tf.cast(tf.zeros_like(X),dtype=tf.bool)
ones = tf.cast(tf.ones_like(X),dtype=tf.bool)
loc = tf.where(input!=0,ones,zeros)
result=tf.boolean_mask(input,loc)
with tf.Session() as sess:
 out = sess.run([result],feed_dict={X:input})
 print (np.array(out))


2018.01.19
subplot axis is not visible but can use ylabel
fig = plt.figure(frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)tensorflow restore
plt.title("score on generated images")
#ipdb.set_trace()
for j in range(16):
    ax1 = fig.add_subplot(4, 4, j+1)
    #ax1.set_axis_off()
    #fig.add_axes(ax1)
    im = imgtest[j, :, :, 0]
    plt.imshow(im.reshape([128, 128]), cmap='Greys')
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    plt.ylabel("score={}".format(np.int(d_result[j]*10000)/10000.))
plt.subplots_adjust(left=0.07, bottom=0.02, right=0.93, top=0.98,
    wspace=0.02, hspace=0.02)

2018.301.20
screen:
    screen -r: show the list of all the screen running
    ctrl A +D: detach the screen, run the program in the background


2018.01.25
tensorflow data input pipline
enqueue files
problem: tf.errors.OutOfRangeError exception
        want to reuse dataqueue in multi-epochs
1. # https://github.com/tensorflow/tensorflow/issues/2514
    '''
    Multi-epoch use of queues might be simplified by adding one of the following:

        A queue.reset(), that throws one tf.errors.OutOfRangeError on dequeue() or some other exception.
        A queue.close(reset=True), that only throws one tf.errors.OutOfRangeError on dequeue() or some other exception.

    example usage of 1):

    q = tf.FIFOQueue(...)
    placeholder = ...
    enqueue_op = q.enqueue(placeholder)
    ....

    def producer(data_dir, sess, q, enqueue_op, placeholder):
      for ...:
        sess.run(enqueue_op, {placeholder:...})
      sess.run(q.reset())

    def do_epoch(data_dir, learn):
      threading.Thread(target=producer, args=(data_dir, sess, q, enqueue_op, placeholder)).start()
      while True:
        try:
          sess.run(...)
        exception tf.errors.OutOfRangeError:
          break

    for epoch in range(NUM_EPOCHS):
      ... = do_epoch(TRAIN_DIR, learn=True)
      ... = do_epoch(TEST_DIR, learn=False)
'''
2. https://www.tensorflow.org/versions/r1.3/programmers_guide/datasets
3. https://stackoverflow.com/questions/44132579/feed-data-into-a-tf-contrib-data-dataset-like-a-queue/45928467#45928467
    The new Dataset.from_generator() method allows you to define a Dataset that is fed by a Python generator.

4. https://stackoverflow.com/questions/33849617/how-do-i-convert-a-directory-of-jpeg-images-to-tfrecords-file-in-tensorflow

2018.01.31
'''example-of-tensorflows-new-input-pipeline.html'''
https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html

reading from files:
https://www.tensorflow.org/api_guides/python/reading_data
A typical pipeline for reading records from files has the following stages:
1. The list of filenames
2. Optional filename shuffling
3. Optional epoch limit
4. Filename queue
5. A Reader for the file format
6. A decoder for a record read by the reader
7. Optional preprocessing
8. Example queue


2018.3013
tomato timer for linux
sleep 1500 && zenity --warning --text="25 passed. Take a break!"; sleep 300 && zenity --warning --text="Get back to work!"
sleep 1500 && zenity --warning --text="25 passed. Take a break!"; sleep 300 && zenity --warning --text="Get back to work!"
sleep 1500 && zenity --warning --text="25 passed. Take a break!"; sleep 300 && zenity --warning --text="Get back to work!"
sleep 1500 && zenity --warning --text="25 passed. Take a break!"; sleep 900 && zenity --warning --text="Get back to work!"


countdown in linum terminal
countdown=1500 now=$(date +%s) watch -tpn1 echo '$((now-$(date +%s)+countdown))'


2018.03.26
1. creat MNE data format
http://martinos.org/mne/stable/auto_tutorials/plot_creating_data_structures.html#creating-raw-objects

2. Creating MNE objects from data arrays
http://martinos.org/mne/stable/auto_examples/io/plot_objects_from_arrays.html#sphx-glr-auto-examples-io-plot-objects-from-arrays-py

2018.03.27
1. rename all files in Ubuntu
    rename 's/\.txt$/.csv/' *
    >> rename all files to .csv
2. recursively download files with a web_link
    wget -r -np -R "index.html*" your_link
3. convert an array to one-hot vector
    np.eye(n_classes)[values]
4. rename part of the file parttern:
    for i in *.csv; do mv $i $(echo $i | sed 's/:/_/g'); done   ## repplace the : with _ in all filenames"
2018.03.29
1. downsampling data
    from scipy.signal import decimate
    ds_data =  decimate(data, ds_factor)
2. in Geany, how to find a functiosn defined somewhere
    cntr + left-click

2018.04.10
plot smooth data    Epilepsy/function.py
plot fill_data   Epilepsy/function.py
colors I like:
    green:
    blue:
    red:
    purple:

2018.04.16
save array as csv

2018.04.18
tesnsorflow understand lstm:
    https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
how  to self define func in tensorflow dataset:
    https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html
    https://www.tensorflow.org/api_docs/python/tf/data/Dataset
DeepConvLSTM on sensory recording:
    https://github.com/sussexwearlab/DeepConvLSTM/blob/master/DeepConvLSTM.ipynb
Error:
    I think because I take too many samples, run OOM
    {x: mnist.test.images, y: mnist.test.labels}

visualize VAE prior, very good notebook:
    Tutorial: https://github.com/hsaghir/VAE_intuitions/blob/master/VAE_MNIST_keras.ipynb
    Original blog: https://hsaghir.github.io/data_science/denoising-vs-variational-autoencoder/

ConvVAE:
    https://www.kaggle.com/rvislaywade/visualizing-mnist-using-a-variational-autoencoder

Atrous convolution:
    http://liangchiehchen.com/projects/DeepLab.html

Wavenet from paper to code:
    https://www.youtube.com/watch?v=LPMwJ-67SpE

Atrous_conv:
    out_conv = []
    # dilation rate lets us use ngrams and skip grams to process
    for dilation_rate in range(max_dilation_rate):
        x = prefilt_x
        for i in range(3):
            x = Conv1D(32*2**(i),
                       kernel_size = 3,
                       dilation_rate = dilation_rate+1)(x)
        out_conv += [Dropout(0.5)(GlobalMaxPool1D()(x))]
    x = concatenate(out_conv, axis = -1)

2018.4.23
tensorflow profile tracing:
    with tf.Session() as sess:

        #profiler = tf.profiler.Profiler(sess.graph)
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        _, acc, c, summary = sess.run([optimizer, accuracy, cost, summaries], feed_dict={x: batch_data, y: batch_labels}, options=options, run_metadata=run_metadata)
        ####### # Create the Timeline object, and write it to a json file
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open(save_name + 'timeline_{}.json'.format(batch), 'w') as f:
            f.write(chrome_trace)
    then go to chrome://tracing load .json

how to install google chrome without sudo:
    http://indiayouthtechtips.blogspot.de/2012/03/how-to-install-google-chrome-without.html

2018.04.27
Dilated CNN:
    https://sthalles.github.io/deep_segmentation_network/
    @slim.add_arg_scope
    def atrous_spatial_pyramid_pooling(net, scope, depth=256):
        """
        ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
        (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
        :param net: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
        :param scope: scope name of the aspp layer
        :return: network layer with aspp applyed to it.
        """
        with tf.variable_scope(scope):
            feature_map_size = tf.shape(net)

            # apply global average pooling
            image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keep_dims=True)
            image_level_features = slim.conv2d(image_level_features, depth, [1, 1], scope="image_level_conv_1x1", activation_fn=None)
            image_level_features = tf.image.resize_bilinear(image_level_features, (feature_map_size[1], feature_map_size[2]))

            at_pool1x1 = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_0", activation_fn=None)

            at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1", rate=6, activation_fn=None)

            at_pool3x3_2 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_2", rate=12, activation_fn=None)

            at_pool3x3_3 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_3", rate=18, activation_fn=None)

            net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3,
                            name="concat")
            net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)
            return net

tensorflow input .csv:
    https://stackoverflow.com/questions/43621637/tensorflow-input-pipeline-error-while-loading-a-csv-file
    https://www.tensorflow.org/get_started/datasets_quickstart
    format reading .csv per line: https://stackoverflow.com/questions/37091899/how-to-actually-read-csv-data-in-tensorflow
    QueueRunner with .csv files: https://www.tensorflow.org/api_guides/python/reading_data:
        def read_my_file_format(filename_queue):
            reader = tf.SomeReader()
            key, record_string = reader.read(filename_queue)
            example, label = tf.some_decoder(record_string)
            processed_example = some_processing(example)
            return processed_example, label

        def input_pipeline(filenames, batch_size, num_epochs=None):
            filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=True)
            example, label = read_my_file_format(filename_queue)
            # min_after_dequeue defines how big a buffer we will randomly sample
            #   from -- bigger means better shuffling but slower start up and more
            #   memory used.
            # capacity must be larger than min_after_dequeue and the amount larger
            #   determines the maximum we will prefetch.  Recommendation:
            #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
            min_after_dequeue = 10000
            capacity = min_after_dequeue + 3 * batch_size
            example_batch, label_batch = tf.train.shuffle_batch(
            [example, label], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
            return example_batch, label_batch
        # Create the graph, etc.
        init_op = tf.global_variables_initializer()

        # Create a session for running operations in the Graph.
        sess = tf.Session()
        # Initialize the variables (like the epoch counter).
        sess.run(init_op)
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                # Run training steps or whatever
                sess.run(train_op)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()

2018.05.02
tensorflow input pipeline:
    1. one .csv
    2. GOOD!! read multiple .csv files each a sone training sample
        https://stackoverflow.com/questions/49525056/tensorflow-python-reading-2-files/49548224#49548224
    3: GOOD!!! works one row by row training sample:
        https://stackoverflow.com/questions/49525056/tensorflow-python-reading-2-files/49548224#49548224
    4. Good!:
        https://learningtensorflow.com/ReadingFilesBasic/
    5. :
        https://stackoverflow.com/questions/49899526/tensorflow-input-pipeline-where-multiple-rows-correspond-to-a-single-observation
    6. : https://www.programcreek.com/python/example/90498/tensorflow.TextLineReader
        def read_csv(batch_size, file_name):
            filename_queue = tf.train.string_input_producer([file_name])
            reader = tf.TextLineReader(skip_header_lines=0)
            key, value = reader.read(filename_queue)
            # decode_csv will convert a Tensor from type string (the text line) in
            # a tuple of tensor columns with the specified defaults, which also
            # sets the data type for each column
            decoded = tf.decode_csv(
                value,
                field_delim=' ',
                record_defaults=[[0] for i in range(FLAGS.max_sentence_len * 2)])

            # batch actually reads the file and loads "batch_size" rows in a single tensor
            return tf.train.shuffle_batch(decoded,
                                          batch_size=batch_size,
                                          capacity=batch_size * 50,
                                          min_after_dequeue=batch_size)
    7.: high level performance: https://www.tensorflow.org/performance/datasets_performance
        ### load data
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename).skip(0).map(decode_csv))

        dataset = dataset.batch(total_rows).shuffle(buffer_size=1000).repeat()   ###repeat().

        iterator = dataset.make_initializable_iterator()
        batch_data = iterator.get_next()
    8. data pipeline tutorial GOOD: https://cs230-stanford.github.io/tensorflow-input-data.html

    7. Yay! works:
        def decode_csv(line):
            defaults = [[0.0]]*512
            csv_row = tf.decode_csv(line, record_defaults=defaults)#
            data = tf.stack(csv_row)
            return data
        filenames = ['data/test/2014-10-06T21:28:54.csv', 'data/test/2014-11-03T12:01:09.csv', 'data/test/BL-2014-09-26T00:25:34.csv']
        dataset5 = tf.data.Dataset.from_tensor_slices(filenames)
        dataset5 = dataset5.flat_map(lambda filename: tf.data.TextLineDataset(filename).skip(0).map(decode_csv))

        dataset5 = dataset5.batch(4).shuffle(buffer_size=1000).repeat(20)   ###repeat().

        iterator5 = dataset5.make_initializable_iterator()
        next_element5 = iterator5.get_next()

        t1 = time.time()
        with tf.Session() as sess:
            # Train 2 epochs. Then validate train set. Then validate dev set.
            sess.run(iterator5.initializer)
            for _ in range(10):
                features = sess.run(next_element5)
                      # Train...
                #print("shape:", features.shape)
                print("label", features.shape, 'time', time.time()-t1)

            # Validate (cost, accuracy) on train set
            ipdb.set_trace()
            print("\nDone with the first iterator\n")


2018.05.07
Very nice tutorial on :
    https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607

2018.5.17
visualize conv filters:
    https://stackoverflow.com/questions/33783672/how-can-i-visualize-the-weightsvariables-in-cnn-in-tensorflow/33794463#33794463
    https://github.com/grishasergei/conviz
    https://medium.com/@awjuliani/visualizing-neural-network-layer-activation-tensorflow-tutorial-d45f8bf7bbc4
correct sliding window function:
    def sliding_window(data_x, data_y, num_seg=5, window=128, stride=64):
        '''
        Param:
            datax: array-like data shape (batch_size, seq_len, channel)
            data_y: shape (num_seq, num_classes)
            num_seg: number of segments you want from one seqeunce
            window: int, number of frames to stack together to predict future
            noverlap: int, how many frames overlap with last window
        Return:
            expand_x : shape(batch_size, num_segment, window, channel)
            expand_y : shape(num_seq, num_segment, num_classes)
            '''
        assert len(data_x.shape) == 3
        expand_data = []
        for ii in range(data_x.shape[0]):
            num_seg = (data_x.shape[1] - window) // stride + 1
            shape = (num_seg, window, data_x.shape[-1])      ## done change the num_seq
            strides = (data_x.itemsize*stride*data_x.shape[-1], data_x.itemsize*data_x.shape[-1], data_x.itemsize)
            expand_x = np.lib.stride_tricks.as_strided(data_x[ii, :, :], shape=shape, strides=strides)
            expand_data.append(expand_x)
        expand_y = np.repeat(data_y,  num_seg, axis=0).reshape(data_y.shape[0], num_seg, data_y.shape[1]).reshape(-1, data_y.shape[1])
        return np.array(expand_data).reshape(-1, window, data_x.shape[-1]), expand_y

2018.06.05
time series train and test split cross validation
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

2018.06.06
access variables with name:
    with tf.variable_scope('fc_0/fully_connected', reuse=True):
        ww = tf.get_variable('weights')
2. get all variables associated with a layer,
    with tf.variable_scope('fc_0/fully_connected', reuse=True):
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc_0/fully_connected')
3. Pandas:
    https://jeffdelaney.me/blog/useful-snippets-in-pandas/

https://github.com/tensorlayer/tensorlayer/issues/146

2018.06.19
t-SNE tutorial:
    https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

2018.06.26
sort directory by space usage:
    >>cd elu/LU/
    >>du --max-depth=1
    elu@digda ~/LU$ >>du --max-depth=1
        3879752 ./1_Infant_speech_acquisition
        70724   ./documents of all time
        599712  ./software
        354048  ./Books
        36136   ./5_Goal_Robot
        76544   ./Plotting_data_tips
        25382500    ./3_speech_recognition
        347148  ./learning
        8132    ./courses
        79884952    ./2_Neural_Network
        75396   ./4_discussion_meeting_paper_other_reading_
        2232    ./4_paper_other_reading_discussion_meeting
        110729868   .

2018.06.27
plot without frame
for cluster in range(num_clusters):
        ind_cluster = np.where(km.labels_ == cluster)[0]
        fig, axes = plt.subplots(10, 10,
                     subplot_kw={'xticks': [], 'yticks': []})   ## figsize=(12, 6),

        #fig.subplots_adjust(hspace=0.3, wspace=0.05)
        for ind in range(np.min(ind_cluster.size), 100):
            for ax in axes.flat:
                ax.plot(data[ind_cluster[ind]])

2018.06.29
np.histgram(aa, bins='auto'))
plt.hist(aa, bins)

2018.07.02
check swap space usage: https://www.tecmint.com/commands-to-monitor-swap-space-usage-in-linux/

2018.07.02
load data repeat:  generator
def EEG_data(data_dir, pattern='Data*.csv', withlabel=False, num_samples=784, batch_size=128):

    files = find_files(data_dir, pattern='Data*.csv', withlabel=False)

    datas = np.zeros((len(files), 10240*2))

    for ind, filename in enumerate(files):
        data = read_data(filename, header=None, ifnorm=True)
        datas[ind, :] = np.append(data[:, 0], data[:, 1])  ## only use one channel

    useful_length = (datas.size // input_dim) * input_dim
    datas = datas.flatten()[0: useful_length].reshape(-1, input_dim)  ##
    while True:
        try:

            start = np.random.randint(len(datas) - batch_size)
            batch_x = datas[start: start+batch_size]

            yield np.asarray(batch_x)

        except Exception as e:
            print('Could not produce batch of sinusoids because: {}'.format(e))
            sys.exit(1)

2018.07.05
add parent path in import
import sys
sys.path.insert(0, os.path.abspath('..'))

2018.07.16
Valentin data in FIAS local storage:
    elu@digda /home>> cd epilepsy-data
read HDF5 file:
    https://confluence.slac.stanford.edu/display/PSDM/How+to+access+HDF5+data+from+Python
import h5py
import matplotlib.pyplot as plt
from scipy.signal import detrend
hf = h5py.File('EpimiRNA_1.2-27_recordings.h5', 'r')
key = hf.keys()   ##[u'data', u'meta']
data = hf.get(hf.keys()[0])   ##HDF5 group "/data" (966 members)
data.items()  ## 966 (u'2014-11-03T15:01:08', <HDF5 group "/data/2014-11-03T15:01:08" (4 members)>)]
name = data.items()[ind][0]  # '2014-11-03T15:01:08'
values = data.items()[ind][1]  # values = data['2014-11-03T15:01:08']. <HDF5 group "/data/2014-11-03T15:01:08" (4 members)>
group_names = values.items()### 'NDF_File_Name', 'NDF_local_timestamp', 'time', 'voltage'
voltage = values.items()[3]  #(u'voltage', <HDF5 dataset "voltage": shape (1843200,), type "<f8">)
voltage = values.items()[3][1]
dev1 = detrend(voltage)
plot:
>>> name = '2014-10-01T06:54:15'
>>> values = data[name]
>>> vol = values.items()[3][1]
>>> dev = detrend(vol)
>>> plt.subplot(211), plt.title('Signal and spectrogram of {}'.format(name), fontsize=22), plt.plot(np.arange(dev.size)/ 512.0, dev, 'purple'), plt.xlabel('time / s', fontsize=20), plt.ylabel('amplitude', fontsize=20), plt.xlim([0, dev.size/512.0]), plt.subplot(212), plt.specgram(dev, detrend='linear', cmap='viridis', NFFT=5120, Fs=512, noverlap=3072, scale_by_freq=True, vmin=-2), plt.xlim([0, dev.size/ 512.0]), plt.ylim([0, 100]), plt.xlabel('time / s', fontsize=20), plt.ylabel('frequency', fontsize=20), plt.show()

>>> plt.subplot(211), plt.title('Signal and spectrogram of {}'.format(name), fontsize=22), plt.plot( np.arange(dev.size)/ 512.0, dev, 'purple'), plt.xlabel('time / s', fontsize=20), plt.ylabel('amplitude', fontsize=20), plt.xlim([0, dev.size/512.0]), plt.subplot(212), plt.specgram(dev, detrend='linear', cmap='YlGnBu', NFFT=5120, Fs=512, noverlap=3072, scale_by_freq=True, vmin=-2), plt.xlim([0, dev.size/ 512.0]), plt.ylim([0, 100]), plt.xlabel('time / s', fontsize=20), plt.ylabel('frequency', fontsize=20), plt.yticks([2.0, 7.0, 12, 20, 40, 60, 80, 100, 150]), plt.show()

### get plots for multiple data recordings
names = data.items()[61:86]
values, vols, devs, ns = [], [], [], []
for i in range(len(names)): values.append(names[i][1])
for i in range(len(names)): vols.append(values[i].items()[3][1])
for i in range(len(names)): devs.append(detrend(vols[i]))
for i in range(len(names)): ns.append(names[i][0])
for name, dev in zip(ns[1:], devs[1:]): plt.figure(figsize=(16, 11)), plt.subplot(211), plt.title('Signal and spectrogram of {}'.format(name), fontsize=22), plt.plot(np.arange(dev.size)/ 512.0, dev, 'purple'), plt.xlabel('time / s', fontsize=20), plt.ylabel('amplitude', fontsize=20), plt.xlim([0, dev.size/512.0]), plt.subplot(212), plt.specgram(dev, detrend='linear', cmap='YlGnBu', NFFT=3072, Fs=512, noverlap=128, scale_by_freq=True, vmin=-2), plt.xlim([0, dev.size/ 512.0]), plt.ylim([0, 150]), plt.xlabel('time / s', fontsize=20), plt.ylabel('frequency', fontsize=20), plt.yticks([2, 7, 12, 20, 40, 60, 80, 100, 150]), plt.savefig('/home/elu/Desktop/1243/'+'signal{}.png'.format(name), format='png'), plt.close()

## save data
name = data.items()[61]
value = dd[1].items()[3][1]
np.savetxt('/home/elu/Desktop/1227/data'+'BL-{}.csv'.format(name), dev.reshape(-1, 512), delimiter=',', fmt="%10.5f", comments='')

for i in range(60): np.savetxt('/home/elu/Desktop/1227-MFE-1103/'+'signal{}_min{}.csv'.format(name[0], i), np.array(dev[i*30720:(i+1)*30720]), header=name[0], delimiter=',', fmt="%10.5f", comments='')


2018.07.17
How to visualize the CNN kernels?
    2. Google search items:
        RIght: https://www.google.com/search?newwindow=1&client=ubuntu&channel=fs&ei=QuZNW7rJJMm4sQHa5rt4&q=tensorboard+summary+weights&oq=tensorboard+&gs_l=psy-ab.1.0.35i39k1l2j0l2j0i20i263k1j0l5.123896.134636.0.137860.36.24.10.0.0.0.184.1958.12j7.20.0....0...1.1.64.psy-ab..6.30.2118.6..0i67k1.86.QJAWfPyEgXg
    3. Very good tensorboard tutorial with tensorboard example and code:
        https://jhui.github.io/2017/03/12/TensorBoard-visualize-your-learning/
    4. very good example with stacked grayscale filters:
        https://stackoverflow.com/questions/33802336/visualizing-output-of-convolutional-layer-in-tensorflow
    1. function put_kernels_on_grid:
        https://gist.github.com/kukuruza/03731dc494603ceab0c5
        def put_kernels_on_grid (kernel, grid_Y, grid_X, pad = 1):

            '''Visualize conv. features as an image (mostly for the 1st layer).
            Place kernel into a grid, with some paddings between adjacent filters.

            Args:
              kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
              (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                                   User is responsible of how to break into two multiples.
              pad:               number of black pixels around each filter (between them)

            Return:
              Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
            '''

            x_min = tf.reduce_min(kernel)
            x_max = tf.reduce_max(kernel)

            kernel1 = (kernel - x_min) / (x_max - x_min)

            # pad X and Y
            x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

            # X and Y dimensions, w.r.t. padding
            Y = kernel1.get_shape()[0] + 2 * pad
            X = kernel1.get_shape()[1] + 2 * pad

            channels = kernel1.get_shape()[2]

            # put NumKernels to the 1st dimension
            x2 = tf.transpose(x1, (3, 0, 1, 2))
            # organize grid on Y axis
            x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels])) #3

            # switch X and Y axes
            x4 = tf.transpose(x3, (0, 2, 1, 3))
            # organize grid on X axis
            x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels])) #3

            # back to normal order (not combining with the next step for clarity)
            x6 = tf.transpose(x5, (2, 1, 3, 0))

            # to tf.image_summary order [batch_size, height, width, channels],
            #   where in this case batch_size == 1
            x7 = tf.transpose(x6, (3, 0, 1, 2))

            # scale to [0, 255] and convert to uint8
            return tf.image.convert_image_dtype(x7, dtype = tf.uint8)

Good!!!
with tf.name_scope(‘fc1’):
    layer1 = tf.layers.dense(features, 512, activation=tf.nn.relu, name=’fc1′)
    fc1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ‘fc1’)
    tf.summary.histogram(‘kernel’, fc1_vars[0])
    tf.summary.histogram(‘bias’, fc1_vars[1])
    tf.summary.histogram(‘act’, layer1)

with tf.variable_scope('conv1') as scope:
    conv = tf.layers.con2d()

    kernel, bias = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name)
    grid = func.put_kernels_on_grid (kernel, pad = 2)
    tf.image.summary(scope.name, grid, max_outputs=1)

2018.07.18
Profile script:
    https://docs.python.org/2/library/profile.html

steps:
    1. python -m cProfile [-o output_file] [-s sort_order] myscript.py
    2. python
    3. import pstats
        p = pstats.Stats('tf_profile')
        p.sort_stats('tottime').print_stats(20)

2018.07.18
Replace dataset iterator with tfrecords and queue
### tensorflow dataset
    dataset_train = tf.data.Dataset.from_tensor_slices((files_train, labels_train)).repeat().batch(batch_size).shuffle(buffer_size=10000)
    iter = dataset_train.make_initializable_iterator()
    ele = iter.get_next()   #you get the filename

2018.07.23
compute area_under_curve:
    1.from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels_train_hot, outputs)
    2. area_under_curve = tf.contrib.metrics.streaming_auc(labels=y, predictions=outputs, name='auc')

2018.07.25
autocorrelation:
    err = datas[ind, :, 0] - np.mean(datas[ind, :, 0])
    variance = np.sum(err ** 2) / datas[ind, :, 0].size
    correlated = np.correlate(err, err, mode='full')/variance
    correlated = correlated[correlated.size//2:]

nice CNN tutorial:
    setosa.io/ev/image-kernels

2018.07.26
how to interpret learned weights:
    https://stackoverflow.com/questions/47745313/how-to-interpret-weight-distributions-of-neural-net-layers
weights initialization:
    https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94

2018.07.30
add subplots to subplots:
    https://www.python-kurs.eu/matplotlib_unterdiagramme.php


2018.08.27
plt sparse matrix:
    import scipy.sparse as sps
    import matplotlib.pyplot as plt
    a = sps.rand(1000, 1000, density=0.001, format='csr')
    plt.spy(a)
    plt.show()

plt smooth histogram (density):
    import pandas as pd
    pd.DataFrame(data).plot(kind='density'))



2018.08.30
Valentin data in FIAS local storage:
    elu@digda /home>> cd epilepsy-data
read HDF5 file:
    https://confluence.slac.stanford.edu/display/PSDM/How+to+access+HDF5+data+from+Python
import h5py
hf = h5py.File('EpimiRNA_1.2-27_recordings.h5', 'r')
key = hf.keys()
data = hf.get(key[0])
data.items()  ## (u'2014-11-03T15:01:08', <HDF5 group "/data/2014-11-03T15:01:08" (4 members)>)]
name = data.items()[0][0]  # '2014-09-22T11:24:31'~'2014-11-03T15:01:08'
values = data.items()[0][1]  # <HDF5 group "/data/2014-11-03T15:01:08" (4 members)>
group_names = values.items()### 'NDF_File_Name', 'NDF_local_timestamp', 'time', 'voltage'
voltage = values.items()[3]  #(u'voltage', <HDF5 dataset "voltage": shape (1843200,), type "<f8">)
voltage = values.items()[3][1]

from scipy import detrend
detr_v = detrend(voltage)
zvolt = scipy.stats.zscore(volt)
fr, psd = scipy.signal.welch(zvolt)
plt.semilogx(fr, psd)

plt.subplot(211), plt.plot(dev1, 'purple'), plt.subplot(212), plt.specgram(dev1, detrend='constant', NFFT=256, Fs=512, noverlap=128), plt.title('no trend in spectrogram; periodic high freq'), plt.colorbar()


2018.09.07
plot spectrogram:
    plt.subplot(211), plt.title('Signal and spectrogram of {}'.format(data.items()[ind][0]), fontsize=22), plt.plot(np.arange(dev1.size)/ 512.0, dev1, 'purple'), plt.xlabel('time / s', fontsize=20), plt.ylabel('amplitude', fontsize=20), plt.xlim([0, dev1.size/512.0]), plt.subplot(212), plt.specgram(dev1, detrend='linear', cmap='viridis', NFFT=3072, Fs=512, noverlap=128, scale_by_freq=True, vmin=vmin), plt.xlim([t[0], t[-1]]), plt.ylim([0, 100]), plt.xlabel('time / s', fontsize=20), plt.ylabel('frequency', fontsize=20), plt.show()

2018.09.17
position legend box:
    https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot


2018.10.17
tensorflow Textlinedataset read multiple lines as one training sample:
    https://stackoverflow.com/questions/49899526/tensorflow-input-pipeline-where-multiple-rows-correspond-to-a-single-observation
    def _parse_and_decode(filename, group_size):
        '''input would be (filename, label), decode the file in TextLineDataset and return decoded dataset
        decode csv file, get group_size seconds of data, give the label and return
        '''
        ## decode csv file, read group_size row as one sample
        ds = tf.data.TextLineDataset(filename)
        ds = ds.batch(group_size).skip(0).map(lambda line: decode_csv(line, group_size))

        return ds
    ## way 2
    filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]

    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    # Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
    # and then concatenate their contents sequentially into a single "flat" dataset.
    # * Skip the first line (header row).
    # * Filter out lines beginning with "#" (comments).
    dataset = dataset.flat_map(
        lambda filename: (
            tf.data.TextLineDataset(filename)
            .skip(1)
            .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))


tensorflow np.repeat() equivilant:
    labels = np.array([1, 2, 2, 0, 1, 2, 1, 2, 0, 0, 1, 0])
    aa = tf.tile(tf.reshape(labels, [-1, 1]), [1, 3])  ##repeat 3 times
    bb = tf.reshape(aa, [-1])
    repeatlabels = tf.reshape(bb)

2018.10.26
tensorflow datast train and test split:
    dataset.take()
    dataset.skip()
    https://stackoverflow.com/questions/47735896/get-length-of-a-dataset-in-tensorflow
GREAT dataset tutorial:
    https://cs230-stanford.github.io/tensorflow-input-data.html
    https://cs230-stanford.github.io/tensorflow-model.html

Order for dataset:
    To summarize, one good order for the different transformations is:
        create the dataset
        shuffle (with a big enough buffer size)
        repeat
        map with the actual work (preprocessing, augmentation…) using multiple parallel calls
        batch
        prefetch

2018.10.29
github contributor:
    https://github.com/CoolProp/CoolProp/wiki/Contributing%3A-git-development-workflow


2018.11.06
Pycharm shortcut:
    ctrl + B -- go to the declaration of a class, method or variable
    shift + F6 -- rename all places
    ctrl + Q -- see the documentation
    ctrl + shift + up/down  Code | Move Statement Up/Down action
    Ctrl+P brings up a list of valid parameters.
    Ctrl+Shift+Backspace (Navigate | Last Edit Location) brings you back to the last place where you made changes in the code.
    Ctrl+Shift+F7 (Edit | Find | Highlight Usages in File) to quickly highlight usages of some variable in the current file. Use F3 and Shift+F3 keys to navigate through highlighted usages. Press Escape to remove highlighting
    Ctrl+Space basic code completion ()
    Alt+Up and Alt+Down keys to quickly move between methods in the editor.

2018.11.09
get total size:
    total_bytes += os.path.getsize(os.path.join(root, f))
very good answer on map and flat_map:
    https://stackoverflow.com/questions/49116343/dataset-api-flat-map-method-producing-error-for-same-code-which-works-with-ma

2018.11.20
run charles classifier

2018.12.11
rename filenames in batch (Linux)):
    for f in $(find . -name '*.csv'); do mv $f ${f/:/_}; done;
    for f in $(find . -name '*'); do mv $f ${f/1270/32140}; done;


2018.12.17
'global_step': tf.train.get_global_step()

# Add summaries manually to writer at global_step
if writer is not None:
    global_step = results[-1]['global_step']
    for name, val in metrics_test.items():
        if 'matrix' not in name:
            summ = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=val[0])])
writer.add_summary(summ, global_step)

# Metrics for evaluation using tf.metrics (average over whole dataset)
# with tf.name_scope("metrics"):
#     # Streaming a confusion matrix for group with metrics together to update or init
#     batch_confusion = tf.confusion_matrix(labels_int, post_pred_int, num_classes=args.num_classes, name='confusion')
    # create an accumulator to hold the counts
    # confusion = tf.Variable(tf.zeros([args.num_classes, args.num_classes], dtype=tf.int32))
    # Create the update op for doing a "+=" accumulation on the batch
    # conf_update_op = confusion.assign(confusion + batch_confusion)
    # metrics = {
    #     'accuracy': tf.metrics.accuracy(labels=labels_int, predictions=post_pred_int),
    #     'loss': tf.metrics.mean(loss)
    #     # 'conf_matrix': (confusion, conf_update_op)
    # }
# TODO: there are two values in metrics["accuracy"], metrics["loss"], conf_matrix is not updated properly

# Group the update ops for the tf.metrics
# update_metrics_op = tf.group(conf_update_op, *[op for _, op in metrics.values()])
# metrics['conf_matrix'] = confusion

# Get the op to reset the local variables used in tf.metrics
# metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
# metrics_init_op = tf.variables_initializer(metric_variables)

2. def reduce_data_mean(ret):
    N = len(ret)
    mean_val = {}
    for key in ret[0].keys():
        if key != 'train_op':
            mean_val[key] = sum([b[key] for b in ret]) / N
    return mean_val

3. check GPU info on the clusters: scontrol show node name_node
4. def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    logging.info("Available GPU")
    logging.info([x.name for x in local_device_protos if x.device_type == 'GPU'])
    return

2018.12.20
map zscore normalization to dataset: better to noam it in the parse function:
    def parse_function(filename, label, args):
        """
        parse the file. It does following things:
        1. init a TextLineDataset to read line in the files
        2. decode each line and group args.secs_per_samp*args.num_segs rows together as one sample
        3. repeat the label for each long chunk
        4. return the transformed dataset
        :param filename: str, file name
        :param label: int, label of the file
        :param args: Param object, contains hyperparams
        :return: transformed dataset with the label of the file assigned to each batch of data from the file
        """
        skip = np.random.randint(0, args.secs_per_samp)
        decode_ds = tf.data.TextLineDataset(filename).skip(skip).map(decode_csv).batch(args.secs_per_samp*args.num_segs)
        decode_ds = decode_ds.map(scale_to_zscore)
        label = tf.tile(tf.reshape(label, [-1, 1]), [1, np.int((args.secs_per_file - skip) // (args.secs_per_samp * args.num_segs))])
        label = tf.reshape(label, [-1])  ## tensorflow np.repeat equivalent
        label_ds = tf.data.Dataset.from_tensor_slices(label)  ## make a label dataset

        transform_ds= tf.data.Dataset.zip((decode_ds, label_ds))

        return transform_ds
    2. if train on spectrogram, then we can use longer segments

2019.01.03
Run jobs on GPU:
    srun -p sleuths --mem=4000 --gres gpu:titanblack:1 python3 EPG_classification.py             # 100h 14 mins
List all devices:   # keep :1
    srun -p sleuths --gres gpu:titanblack:1 lspci
        
2019.01.17
pytorch load data:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
tensorflow examples:
    https://www.programcreek.com/python/example/90570/tensorflow.decode_csv

tensorflow load one .csv file as one training sample:
    def parse_function(filename, label, args=None):
        """
        parse the file. It does following things:
        1. init a TextLineDataset to read line in the files
        2. decode each line and group args.secs_per_samp*args.num_segs rows together as one sample
        3. repeat the label for each long chunk
        4. return the transformed dataset
        :param filename: str, file name
        :param label: int, label of the file
        :param args: Param object, contains hyperparams
        :return: transformed dataset with the label of the file assigned to each batch of data from the file
        """
        defaults = [[0.0]] * args.seq_len
        content = tf.read_file(filename)
        content = tf.decode_csv(content, record_defaults=default)
        decode_ds = scale_to_zscore(content)  # zscore norm the data
        if args.if_spectrum:
            decode_ds = get_spectrum(decode_ds, args=args)

        return decode_ds, label

2019.01.21
pytorch load data from multiple csv files:
    https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/dataset_loader.py
EEG inplementation cases:
    https://mediatum.ub.tum.de/doc/1422453/552605125571.pdf
EEGNet:
    paper: https://arxiv.org/pdf/1611.08024.pdf
    code: https://github.com/aliasvishnu/EEGNet/blob/master/EEGNet-PyTorch.ipynb

2019.03.11
Linux check disk space with df command

    Open the terminal and type the following command to check disk space.
    The basic syntax for df is:
    df [options] [devices]
    Type:
    df
    df -H:
        Filesystem                  Size  Used Avail Use% Mounted on
        udev                        4.0G     0  4.0G   0% /dev
        tmpfs                       804M   69M  735M   9% /run
        /dev/mapper/vg_system-root  3.2G  2.5G  432M  86% /
        /dev/vg_system/usr           34G   29G  3.3G  90% /usr
        tmpfs                       4.1G  141M  3.9G   4% /dev/shm
        tmpfs                       5.3M  4.1k  5.3M   1% /run/lock
        tmpfs                       4.1G     0  4.1G   0% /sys/fs/cgroup
        /dev/mapper/vg_system-var    11G  2.7G  7.3G  27% /var
        /dev/mapper/vg_system-tmp   353G   75M  335G   1% /tmp
        tmpfs                       804M  222k  804M   1% /run/user/3070
        undici:/home-elu            236G  197G   30G  87% /home/elu
        dodici:/home-epilepsy-data  4.4T  1.2T  3.1T  28% /home/epilepsy-data
        dodici:/home-software       216G  180G   27G  88% /home/software

re
https://www.cyberciti.biz/faq/linux-check-disk-space-command/:
    du -s
    elu@oak ~/LU/2_Neural_Network/2_NN_projects_codes$ du -h -s Epilepsy/
    108G    Epilepsy/
    du -a Epilepsy/ | sort -n -r | head -n 20
    du -a ./ | sort -n -r | head -n 20
    
    elu@oak ~/LU/2_Neural_Network$ du -a -H ./ | sort -n -r | head -n 30

2019.03.13
open .TRC files:
    download field_trip toolbox
    in matlab:
        addpath(abs path to the package)
        run: ft_defaults
Better one:
    1. Do NOT add fieldtrip to path!
    2. go to C:\Users\gk\Desktop\trc\fieldtrip\fileio\private
    3. load samples as follows:
    data1 = read_micromed_trc('C:\Users\gk\Desktop\trc\EEG_21.TRC', 0, 99);
    data2 = read_micromed_trc('C:\Users\gk\Desktop\trc\EEG_21.TRC', 100, 199);
    etc.
    4. you can merge them as follows if you have enough RAM:
    allData = [data1; data2];
    5. And you can plot them:
    plot(allData')
 
    If you run
    info = read_micromed_trc('C:\Users\gk\Desktop\trc\EEG_21.TRC')
    you will get some info about the file such as the recording date and
the number of samples.

2018.03.18
Keras with tf.dataset:
    https://stackoverflow.com/questions/46135499/how-to-properly-combine-tensorflows-dataset-api-and-keras
save weights in keras and load:
    https://machinelearningmastery.com/save-load-keras-deep-learning-models/

2018.03.19
amazing Keras working with generator. Machine learning framework_fin:
    https://www.pyimagesearch.com/2018/12/10/keras-save-and-load-your-deep-learning-models/

2019.03.25
eager_execution with dataset API:
    https://towardsdatascience.com/eager-execution-tensorflow-8042128ca7be

2019.04.18
change to the epilepsy-data folder:
    elu@oak /home$ cd epilepsy-data

2019.04.23
srun -p sleuths -w jetski --mem=6000 --reservation triesch-shared --gres gpu:rtx2080ti:1 --pty env PYTHONPATH=~software/tensorflow-py3-amd64-gpu python3
scontrol show nodes jetski

2019.04.25
Sorted list:
    sorted_percent = sorted(list(data), key=lambda x: x[1])
    np.savetxt((txts[0][0:-4] + '_sorted.txt'), np.array(sorted_percent), fmt="%s", delimiter=", ")
    # LOad file with str and float
    pd.read_csv(txts[0]).values

2019.04.26
Go to the root folder with GUI:
    Go to computer --> home --> there you are!


when the screen is frozen:
    ctrl + alt + F1:
        1:
            killall -9 python3.5
            ctrl + alt + F7
        2:
            htop
            select: 9: killsignal

2019.05.09
try to pass the number of rows as an element of the dataset and do repeat labels in parsing function, but the tf.tile needs a real value
tensorflow dataset:
    https://zhuanlan.zhihu.com/p/43356309
    
2019.05.16
export edf files to csv:
    1. edf2csv.py, export each hour of recordings given the corresponding machine and channel name and dates
    2. matlab filter the outliers
    3. exclude all the artifacts and resave the file with filename, label and 5 sec-each line

2019.05.20
https://github.com/InFoCusp/tf_cnnvis/blob/master/examples/tf_cnnvis_Example1.ipynb
visualize kernels:
    get all the kernels:
        kernels = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('kernel:0')]
    2. great activation maximization visualization method:
        https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030

2019.05.27
Save multiple variables into txt:
    import pickle
    with open('test_indiv_hour_data_acc.txt', 'wb') as f:
        pickle.dump({"filenames": np.array(filenames),
                     "true_labels": np.array(int_labels),
                     "fns_w_freq": np.array(fns_w_freq),
                     "result_data": result_data}, f)

    file = open('dump.txt', 'r')
    dict = pickle.load(file)

2019.07.02
custom color map:
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","violet","blue"])

2019.08.01
how to download the certificate of enrolment:
    1. login:
        https://qis.server.uni-frankfurt.de/qisserver
        s3722145  LDYdhrz032122$
    2. my functions --> administration of studies --> Study reports for all terms
    
        https://qis.server.uni-frankfurt.de/qisserver/rds?state=change&type=1&moduleParameter=studySOSMenu&nextdir=change&next=menu.vm&subdir=applications&xml=menu&purge=y&navigationPosition=functions%2CstudySOSMenu&breadcrumb=studySOSMenu&topitem=functions&subitem=studySOSMenu

2019.10.07
go to the GUI of squeue:
    sview &

2019.11.26
amazing VAE code github:
    https://github.com/respeecher/vae_workshop

2020.08.30
1. # change the terminal tab title
PROMPT_COMMAND='echo -ne "\033]0; bash-speedboat \007"'
2. # allocate memory for bash running
salloc -p sleuths -w jetski --mem=20000 --reservation triesch-shared --gres gpu:rtx2080ti:1 -t UNLIMITED

2020.08.31
1. sbatch: error: Batch script contains DOS line breaks (\r\n)
   sbatch: error: instead of expected UNIX line breaks (\n).
   dos2unix doesn't exist in linux
   solution: tr -d '\r' < cluster.sh > cluster2.sh

2. scancel batch of jobs:
    scancel{JobID_1..JobID_2}

2020.09.10

To use UMAP you need to install umap-learn not umap.
So, in case you installed umap run the following commands to uninstall umap and install upam-learn instead:
pip uninstall umap
pip install umap-learn
And then in your python code make sure you are importing the module using:
import umap.umap_ as umap
Instead of
import umap

2020.10.01
1. access dict with dot
    from types import SimpleNamespace
                args = SimpleNamespace(**params)
17.08.2017
# short time fourier transform
https://kevinsprojects.wordpress.com/2014/12/13/short-time-fourier-transform-using-python-and-numpy/
STFT Algorithm:

So, we understand what we’re trying to make – now we have to figure out how to make it.  The data flow we have to achieve is pretty simple, as we only need to do the following steps:

    Pick out a short segment of data from the overall signal
    Multiply that segment against a half-cosine function
    Pad the end of the segment with zeros
    Take the Fourier transform of that segment and normalize it into positive and negative frequencies
    Combine the energy from the positive and negative frequencies together, and display the one-sided spectrum
    Scale the resulting spectrum into dB for easier viewing
    Clip the signal to remove noise past the noise floor which we don’t care about


# activation functions and their derivatives
def nonlin(self, z, deriv=False, nonlinearFun='relu'):
    #prime = 1000 * self.sigmoid_like(x) * (1 - self.sigmoid_like(x))
    if nonlinearFun=='relu':
        if deriv:
            return 1. * (z - 0 > 0)
        else:
            return z * (z - 0 > 0)
    if nonlinearFun=='leakyrelu':
        leaky = 0.01
        if deriv:
            return 1. * (z - 0 > 0) + leaky*(z-0<0)
        else:
            return z * (z - 0 > 0) + leaky*(z-0<0)
    if nonlinearFun=='sigmoid':
        if deriv:
            return 1 / (1+np.exp(-z)) * (1 - 1 / (1+np.exp(-z)))
        else:
            return 1 / (1+np.exp(-z))
    if nonlinearFun=='softmax':
        if deriv:
            return np.exp(z) / np.sum(np.exp(z)) - (np.exp(z) / (np.sum(np.exp(z))))**2
        else:
            return np.exp(z) / np.sum(np.exp(z), axis=0)
    if nonlinearFun=='tanh':
        if deriv:
            return (1-z**2)
        else:
            return np.tanh(z)

# matrix derivatives
Y = A * X
dYdX = A.T

Y = X * A
dYdX = A

Y = A.T * X * B
DYDX = A*B.T

Y = A.T * X.T * B
DYDX = B * A.T

D(X.T*A) = (DX.T/DX)*A + X.T(DA/DX) = I*A + X.T*0 = A

# what is the betst way to make a copy as new set of parameters?????

# make bptt_step safer from no-initialization
T = 5
bptt_truncate_step = 3
for t in np.arange(T)[::-1]:
    for bptt in np.arange(max(0, t-bptt_truncate_step), t+1)[::-1]:
        print("step {}, bptt step {}".format(t, bptt))
'''step 4, bptt step 4
step 4, bptt step 3
step 4, bptt step 2
step 4, bptt step 1
step 3, bptt step 3
step 3, bptt step 2
step 3, bptt step 1
step 3, bptt step 0
step 2, bptt step 2
step 2, bptt step 1
step 2, bptt step 0
step 1, bptt step 1
step 1, bptt step 0
step 0, bptt step 0
'''

# to_one_hot
def to_one_hot(num_classes, labels):
    '''Make int label into one-hot encoding
    Param:
    num_classes: int, number of classes
    labels: 1D array'''
    ret = np.eye(num_classes)[labels]

    return ret

23.10.2017
nice-plots: 3.5. Validation curves: plotting scores to evaluate models
http://scikit-learn.org/stable/modules/learning_curve.html

24.10.2017
1. Replacing spaces in the file names
    >>rename -n "s/ /_/g" *
2. plot with repeated xlabel:
    plt.figure()
    plt.xticks(range(100), np.tile(np.arange(10), 10))

27.10.2017
plt spectrogram
def plot_specgram(self, frames, sampFreq, title="Spectrogram", save_name="spectrogram", ifsave=False):
    plt.figure()
    cmap = plt.get_cmap('viridis') # this may fail on older versions of matplotlib
    vmin = -40  # hide anything below -40 dB
    cmap.set_under(color='k', alpha=None)

    sampFreq, frames = wavfile.read("song.wav")
    fig, ax = plt.subplots()
    if len(frames.shape) != 1:
        frames = frames[:, 0]     # first channel
    pxx, freq, t, cax = ax.specgram(frames,
                                    Fs=sampFreq,      # to get frequency axis in Hz
                                    cmap=cmap, vmin=vmin)
    cbar = fig.colorbar(cax)
    cbar.set_label('Intensity dB')
    ax.axis("tight")

    # Prettify
    import matplotlib
    import datetime

    ax.set_xlabel('time h:mm:ss')
    ax.set_ylabel('frequency kHz')

    scale = 1e3                     # KHz
    ticks = matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))
    ax.yaxis.set_major_formatter(ticks)

    def timeTicks(x, pos):
        d = datetime.timedelta(seconds=x)
        return str(d)
    formatter = matplotlib.ticker.FuncFormatter(timeTicks)
    ax.xaxis.set_major_formatter(formatter)

Paper:
    FOrmal models of language learning----nice open ended learning chart

2017.11.13
count occerrence of values in an array
>>np.unique([1, 1, 2, 2, 3, 3, 4])
array([1, 2, 3, 4])

Launching TensorBoard
To run TensorBoard, use the following command (alternatively >>python -m tensorboard.main)
>>tensorboard --logdir=path/to/log-directory
TensorBoard 0.1.8 at http://digda:6006 (Press CTRL+C to quit)


2017.11.14
deal with the first nonvalid step
# Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=args.max_checkpoints)

    try:
        saved_global_step = load(saver, sess, restore_from)
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1

    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)

    step = None
    last_saved_step = saved_global_step
    try:
        for step in range(saved_global_step + 1, args.num_steps):
            ...
    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()

2017.11.16

'''plot like animation'''
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)   # plot original
plt.ion()
plt.show()
while in training:
    try:   # in case it is the first time, you can't do the operation
        ax.lines.remove(lines[0])
    except:
        pass

    lines = ax.plt(x_data, prediction, 'r-')
    plt.pause(0.1)

optimizer:
    SGD:
        W += -learning_rate * dx   # zig zag to the mountain feet
    Momentum:
        m = b1 * m - learning_rate * dx  # inertia go downwards
        W += m
    AdaGrad:
        v += dx^2
        W += -learning_rate * dx / sqrt(v)   # bad shoe, so go straight line
    RMSProp:   # combine momentum and AdaGrad
        v = b1 * v - (1 - b1) * dx^2
        W += -learning_rate * dx / sqrt(v)
    Adam: # add the -learning_rate*dx part from momentum in RMSProp
        m = b1 * m - (1 - b1) * dx -----------Momentum
        v = b2 * v - (1 - b2) * dx^2 ---------AdaGrad
        W += -learning_rate * m / sqrt(v)

tensorboard:
    sess = tf.Session()
    merged = tf.merge_all)summaries()
    write = tf.train.SummaryWriter("logs/", sess.graph)

    for step....:

        if step % 50 == 0:
            result = sess.run(merged, feed_dict={x: x_data, y: y_data})
            writer.add_summary(result, step)

2017.11.21 deal with out of memory problem

SLURM:
    >>srun -p x-men -c 10 python3 train.py --data_dir=corpus --gc_channels=32 --restore_from logdir/train/2017-11-20T11-05-18

Commands
'salloc' is used to allocate resources for a job in real time. Typically this is used to allocate resources and spawn a shell. The shell is then used to execute srun commands to launch parallel tasks.
'sbatch' is used to submit a job script for later execution. The script will typically contain one or more srun commands to launch parallel tasks.
'scancel #JOB number' is used to cancel a pending or running job or job step. It can also be used to send an arbitrary signal to all processes associated with a running job or job step.
'sinfo' reports the state of partitions and nodes managed by Slurm. It has a wide variety of filtering, sorting, and formatting options.
'smap' reports state information for jobs, fftpartitions, and nodes managed by Slurm, but graphically displays the information to reflect network topology.
'squeue' reports the state of jobs or job steps. It has a wide variety of filtering, sorting, and formatting options. By default, it reports the running jobs in priority order and then the pending jobs in priority order.
'srun' is used to submit a job for execution or initiate job steps in real time. srun has a wide variety of options to specify resource requirements, including: minimum and maximum node count, processor count, specific nodes to use or not use, and specific node characteristics (so much memory, disk space, certain required features, etc.). A job can contain multiple job steps executing sequentially or in parallel on independent or shared resources within the job's node allocation.

2017-11-23
train wavenet
Generate wavenet:
    'with no global condition'
    >>python3 generate.py --num_samples 16000  --wav_out_path results/generate/2017-11-15T13-46-47_speaker228_001_noGlobalCOndition.wav logdir/train/2017-11-15T13-46-47/model.ckpt-88
    'with no condition on OLLO'
    >>python3 generate.py --num_samples 32000  --wav_out_path results/generate/epoch125000_OLLO_NOconditioned.wav logdir/train/2017-12-18T11-31-30_OLLO_125000/model.ckpt-27500 --gc_channels=32 --gc_id=404 --gc_cardinality=411
    'with global condition'
    >>python3 generate.py --num_samples 32000  --wav_out_path results/generate/epoch125000_NOconditioned.wav logdir/train/2017-12-18T11-31-30_OLLO_125000/model.ckpt-27500 --gc_id=311 --gc_cardinality=377 --gc_channels=32 --wav_seed=rest_corpus/p226/p226_003.wav
    'with global condition on OLLO'
    >>python3 generate.py --num_samples 32000  --wav_out_path results/generate/epoch125000_OLLO_NOconditioned.wav logdir/train/2017-12-18T11-31-30_OLLO_125000/model.ckpt-27500 --gc_id=311 --gc_cardinality=377 --gc_channels=32 --wav_seed=rest_corpus/p226/p226_003.wav
Train:
    >>tensorboard --logdir=logdir/generate/2017-12-08T16-39-18
    'ON rest_corpus'
    >>python3 train.py --data_dir=rest_corpus --gc_channels=32 --restore_from logdir/train/2017-12-04T13-48-11_12000   #only give the dir
    'ON OLLO'
    >>python3 train.py --data_dir= --gc_channels=32 --restore_from logdir/train/2017-12-04T13-48-11_12000   #only give the dir

    >>srun -p x-men python3 train.py --gc_channels=32 --data_dir=rest_corpus --restore_from logdir/train/2017-11-21T14-12-14_42500


rename file "replace space in the file name with _"
>>for file in *; do mv "$file" ${file// /_}; done

run with cluster:
    srun -p x-men --mem 10GB -c 10 python hello.py


2017.11.29
clean git:
    everyday:
        while not going home:
            1. go to your master branch and pull from origin >>git pull
            2. work on an issue A, open a branch "issueA"  >>git checkout -b issueA
            3. constantly go back to your master and keep your master uptodate   >>git checkout master
            4. go back to your branch "issueA", >>git rebase

        >>git status
        "nothing is new!!"

prediction error:
    learn to make the movements so that it will minimize the error between
predict the change of bases function given the movement


20017.11.30
### Git Notes from Alex
# Interactive file adding
https://alblue.bandlem.com/2011/10/git-tip-of-week-interactive-adding.html

# Undo commit
https://stackoverflow.com/questions/927358/undo-the-last-git-commit

# Amend your last commit
$ git add .
$ git commit --amend

# List all files currently being tracked under the branch master
$ git ls-tree -r master --name-only

# List all branches of remote
$ git ls-remote <remote>
$ git remote show <remote>

# List all branches with last modification date being tracked locally
$ for k in `git branch -r | perl -pe 's/^..(.*?)( ->.*)?$/\1/'`; do echo -e `git
show --pretty=format:"%Cgreen%ci %Cblue%cr%Creset" $k -- | head -n 1`\\t$k; done |
sort -r

# Push a new local branch to a remote Git repository and track it too
$ git checkout -b <feature_branch_name>
... edit files, add and commit ...
$ git push -u origin <feature_branch_name>

# Push a local Git branch to master branch in the remote
$ git push <remote> <local_branch_name>:<remote_branch_to_push_into>
$ git push origin develop:master

# Delete a Git branch on remote
$ git push origin --delete <branch_name>

# Delete a Git branch locally
$ git branch -D <branch_name>

# Ignore files in Git
https://git-scm.com/docs/gitignore

# Gitignore is not working
https://stackoverflow.com/questions/11451535/gitignore-is-not-working

## More Advanced commands, handle with care
# Make a patch
https://ariejan.net/2009/10/26/how-to-create-and-apply-a-patch-with-git/

# Rewrite Git history with rebase
http://git-scm.com/book/en/Git-Tools-Rewriting-History


2017.12.13

def reconstruct_audio_from_spec( data):
    '''
    Param:
    data: array-like data, wav data'''

    dt = 0.1  #  define a time increment (seconds per sample)
    N = len(data)

    Nyquist = 1/(2*dt)  #  Define Nyquist frequency
    df = 1 / (N*dt)  #  Define the frequency increment

    G = np.fft.fftshift(np.fft.fft(data))  #  Convert "data" into frequency domain and shift to frequency range below
    f = np.arange(-Nyquist, Nyquist-df, df) #  define frequency range for "G"

    if len(G) != len(f):
        length = min(len(G), len(f))
    G_new = G[:length]*(1j*2*np.pi*f[:length])

    data_rec = np.abs(np.fft.ifft(np.fft.ifftshift(G_new)))

    plt.figure()
    plt.plot(data, 'b-', label='original')
    plt.hold(True)
    plt.plot(data_rec, 'm-', label='reconstruction', alpha=0.6)
    plt.legend(loc="best")
    plt.xlabel("frames")

2017.12.22
-split audio in SORN_WithFSD_LU
-Audio reconstruction on phase:  The most often-used phase reconstruction technique comes from  Griffin and Lim [1984] in ibrosa library
-DCGAN and spectrogram: http://deepsound.io/dcgan_spectrograms.html


2018.1.08
make gifs
# Make the gifs -- vae repo
if FLAGS.latent_dim == 2:
os.system(
    'convert -delay 15 -loop 0 {0}/posterior_predictive_map_frame*png {0}/posterior_predictive.gif'
    .format(FLAGS.logdir))

2018.01.09
multiprocessing make parallel plots
"""
# define what to do with each data pair ( p=[3,5] ), example: calculate product
def myfunc(p):
    #product_of_list = np.prod(p)

    xx = p[0]
    yy =  p[1]
    plt.plot(1, 2)
    plt.savefig("{}.png".format(yy))

def multi():
    xx = [2,1,8,9]
    yy = [3,4,5,7]
    data_pairs = map(lambda x,y: [x, y], xx, yy)

    pool = Pool()
    print data_pairs
    pool.map(myfunc, data_pairs)

multi()

"""

2018.01.12   spectrogram to audio_dir

# whole process: load audio--get spectrogram--scale to pixels--
#                from pixel values--scale up to (0, -4) spectrogram--recover audio
# functions are in usefull.../utils/spectrogramer.py
rate, data = wav.read('OLLO2.wav')
IPython.display.Audio(data=data, rate=rate)
wav_spectrogram = pretty_spectrogram(data.astype('float64'), fft_size = fft_size,
                                   step_size = step_size, log = True, thresh = spec_thresh)

# t, f, ss = signal.
scale_spec = scale_data(np.transpose(wav_spectrogram))   # scale up to image data
scale_spec = np.round(scale_spec)

crop_spec = crop_center(scale_spec,128,0)    # cut middle 128*128

BACK = scale_data(np.transpose(scale_spec), new_max=0, new_min=-4)   # scale back to wav spectrogram


recovered_audio_orig = invert_pretty_spectrogram(BACK, fft_size = fft_size,
                                            step_size = step_size, log = True, n_iter = 10)
IPython.display.Audio(data=recovered_audio_orig, rate=rate) # play the audio
# scale_spec, wav_spectrogram, BACK

inverted_spectrogram = pretty_spectrogram(recovered_audio_orig.astype('float64'), fft_size = fft_size,
                                   step_size = step_size, log = True, thresh = spec_thresh)
fig, ax = plt.subplots(nrows=1,ncols=1)
ax.matshow(scale_spec, interpolation='nearest', aspect='auto', cmap="gray_r", origin='lower')

2018.01.18
# tensorflow mask out zero values
import numpy as np
import tensorflow as tf
input = np.array([[1,0,3,5,0,8,6]])
X = tf.placeholder(tf.int32,[None,7])
zeros = tf.cast(tf.zeros_like(X),dtype=tf.bool)
ones = tf.cast(tf.ones_like(X),dtype=tf.bool)
loc = tf.where(input!=0,ones,zeros)
result=tf.boolean_mask(input,loc)
with tf.Session() as sess:
 out = sess.run([result],feed_dict={X:input})
 print (np.array(out))


2018.01.19
subplot axis is not visible but can use ylabel
fig = plt.figure(frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)tensorflow restore
plt.title("score on generated images")
#ipdb.set_trace()
for j in range(16):
    ax1 = fig.add_subplot(4, 4, j+1)
    #ax1.set_axis_off()
    #fig.add_axes(ax1)
    im = imgtest[j, :, :, 0]
    plt.imshow(im.reshape([128, 128]), cmap='Greys')
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    plt.ylabel("score={}".format(np.int(d_result[j]*10000)/10000.))
plt.subplots_adjust(left=0.07, bottom=0.02, right=0.93, top=0.98,
    wspace=0.02, hspace=0.02)

2018.301.20
screen:
    screen -r: show the list of all the screen running
    ctrl A +D: detach the screen, run the program in the background


2018.01.25
tensorflow data input pipline
enqueue files
problem: tf.errors.OutOfRangeError exception
        want to reuse dataqueue in multi-epochs
1. # https://github.com/tensorflow/tensorflow/issues/2514
    '''
    Multi-epoch use of queues might be simplified by adding one of the following:

        A queue.reset(), that throws one tf.errors.OutOfRangeError on dequeue() or some other exception.
        A queue.close(reset=True), that only throws one tf.errors.OutOfRangeError on dequeue() or some other exception.

    example usage of 1):

    q = tf.FIFOQueue(...)
    placeholder = ...
    enqueue_op = q.enqueue(placeholder)
    ....

    def producer(data_dir, sess, q, enqueue_op, placeholder):
      for ...:
        sess.run(enqueue_op, {placeholder:...})
      sess.run(q.reset())

    def do_epoch(data_dir, learn):
      threading.Thread(target=producer, args=(data_dir, sess, q, enqueue_op, placeholder)).start()
      while True:
        try:
          sess.run(...)
        exception tf.errors.OutOfRangeError:
          break

    for epoch in range(NUM_EPOCHS):
      ... = do_epoch(TRAIN_DIR, learn=True)
      ... = do_epoch(TEST_DIR, learn=False)
'''
2. https://www.tensorflow.org/versions/r1.3/programmers_guide/datasets
3. https://stackoverflow.com/questions/44132579/feed-data-into-a-tf-contrib-data-dataset-like-a-queue/45928467#45928467
    The new Dataset.from_generator() method allows you to define a Dataset that is fed by a Python generator.

4. https://stackoverflow.com/questions/33849617/how-do-i-convert-a-directory-of-jpeg-images-to-tfrecords-file-in-tensorflow

2018.01.31
'''example-of-tensorflows-new-input-pipeline.html'''
https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html

reading from files:
https://www.tensorflow.org/api_guides/python/reading_data
A typical pipeline for reading records from files has the following stages:
1. The list of filenames
2. Optional filename shuffling
3. Optional epoch limit
4. Filename queue
5. A Reader for the file format
6. A decoder for a record read by the reader
7. Optional preprocessing
8. Example queue


2018.3013
tomato timer for linux
sleep 1500 && zenity --warning --text="25 passed. Take a break!"; sleep 300 && zenity --warning --text="Get back to work!"
sleep 1500 && zenity --warning --text="25 passed. Take a break!"; sleep 300 && zenity --warning --text="Get back to work!"
sleep 1500 && zenity --warning --text="25 passed. Take a break!"; sleep 300 && zenity --warning --text="Get back to work!"
sleep 1500 && zenity --warning --text="25 passed. Take a break!"; sleep 900 && zenity --warning --text="Get back to work!"


countdown in linum terminal
countdown=1500 now=$(date +%s) watch -tpn1 echo '$((now-$(date +%s)+countdown))'


2018.03.26
1. creat MNE data format
http://martinos.org/mne/stable/auto_tutorials/plot_creating_data_structures.html#creating-raw-objects

2. Creating MNE objects from data arrays
http://martinos.org/mne/stable/auto_examples/io/plot_objects_from_arrays.html#sphx-glr-auto-examples-io-plot-objects-from-arrays-py

2018.03.27
1. rename all files in Ubuntu
    rename 's/\.txt$/.csv/' *
    >> rename all files to .csv
2. recursively download files with a web_link
    wget -r -np -R "index.html*" your_link
3. convert an array to one-hot vector
    np.eye(n_classes)[values]
4. rename part of the file parttern:
    for i in *.csv; do mv $i $(echo $i | sed 's/:/_/g'); done   ## repplace the : with _ in all filenames"
2018.03.29
1. downsampling data
    from scipy.signal import decimate
    ds_data =  decimate(data, ds_factor)
2. in Geany, how to find a functiosn defined somewhere
    cntr + left-click

2018.04.10
plot smooth data    Epilepsy/function.py
plot fill_data   Epilepsy/function.py
colors I like:
    green:
    blue:
    red:
    purple:

2018.04.16
save array as csv

2018.04.18
tesnsorflow understand lstm:
    https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
how  to self define func in tensorflow dataset:
    https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html
    https://www.tensorflow.org/api_docs/python/tf/data/Dataset
DeepConvLSTM on sensory recording:
    https://github.com/sussexwearlab/DeepConvLSTM/blob/master/DeepConvLSTM.ipynb
Error:
    I think because I take too many samples, run OOM
    {x: mnist.test.images, y: mnist.test.labels}

visualize VAE prior, very good notebook:
    Tutorial: https://github.com/hsaghir/VAE_intuitions/blob/master/VAE_MNIST_keras.ipynb
    Original blog: https://hsaghir.github.io/data_science/denoising-vs-variational-autoencoder/

ConvVAE:
    https://www.kaggle.com/rvislaywade/visualizing-mnist-using-a-variational-autoencoder

Atrous convolution:
    http://liangchiehchen.com/projects/DeepLab.html

Wavenet from paper to code:
    https://www.youtube.com/watch?v=LPMwJ-67SpE

Atrous_conv:
    out_conv = []
    # dilation rate lets us use ngrams and skip grams to process
    for dilation_rate in range(max_dilation_rate):
        x = prefilt_x
        for i in range(3):
            x = Conv1D(32*2**(i),
                       kernel_size = 3,
                       dilation_rate = dilation_rate+1)(x)
        out_conv += [Dropout(0.5)(GlobalMaxPool1D()(x))]
    x = concatenate(out_conv, axis = -1)

2018.4.23
tensorflow profile tracing:
    with tf.Session() as sess:

        #profiler = tf.profiler.Profiler(sess.graph)
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        _, acc, c, summary = sess.run([optimizer, accuracy, cost, summaries], feed_dict={x: batch_data, y: batch_labels}, options=options, run_metadata=run_metadata)
        ####### # Create the Timeline object, and write it to a json file
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open(save_name + 'timeline_{}.json'.format(batch), 'w') as f:
            f.write(chrome_trace)
    then go to chrome://tracing load .json

how to install google chrome without sudo:
    http://indiayouthtechtips.blogspot.de/2012/03/how-to-install-google-chrome-without.html

2018.04.27
Dilated CNN:
    https://sthalles.github.io/deep_segmentation_network/
    @slim.add_arg_scope
    def atrous_spatial_pyramid_pooling(net, scope, depth=256):
        """
        ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
        (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
        :param net: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
        :param scope: scope name of the aspp layer
        :return: network layer with aspp applyed to it.
        """
        with tf.variable_scope(scope):
            feature_map_size = tf.shape(net)

            # apply global average pooling
            image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keep_dims=True)
            image_level_features = slim.conv2d(image_level_features, depth, [1, 1], scope="image_level_conv_1x1", activation_fn=None)
            image_level_features = tf.image.resize_bilinear(image_level_features, (feature_map_size[1], feature_map_size[2]))

            at_pool1x1 = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_0", activation_fn=None)

            at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1", rate=6, activation_fn=None)

            at_pool3x3_2 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_2", rate=12, activation_fn=None)

            at_pool3x3_3 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_3", rate=18, activation_fn=None)

            net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3,
                            name="concat")
            net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)
            return net

tensorflow input .csv:
    https://stackoverflow.com/questions/43621637/tensorflow-input-pipeline-error-while-loading-a-csv-file
    https://www.tensorflow.org/get_started/datasets_quickstart
    format reading .csv per line: https://stackoverflow.com/questions/37091899/how-to-actually-read-csv-data-in-tensorflow
    QueueRunner with .csv files: https://www.tensorflow.org/api_guides/python/reading_data:
        def read_my_file_format(filename_queue):
            reader = tf.SomeReader()
            key, record_string = reader.read(filename_queue)
            example, label = tf.some_decoder(record_string)
            processed_example = some_processing(example)
            return processed_example, label

        def input_pipeline(filenames, batch_size, num_epochs=None):
            filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=True)
            example, label = read_my_file_format(filename_queue)
            # min_after_dequeue defines how big a buffer we will randomly sample
            #   from -- bigger means better shuffling but slower start up and more
            #   memory used.
            # capacity must be larger than min_after_dequeue and the amount larger
            #   determines the maximum we will prefetch.  Recommendation:
            #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
            min_after_dequeue = 10000
            capacity = min_after_dequeue + 3 * batch_size
            example_batch, label_batch = tf.train.shuffle_batch(
            [example, label], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
            return example_batch, label_batch
        # Create the graph, etc.
        init_op = tf.global_variables_initializer()

        # Create a session for running operations in the Graph.
        sess = tf.Session()
        # Initialize the variables (like the epoch counter).
        sess.run(init_op)
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                # Run training steps or whatever
                sess.run(train_op)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()

2018.05.02
tensorflow input pipeline:
    1. one .csv
    2. GOOD!! read multiple .csv files each a sone training sample
        https://stackoverflow.com/questions/49525056/tensorflow-python-reading-2-files/49548224#49548224
    3: GOOD!!! works one row by row training sample:
        https://stackoverflow.com/questions/49525056/tensorflow-python-reading-2-files/49548224#49548224
    4. Good!:
        https://learningtensorflow.com/ReadingFilesBasic/
    5. :
        https://stackoverflow.com/questions/49899526/tensorflow-input-pipeline-where-multiple-rows-correspond-to-a-single-observation
    6. : https://www.programcreek.com/python/example/90498/tensorflow.TextLineReader
        def read_csv(batch_size, file_name):
            filename_queue = tf.train.string_input_producer([file_name])
            reader = tf.TextLineReader(skip_header_lines=0)
            key, value = reader.read(filename_queue)
            # decode_csv will convert a Tensor from type string (the text line) in
            # a tuple of tensor columns with the specified defaults, which also
            # sets the data type for each column
            decoded = tf.decode_csv(
                value,
                field_delim=' ',
                record_defaults=[[0] for i in range(FLAGS.max_sentence_len * 2)])

            # batch actually reads the file and loads "batch_size" rows in a single tensor
            return tf.train.shuffle_batch(decoded,
                                          batch_size=batch_size,
                                          capacity=batch_size * 50,
                                          min_after_dequeue=batch_size)
    7.: high level performance: https://www.tensorflow.org/performance/datasets_performance
        ### load data
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename).skip(0).map(decode_csv))

        dataset = dataset.batch(total_rows).shuffle(buffer_size=1000).repeat()   ###repeat().

        iterator = dataset.make_initializable_iterator()
        batch_data = iterator.get_next()
    8. data pipeline tutorial GOOD: https://cs230-stanford.github.io/tensorflow-input-data.html

    7. Yay! works:
        def decode_csv(line):
            defaults = [[0.0]]*512
            csv_row = tf.decode_csv(line, record_defaults=defaults)#
            data = tf.stack(csv_row)
            return data
        filenames = ['data/test/2014-10-06T21:28:54.csv', 'data/test/2014-11-03T12:01:09.csv', 'data/test/BL-2014-09-26T00:25:34.csv']
        dataset5 = tf.data.Dataset.from_tensor_slices(filenames)
        dataset5 = dataset5.flat_map(lambda filename: tf.data.TextLineDataset(filename).skip(0).map(decode_csv))

        dataset5 = dataset5.batch(4).shuffle(buffer_size=1000).repeat(20)   ###repeat().

        iterator5 = dataset5.make_initializable_iterator()
        next_element5 = iterator5.get_next()

        t1 = time.time()
        with tf.Session() as sess:
            # Train 2 epochs. Then validate train set. Then validate dev set.
            sess.run(iterator5.initializer)
            for _ in range(10):
                features = sess.run(next_element5)
                      # Train...
                #print("shape:", features.shape)
                print("label", features.shape, 'time', time.time()-t1)

            # Validate (cost, accuracy) on train set
            ipdb.set_trace()
            print("\nDone with the first iterator\n")


2018.05.07
Very nice tutorial on :
    https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607

2018.5.17
visualize conv filters:
    https://stackoverflow.com/questions/33783672/how-can-i-visualize-the-weightsvariables-in-cnn-in-tensorflow/33794463#33794463
    https://github.com/grishasergei/conviz
    https://medium.com/@awjuliani/visualizing-neural-network-layer-activation-tensorflow-tutorial-d45f8bf7bbc4
correct sliding window function:
    def sliding_window(data_x, data_y, num_seg=5, window=128, stride=64):
        '''
        Param:
            datax: array-like data shape (batch_size, seq_len, channel)
            data_y: shape (num_seq, num_classes)
            num_seg: number of segments you want from one seqeunce
            window: int, number of frames to stack together to predict future
            noverlap: int, how many frames overlap with last window
        Return:
            expand_x : shape(batch_size, num_segment, window, channel)
            expand_y : shape(num_seq, num_segment, num_classes)
            '''
        assert len(data_x.shape) == 3
        expand_data = []
        for ii in range(data_x.shape[0]):
            num_seg = (data_x.shape[1] - window) // stride + 1
            shape = (num_seg, window, data_x.shape[-1])      ## done change the num_seq
            strides = (data_x.itemsize*stride*data_x.shape[-1], data_x.itemsize*data_x.shape[-1], data_x.itemsize)
            expand_x = np.lib.stride_tricks.as_strided(data_x[ii, :, :], shape=shape, strides=strides)
            expand_data.append(expand_x)
        expand_y = np.repeat(data_y,  num_seg, axis=0).reshape(data_y.shape[0], num_seg, data_y.shape[1]).reshape(-1, data_y.shape[1])
        return np.array(expand_data).reshape(-1, window, data_x.shape[-1]), expand_y

2018.06.05
time series train and test split cross validation
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

2018.06.06
access variables with name:
    with tf.variable_scope('fc_0/fully_connected', reuse=True):
        ww = tf.get_variable('weights')
2. get all variables associated with a layer,
    with tf.variable_scope('fc_0/fully_connected', reuse=True):
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc_0/fully_connected')
3. Pandas:
    https://jeffdelaney.me/blog/useful-snippets-in-pandas/

https://github.com/tensorlayer/tensorlayer/issues/146

2018.06.19
t-SNE tutorial:
    https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

2018.06.26
sort directory by space usage:
    >>cd elu/LU/
    >>du --max-depth=1
    elu@digda ~/LU$ >>du --max-depth=1
        3879752 ./1_Infant_speech_acquisition
        70724   ./documents of all time
        599712  ./software
        354048  ./Books
        36136   ./5_Goal_Robot
        76544   ./Plotting_data_tips
        25382500    ./3_speech_recognition
        347148  ./learning
        8132    ./courses
        79884952    ./2_Neural_Network
        75396   ./4_discussion_meeting_paper_other_reading_
        2232    ./4_paper_other_reading_discussion_meeting
        110729868   .

2018.06.27
plot without frame
for cluster in range(num_clusters):
        ind_cluster = np.where(km.labels_ == cluster)[0]
        fig, axes = plt.subplots(10, 10,
                     subplot_kw={'xticks': [], 'yticks': []})   ## figsize=(12, 6),

        #fig.subplots_adjust(hspace=0.3, wspace=0.05)
        for ind in range(np.min(ind_cluster.size), 100):
            for ax in axes.flat:
                ax.plot(data[ind_cluster[ind]])

2018.06.29
np.histgram(aa, bins='auto'))
plt.hist(aa, bins)

2018.07.02
check swap space usage: https://www.tecmint.com/commands-to-monitor-swap-space-usage-in-linux/

2018.07.02
load data repeat:  generator
def EEG_data(data_dir, pattern='Data*.csv', withlabel=False, num_samples=784, batch_size=128):

    files = find_files(data_dir, pattern='Data*.csv', withlabel=False)

    datas = np.zeros((len(files), 10240*2))

    for ind, filename in enumerate(files):
        data = read_data(filename, header=None, ifnorm=True)
        datas[ind, :] = np.append(data[:, 0], data[:, 1])  ## only use one channel

    useful_length = (datas.size // input_dim) * input_dim
    datas = datas.flatten()[0: useful_length].reshape(-1, input_dim)  ##
    while True:
        try:

            start = np.random.randint(len(datas) - batch_size)
            batch_x = datas[start: start+batch_size]

            yield np.asarray(batch_x)

        except Exception as e:
            print('Could not produce batch of sinusoids because: {}'.format(e))
            sys.exit(1)

2018.07.05
add parent path in import
import sys
sys.path.insert(0, os.path.abspath('..'))

2018.07.16
Valentin data in FIAS local storage:
    elu@digda /home>> cd epilepsy-data
read HDF5 file:
    https://confluence.slac.stanford.edu/display/PSDM/How+to+access+HDF5+data+from+Python
import h5py
import matplotlib.pyplot as plt
from scipy.signal import detrend
hf = h5py.File('EpimiRNA_1.2-27_recordings.h5', 'r')
key = hf.keys()   ##[u'data', u'meta']
data = hf.get(hf.keys()[0])   ##HDF5 group "/data" (966 members)
data.items()  ## 966 (u'2014-11-03T15:01:08', <HDF5 group "/data/2014-11-03T15:01:08" (4 members)>)]
name = data.items()[ind][0]  # '2014-11-03T15:01:08'
values = data.items()[ind][1]  # values = data['2014-11-03T15:01:08']. <HDF5 group "/data/2014-11-03T15:01:08" (4 members)>
group_names = values.items()### 'NDF_File_Name', 'NDF_local_timestamp', 'time', 'voltage'
voltage = values.items()[3]  #(u'voltage', <HDF5 dataset "voltage": shape (1843200,), type "<f8">)
voltage = values.items()[3][1]
dev1 = detrend(voltage)
plot:
>>> name = '2014-10-01T06:54:15'
>>> values = data[name]
>>> vol = values.items()[3][1]
>>> dev = detrend(vol)
>>> plt.subplot(211), plt.title('Signal and spectrogram of {}'.format(name), fontsize=22), plt.plot(np.arange(dev.size)/ 512.0, dev, 'purple'), plt.xlabel('time / s', fontsize=20), plt.ylabel('amplitude', fontsize=20), plt.xlim([0, dev.size/512.0]), plt.subplot(212), plt.specgram(dev, detrend='linear', cmap='viridis', NFFT=5120, Fs=512, noverlap=3072, scale_by_freq=True, vmin=-2), plt.xlim([0, dev.size/ 512.0]), plt.ylim([0, 100]), plt.xlabel('time / s', fontsize=20), plt.ylabel('frequency', fontsize=20), plt.show()

>>> plt.subplot(211), plt.title('Signal and spectrogram of {}'.format(name), fontsize=22), plt.plot( np.arange(dev.size)/ 512.0, dev, 'purple'), plt.xlabel('time / s', fontsize=20), plt.ylabel('amplitude', fontsize=20), plt.xlim([0, dev.size/512.0]), plt.subplot(212), plt.specgram(dev, detrend='linear', cmap='YlGnBu', NFFT=5120, Fs=512, noverlap=3072, scale_by_freq=True, vmin=-2), plt.xlim([0, dev.size/ 512.0]), plt.ylim([0, 100]), plt.xlabel('time / s', fontsize=20), plt.ylabel('frequency', fontsize=20), plt.yticks([2.0, 7.0, 12, 20, 40, 60, 80, 100, 150]), plt.show()

### get plots for multiple data recordings
names = data.items()[61:86]
values, vols, devs, ns = [], [], [], []
for i in range(len(names)): values.append(names[i][1])
for i in range(len(names)): vols.append(values[i].items()[3][1])
for i in range(len(names)): devs.append(detrend(vols[i]))
for i in range(len(names)): ns.append(names[i][0])
for name, dev in zip(ns[1:], devs[1:]): plt.figure(figsize=(16, 11)), plt.subplot(211), plt.title('Signal and spectrogram of {}'.format(name), fontsize=22), plt.plot(np.arange(dev.size)/ 512.0, dev, 'purple'), plt.xlabel('time / s', fontsize=20), plt.ylabel('amplitude', fontsize=20), plt.xlim([0, dev.size/512.0]), plt.subplot(212), plt.specgram(dev, detrend='linear', cmap='YlGnBu', NFFT=3072, Fs=512, noverlap=128, scale_by_freq=True, vmin=-2), plt.xlim([0, dev.size/ 512.0]), plt.ylim([0, 150]), plt.xlabel('time / s', fontsize=20), plt.ylabel('frequency', fontsize=20), plt.yticks([2, 7, 12, 20, 40, 60, 80, 100, 150]), plt.savefig('/home/elu/Desktop/1243/'+'signal{}.png'.format(name), format='png'), plt.close()

## save data
name = data.items()[61]
value = dd[1].items()[3][1]
np.savetxt('/home/elu/Desktop/1227/data'+'BL-{}.csv'.format(name), dev.reshape(-1, 512), delimiter=',', fmt="%10.5f", comments='')

for i in range(60): np.savetxt('/home/elu/Desktop/1227-MFE-1103/'+'signal{}_min{}.csv'.format(name[0], i), np.array(dev[i*30720:(i+1)*30720]), header=name[0], delimiter=',', fmt="%10.5f", comments='')


2018.07.17
How to visualize the CNN kernels?
    2. Google search items:
        RIght: https://www.google.com/search?newwindow=1&client=ubuntu&channel=fs&ei=QuZNW7rJJMm4sQHa5rt4&q=tensorboard+summary+weights&oq=tensorboard+&gs_l=psy-ab.1.0.35i39k1l2j0l2j0i20i263k1j0l5.123896.134636.0.137860.36.24.10.0.0.0.184.1958.12j7.20.0....0...1.1.64.psy-ab..6.30.2118.6..0i67k1.86.QJAWfPyEgXg
    3. Very good tensorboard tutorial with tensorboard example and code:
        https://jhui.github.io/2017/03/12/TensorBoard-visualize-your-learning/
    4. very good example with stacked grayscale filters:
        https://stackoverflow.com/questions/33802336/visualizing-output-of-convolutional-layer-in-tensorflow
    1. function put_kernels_on_grid:
        https://gist.github.com/kukuruza/03731dc494603ceab0c5
        def put_kernels_on_grid (kernel, grid_Y, grid_X, pad = 1):

            '''Visualize conv. features as an image (mostly for the 1st layer).
            Place kernel into a grid, with some paddings between adjacent filters.

            Args:
              kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
              (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                                   User is responsible of how to break into two multiples.
              pad:               number of black pixels around each filter (between them)

            Return:
              Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
            '''

            x_min = tf.reduce_min(kernel)
            x_max = tf.reduce_max(kernel)

            kernel1 = (kernel - x_min) / (x_max - x_min)

            # pad X and Y
            x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

            # X and Y dimensions, w.r.t. padding
            Y = kernel1.get_shape()[0] + 2 * pad
            X = kernel1.get_shape()[1] + 2 * pad

            channels = kernel1.get_shape()[2]

            # put NumKernels to the 1st dimension
            x2 = tf.transpose(x1, (3, 0, 1, 2))
            # organize grid on Y axis
            x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels])) #3

            # switch X and Y axes
            x4 = tf.transpose(x3, (0, 2, 1, 3))
            # organize grid on X axis
            x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels])) #3

            # back to normal order (not combining with the next step for clarity)
            x6 = tf.transpose(x5, (2, 1, 3, 0))

            # to tf.image_summary order [batch_size, height, width, channels],
            #   where in this case batch_size == 1
            x7 = tf.transpose(x6, (3, 0, 1, 2))

            # scale to [0, 255] and convert to uint8
            return tf.image.convert_image_dtype(x7, dtype = tf.uint8)

Good!!!
with tf.name_scope(‘fc1’):
    layer1 = tf.layers.dense(features, 512, activation=tf.nn.relu, name=’fc1′)
    fc1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ‘fc1’)
    tf.summary.histogram(‘kernel’, fc1_vars[0])
    tf.summary.histogram(‘bias’, fc1_vars[1])
    tf.summary.histogram(‘act’, layer1)

with tf.variable_scope('conv1') as scope:
    conv = tf.layers.con2d()

    kernel, bias = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name)
    grid = func.put_kernels_on_grid (kernel, pad = 2)
    tf.image.summary(scope.name, grid, max_outputs=1)

2018.07.18
Profile script:
    https://docs.python.org/2/library/profile.html

steps:
    1. python -m cProfile [-o output_file] [-s sort_order] myscript.py
    2. python
    3. import pstats
        p = pstats.Stats('tf_profile')
        p.sort_stats('tottime').print_stats(20)

2018.07.18
Replace dataset iterator with tfrecords and queue
### tensorflow dataset
    dataset_train = tf.data.Dataset.from_tensor_slices((files_train, labels_train)).repeat().batch(batch_size).shuffle(buffer_size=10000)
    iter = dataset_train.make_initializable_iterator()
    ele = iter.get_next()   #you get the filename

2018.07.23
compute area_under_curve:
    1.from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels_train_hot, outputs)
    2. area_under_curve = tf.contrib.metrics.streaming_auc(labels=y, predictions=outputs, name='auc')

2018.07.25
autocorrelation:
    err = datas[ind, :, 0] - np.mean(datas[ind, :, 0])
    variance = np.sum(err ** 2) / datas[ind, :, 0].size
    correlated = np.correlate(err, err, mode='full')/variance
    correlated = correlated[correlated.size//2:]

nice CNN tutorial:
    setosa.io/ev/image-kernels

2018.07.26
how to interpret learned weights:
    https://stackoverflow.com/questions/47745313/how-to-interpret-weight-distributions-of-neural-net-layers
weights initialization:
    https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94

2018.07.30
add subplots to subplots:
    https://www.python-kurs.eu/matplotlib_unterdiagramme.php


2018.08.27
plt sparse matrix:
    import scipy.sparse as sps
    import matplotlib.pyplot as plt
    a = sps.rand(1000, 1000, density=0.001, format='csr')
    plt.spy(a)
    plt.show()

plt smooth histogram (density):
    import pandas as pd
    pd.DataFrame(data).plot(kind='density'))



2018.08.30
Valentin data in FIAS local storage:
    elu@digda /home>> cd epilepsy-data
read HDF5 file:
    https://confluence.slac.stanford.edu/display/PSDM/How+to+access+HDF5+data+from+Python
import h5py
hf = h5py.File('EpimiRNA_1.2-27_recordings.h5', 'r')
key = hf.keys()
data = hf.get(key[0])
data.items()  ## (u'2014-11-03T15:01:08', <HDF5 group "/data/2014-11-03T15:01:08" (4 members)>)]
name = data.items()[0][0]  # '2014-09-22T11:24:31'~'2014-11-03T15:01:08'
values = data.items()[0][1]  # <HDF5 group "/data/2014-11-03T15:01:08" (4 members)>
group_names = values.items()### 'NDF_File_Name', 'NDF_local_timestamp', 'time', 'voltage'
voltage = values.items()[3]  #(u'voltage', <HDF5 dataset "voltage": shape (1843200,), type "<f8">)
voltage = values.items()[3][1]

from scipy import detrend
detr_v = detrend(voltage)
zvolt = scipy.stats.zscore(volt)
fr, psd = scipy.signal.welch(zvolt)
plt.semilogx(fr, psd)

plt.subplot(211), plt.plot(dev1, 'purple'), plt.subplot(212), plt.specgram(dev1, detrend='constant', NFFT=256, Fs=512, noverlap=128), plt.title('no trend in spectrogram; periodic high freq'), plt.colorbar()


2018.09.07
plot spectrogram:
    plt.subplot(211), plt.title('Signal and spectrogram of {}'.format(data.items()[ind][0]), fontsize=22), plt.plot(np.arange(dev1.size)/ 512.0, dev1, 'purple'), plt.xlabel('time / s', fontsize=20), plt.ylabel('amplitude', fontsize=20), plt.xlim([0, dev1.size/512.0]), plt.subplot(212), plt.specgram(dev1, detrend='linear', cmap='viridis', NFFT=3072, Fs=512, noverlap=128, scale_by_freq=True, vmin=vmin), plt.xlim([t[0], t[-1]]), plt.ylim([0, 100]), plt.xlabel('time / s', fontsize=20), plt.ylabel('frequency', fontsize=20), plt.show()

2018.09.17
position legend box:
    https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot


2018.10.17
tensorflow Textlinedataset read multiple lines as one training sample:
    https://stackoverflow.com/questions/49899526/tensorflow-input-pipeline-where-multiple-rows-correspond-to-a-single-observation
    def _parse_and_decode(filename, group_size):
        '''input would be (filename, label), decode the file in TextLineDataset and return decoded dataset
        decode csv file, get group_size seconds of data, give the label and return
        '''
        ## decode csv file, read group_size row as one sample
        ds = tf.data.TextLineDataset(filename)
        ds = ds.batch(group_size).skip(0).map(lambda line: decode_csv(line, group_size))

        return ds
    ## way 2
    filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]

    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    # Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
    # and then concatenate their contents sequentially into a single "flat" dataset.
    # * Skip the first line (header row).
    # * Filter out lines beginning with "#" (comments).
    dataset = dataset.flat_map(
        lambda filename: (
            tf.data.TextLineDataset(filename)
            .skip(1)
            .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))


tensorflow np.repeat() equivilant:
    labels = np.array([1, 2, 2, 0, 1, 2, 1, 2, 0, 0, 1, 0])
    aa = tf.tile(tf.reshape(labels, [-1, 1]), [1, 3])  ##repeat 3 times
    bb = tf.reshape(aa, [-1])
    repeatlabels = tf.reshape(bb)

2018.10.26
tensorflow datast train and test split:
    dataset.take()
    dataset.skip()
    https://stackoverflow.com/questions/47735896/get-length-of-a-dataset-in-tensorflow
GREAT dataset tutorial:
    https://cs230-stanford.github.io/tensorflow-input-data.html
    https://cs230-stanford.github.io/tensorflow-model.html

Order for dataset:
    To summarize, one good order for the different transformations is:
        create the dataset
        shuffle (with a big enough buffer size)
        repeat
        map with the actual work (preprocessing, augmentation…) using multiple parallel calls
        batch
        prefetch

2018.10.29
github contributor:
    https://github.com/CoolProp/CoolProp/wiki/Contributing%3A-git-development-workflow


2018.11.06
Pycharm shortcut:
    ctrl + B -- go to the declaration of a class, method or variable
    shift + F6 -- rename all places
    ctrl + Q -- see the documentation
    ctrl + shift + up/down  Code | Move Statement Up/Down action
    Ctrl+P brings up a list of valid parameters.
    Ctrl+Shift+Backspace (Navigate | Last Edit Location) brings you back to the last place where you made changes in the code.
    Ctrl+Shift+F7 (Edit | Find | Highlight Usages in File) to quickly highlight usages of some variable in the current file. Use F3 and Shift+F3 keys to navigate through highlighted usages. Press Escape to remove highlighting
    Ctrl+Space basic code completion ()
    Alt+Up and Alt+Down keys to quickly move between methods in the editor.

2018.11.09
get total size:
    total_bytes += os.path.getsize(os.path.join(root, f))
very good answer on map and flat_map:
    https://stackoverflow.com/questions/49116343/dataset-api-flat-map-method-producing-error-for-same-code-which-works-with-ma

2018.11.20
run charles classifier

2018.12.11
rename filenames in batch (Linux)):
    for f in $(find . -name '*.csv'); do mv $f ${f/:/_}; done;
    for f in $(find . -name '*'); do mv $f ${f/1270/32140}; done;


2018.12.17
'global_step': tf.train.get_global_step()

# Add summaries manually to writer at global_step
if writer is not None:
    global_step = results[-1]['global_step']
    for name, val in metrics_test.items():
        if 'matrix' not in name:
            summ = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=val[0])])
writer.add_summary(summ, global_step)

# Metrics for evaluation using tf.metrics (average over whole dataset)
# with tf.name_scope("metrics"):
#     # Streaming a confusion matrix for group with metrics together to update or init
#     batch_confusion = tf.confusion_matrix(labels_int, post_pred_int, num_classes=args.num_classes, name='confusion')
    # create an accumulator to hold the counts
    # confusion = tf.Variable(tf.zeros([args.num_classes, args.num_classes], dtype=tf.int32))
    # Create the update op for doing a "+=" accumulation on the batch
    # conf_update_op = confusion.assign(confusion + batch_confusion)
    # metrics = {
    #     'accuracy': tf.metrics.accuracy(labels=labels_int, predictions=post_pred_int),
    #     'loss': tf.metrics.mean(loss)
    #     # 'conf_matrix': (confusion, conf_update_op)
    # }
# TODO: there are two values in metrics["accuracy"], metrics["loss"], conf_matrix is not updated properly

# Group the update ops for the tf.metrics
# update_metrics_op = tf.group(conf_update_op, *[op for _, op in metrics.values()])
# metrics['conf_matrix'] = confusion

# Get the op to reset the local variables used in tf.metrics
# metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
# metrics_init_op = tf.variables_initializer(metric_variables)

2. def reduce_data_mean(ret):
    N = len(ret)
    mean_val = {}
    for key in ret[0].keys():
        if key != 'train_op':
            mean_val[key] = sum([b[key] for b in ret]) / N
    return mean_val

3. check GPU info on the clusters: scontrol show node name_node
4. def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    logging.info("Available GPU")
    logging.info([x.name for x in local_device_protos if x.device_type == 'GPU'])
    return

2018.12.20
map zscore normalization to dataset: better to noam it in the parse function:
    def parse_function(filename, label, args):
        """
        parse the file. It does following things:
        1. init a TextLineDataset to read line in the files
        2. decode each line and group args.secs_per_samp*args.num_segs rows together as one sample
        3. repeat the label for each long chunk
        4. return the transformed dataset
        :param filename: str, file name
        :param label: int, label of the file
        :param args: Param object, contains hyperparams
        :return: transformed dataset with the label of the file assigned to each batch of data from the file
        """
        skip = np.random.randint(0, args.secs_per_samp)
        decode_ds = tf.data.TextLineDataset(filename).skip(skip).map(decode_csv).batch(args.secs_per_samp*args.num_segs)
        decode_ds = decode_ds.map(scale_to_zscore)
        label = tf.tile(tf.reshape(label, [-1, 1]), [1, np.int((args.secs_per_file - skip) // (args.secs_per_samp * args.num_segs))])
        label = tf.reshape(label, [-1])  ## tensorflow np.repeat equivalent
        label_ds = tf.data.Dataset.from_tensor_slices(label)  ## make a label dataset

        transform_ds= tf.data.Dataset.zip((decode_ds, label_ds))

        return transform_ds
    2. if train on spectrogram, then we can use longer segments

2019.01.03
Run jobs on GPU:
    srun -p sleuths --mem=4000 --gres gpu:titanblack:1 python3 EPG_classification.py             # 100h 14 mins
List all devices:   # keep :1
    srun -p sleuths --gres gpu:titanblack:1 lspci
    
2019.01.17
pytorch load data:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
tensorflow examples:
    https://www.programcreek.com/python/example/90570/tensorflow.decode_csv

tensorflow load one .csv file as one training sample:
    def parse_function(filename, label, args=None):
        """
        parse the file. It does following things:
        1. init a TextLineDataset to read line in the files
        2. decode each line and group args.secs_per_samp*args.num_segs rows together as one sample
        3. repeat the label for each long chunk
        4. return the transformed dataset
        :param filename: str, file name
        :param label: int, label of the file
        :param args: Param object, contains hyperparams
        :return: transformed dataset with the label of the file assigned to each batch of data from the file
        """
        defaults = [[0.0]] * args.seq_len
        content = tf.read_file(filename)
        content = tf.decode_csv(content, record_defaults=default)
        decode_ds = scale_to_zscore(content)  # zscore norm the data
        if args.if_spectrum:
            decode_ds = get_spectrum(decode_ds, args=args)

        return decode_ds, label

2019.01.21
pytorch load data from multiple csv files:
    https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/dataset_loader.py
EEG inplementation cases:
    https://mediatum.ub.tum.de/doc/1422453/552605125571.pdf
EEGNet:
    paper: https://arxiv.org/pdf/1611.08024.pdf
    code: https://github.com/aliasvishnu/EEGNet/blob/master/EEGNet-PyTorch.ipynb

2019.03.11
Linux check disk space with df command

    Open the terminal and type the following command to check disk space.
    The basic syntax for df is:
    df [options] [devices]
    Type:
    df
    df -H:
        Filesystem                  Size  Used Avail Use% Mounted on
        udev                        4.0G     0  4.0G   0% /dev
        tmpfs                       804M   69M  735M   9% /run
        /dev/mapper/vg_system-root  3.2G  2.5G  432M  86% /
        /dev/vg_system/usr           34G   29G  3.3G  90% /usr
        tmpfs                       4.1G  141M  3.9G   4% /dev/shm
        tmpfs                       5.3M  4.1k  5.3M   1% /run/lock
        tmpfs                       4.1G     0  4.1G   0% /sys/fs/cgroup
        /dev/mapper/vg_system-var    11G  2.7G  7.3G  27% /var
        /dev/mapper/vg_system-tmp   353G   75M  335G   1% /tmp
        tmpfs                       804M  222k  804M   1% /run/user/3070
        undici:/home-elu            236G  197G   30G  87% /home/elu
        dodici:/home-epilepsy-data  4.4T  1.2T  3.1T  28% /home/epilepsy-data
        dodici:/home-software       216G  180G   27G  88% /home/software

re
https://www.cyberciti.biz/faq/linux-check-disk-space-command/:
    du -s
    elu@oak ~/LU/2_Neural_Network/2_NN_projects_codes$ du -h -s Epilepsy/
    108G    Epilepsy/
    du -a Epilepsy/ | sort -n -r | head -n 20
    du -a ./ | sort -n -r | head -n 20
    
    elu@oak ~/LU/2_Neural_Network$ du -a -H ./ | sort -n -r | head -n 30
    du -a -H  /home/epilepsy-data/data/ | sort -n -r | head -n 30

2019.03.13
open .TRC files:
    download field_trip toolbox
    in matlab:
        addpath(abs path to the package)
        run: ft_defaults
Better one:
    1. Do NOT add fieldtrip to path!
    2. go to C:\Users\gk\Desktop\trc\fieldtrip\fileio\private
    3. load samples as follows:
    data1 = read_micromed_trc('C:\Users\gk\Desktop\trc\EEG_21.TRC', 0, 99);
    data2 = read_micromed_trc('C:\Users\gk\Desktop\trc\EEG_21.TRC', 100, 199);
    etc.
    4. you can merge them as follows if you have enough RAM:
    allData = [data1; data2];
    5. And you can plot them:
    plot(allData')
 
    If you run
    info = read_micromed_trc('C:\Users\gk\Desktop\trc\EEG_21.TRC')
    you will get some info about the file such as the recording date and
the number of samples.

2018.03.18
Keras with tf.dataset:
    https://stackoverflow.com/questions/46135499/how-to-properly-combine-tensorflows-dataset-api-and-keras
save weights in keras and load:
    https://machinelearningmastery.com/save-load-keras-deep-learning-models/

2018.03.19
amazing Keras working with generator. Machine learning framework_fin:
    https://www.pyimagesearch.com/2018/12/10/keras-save-and-load-your-deep-learning-models/

2019.03.25
eager_execution with dataset API:
    https://towardsdatascience.com/eager-execution-tensorflow-8042128ca7be

2019.04.18
change to the epilepsy-data folder:
    elu@oak /home$ cd epilepsy-data

2019.04.23
srun -p sleuths -w jetski --mem=6000 --reservation triesch-shared --gres gpu:rtx2080ti:1 --pty env PYTHONPATH=~software/tensorflow-py3-amd64-gpu python3
scontrol show nodes jetski

2019.04.25
Sorted list:
    sorted_percent = sorted(list(data), key=lambda x: x[1])
    np.savetxt((txts[0][0:-4] + '_sorted.txt'), np.array(sorted_percent), fmt="%s", delimiter=", ")
    # LOad file with str and float
    pd.read_csv(txts[0]).values

2019.04.26
Go to the root folder with GUI:
    Go to computer --> home --> there you are!


when the screen is frozen:
    ctrl + alt + F1:
        1:
            killall -9 python3.5
            ctrl + alt + F7
        2:
            htop
            select: 9: killsignal

2019.05.09
try to pass the number of rows as an element of the dataset and do repeat labels in parsing function, but the tf.tile needs a real value
tensorflow dataset:
    https://zhuanlan.zhihu.com/p/43356309
    
2019.05.16
export edf files to csv:
    1. edf2csv.py, export each hour of recordings given the corresponding machine and channel name and dates
    2. matlab filter the outliers
    3. exclude all the artifacts and resave the file with filename, label and 5 sec-each line

2019.05.20
https://github.com/InFoCusp/tf_cnnvis/blob/master/examples/tf_cnnvis_Example1.ipynb
visualize kernels:
    get all the kernels:
        kernels = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('kernel:0')]
    2. great activation maximization visualization method:
        https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030

2019.05.27
Save multiple variables into txt:
    import pickle
    with open('test_indiv_hour_data_acc.txt', 'wb') as f:
        pickle.dump({"filenames": np.array(filenames),
                     "true_labels": np.array(int_labels),
                     "fns_w_freq": np.array(fns_w_freq),
                     "result_data": result_data}, f)

    file = open('dump.txt', 'r')
    dict = pickle.load(file)

2019.07.02
custom color map:
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","violet","blue"])

2019.08.01
how to download the certificate of enrolment:
    1. login:
        https://qis.server.uni-frankfurt.de/qisserver
        s3722145  LDYdhrz032122$
    2. my functions --> administration of studies --> Study reports for all terms
    
        https://qis.server.uni-frankfurt.de/qisserver/rds?state=change&type=1&moduleParameter=studySOSMenu&nextdir=change&next=menu.vm&subdir=applications&xml=menu&purge=y&navigationPosition=functions%2CstudySOSMenu&breadcrumb=studySOSMenu&topitem=functions&subitem=studySOSMenu

2019.10.07
go to the GUI of squeue:
    sview &

2019.11.26
amazing VAE code github:
    https://github.com/respeecher/vae_workshop

2020.08.30
1. # change the terminal tab title
PROMPT_COMMAND='echo -ne "\033]0; bash-speedboat \007"'
2. # allocate memory for bash running
salloc -p sleuths -w jetski --mem=20000 --reservation triesch-shared --gres gpu:rtx2080ti:1 -t UNLIMITED

2020.08.31
1. sbatch: error: Batch script contains DOS line breaks (\r\n)
   sbatch: error: instead of expected UNIX line breaks (\n).
   dos2unix doesn't exist in linux
   solution: tr -d '\r' < cluster.sh > cluster2.sh

2. scancel batch of jobs:
    scancel{JobID_1..JobID_2}

2020.09.10

To use UMAP you need to install umap-learn not umap.
So, in case you installed umap run the following commands to uninstall umap and install upam-learn instead:
pip uninstall umap
pip install umap-learn
And then in your python code make sure you are importing the module using:
import umap.umap_ as umap
Instead of
import umap

2020.10.01
1. access dict with dot
    from types import SimpleNamespace
                args = SimpleNamespace(**params)
