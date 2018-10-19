## @package dataio
#  This package handles the interface with the hard drive.
#
#  It can in particular read and write matlab or python matrices.
import numpy as np
import tensorflow as tf
import os
import fnmatch


def find_files(directory, pattern='Data*.csv'):
    '''fine all the files in one directory with pattern in the filenames'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

    random.shuffle(files)
    return files


def decode_csv(line):
    '''decode the csv in dataset operation
    '''
    
    defaults = [[0.0]]*301   ## 301 depends on the real shape of the dataset
    csv_row = tf.decode_csv(line, record_defaults=defaults)#
    label = csv_row[0]  # also depend on the final position
    del csv_row[0]   #delete the label
    data = csv_row
    
    return data, label
    
def get_batch(data_dir, batch_size=10):
    '''init a tf.data.Dataset to read the files in a dir, if there are more than one
    param:
        data_dir: string, the dir where the files are located
        batch_size: int'''
    filenames = find_files(data_dir, pattern=pattern)

    dataset = tf.data.Dataset.from_tensor_slices(files)
    
    dataset = dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename).skip(0).map(decode_csv))   ## skip the header, depend on wheather there is any
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).repeat() 

    dataset = dataset.prefetch(2)
    iterator = dataset.make_initializable_iterator()
    batch_data = iterator.get_next()
    
    return batch_data, iterator, num_samples

'''
example:
with tf.name_scope('Data'):
    ### load data to graph
    batch_train, iter_train, num_train = func.get_batch(train_dir, batch_size=batch_size, pattern=pattern)
    with tf.Session() as sess:
        sess.run(iter_train.initializer)
        for batch in range(all_batches):
            batch_data, batch_labels = sess.run(batch_train)
                '''
