## @package argument
#  This package processes the argument passed to the main program.
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    '-f', '--output-matrix-fileformat', type=str, nargs=1, action='store', default='npz', choices=['npz', 'm', 'both'],
    help="Desiered output file format, one of 'npz', 'm' or 'both'."
    )

parser.add_argument(
    '-b', '--maximum-batch-size', type=int, nargs=1, action='store', default=2048,
    help="Number of datapoints to be processed at once during training and testing."
    )
# todo: replace open with a custom function that reads the file and closes it...

parser.add_argument(
    '-n', '--network-conf', type=str, nargs=1, default="../conf/default_network_do_not_change.conf",
    help="Path to the network configuration file."
    )

parser.add_argument(
    '-s', '--seed', type=int, nargs=1, action='store', default=-1,
    help="Random seed initialization."
    )

parser.add_argument(
    '-v', action='count',
    help="Verbosity level. Use -v, -vv, -vvv."
    )
subparsers = parser.add_subparsers(dest="test_or_train") # this line changed
test_parser = subparsers.add_parser('test')
test_parser.add_argument(
    'model path', metavar='MODEL',  type=str, nargs=1,
    help="Path to a previously trained model."
    )

test_parser.add_argument(
    'input data', metavar='INPUT', type=str, nargs=1,
    help="Path to the input data file."
    )

test_parser.add_argument(
    'output path', metavar='OUTPUT', type=str, nargs='?', default='./',
    help="Path to the output data."
    )

test_parser.add_argument(
    '-x', '--data-not-labeled', action='store_true',
    help="Set this flag if the data you want to classify is not labeled."
    )
train_parser = subparsers.add_parser('train')

train_parser.add_argument(
    '-a', '--data-augmentation', action='store_true',
    help="Set this flag if the algorithm should perform data augmentation."
    )
# todo: replace with a custom function that checks 0 < x < 100

train_parser.add_argument(
    '-S', '--train-test-split-ratio', type=float, action='store', nargs=1, default=10, metavar='RATIO',
    help="Determines which percentage of the training data should be used for testing. It can be set to 0 percent, in which case the software does not produce any plot of the training."
    )

train_parser.add_argument(
    '-l', '--learning-rate', type=float, action='store', nargs=1, default=1e-3,
    help="Set the learning rate."
    )

train_parser.add_argument(
    '-L', '--loss-type', type=str, nargs=1, action='store', default='mse', choices=['mse', 'rmse', 'sigmoid_ce'],
    help="Choses a loss type. Can be any of 'mse', 'rmse', 'sigmoid_ce'"
    )
# todo custom function once more

train_parser.add_argument(
    '-e', '--number-of-epochs', type=float, action='store', nargs=1, default='-1', metavar='N',
    help="Determines the amount of times the algorithm will see each of the training examples. If this flag is not set, the algorithm trains until convergence."
    )

train_parser.add_argument(
    '-c', '--train-test-compute-time-ratio', type=float, action='store', nargs=1, default=10, metavar='RATIO',
    help="Determines which amount of time should be spent training the network, and which amount of time should be spent testing. The more time is spent testing, the better the plot will look."
    )

train_parser.add_argument(
    '-r', '--resume-training', type=str, action='store', nargs=1, default="", metavar='PATH',
    help="Path to a pretrained model. Resumes the training from the last checkpoint."
    )

args = parser.parse_args()
