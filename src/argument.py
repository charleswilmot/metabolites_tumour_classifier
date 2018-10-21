## @package argument
#  This package processes the argument passed to the main program.
import argparse
import configreader


def percent(s):
    f = float(s)
    if f < 0:
        return 0
    if f > 100:
        return 100
    return f


def positive(s):
    f = float(s)
    if f <= 0:
        return 0.1
    return f


parser = argparse.ArgumentParser()

parser.add_argument(
    '-f', '--output-matrix-fileformat', type=str, action='store', default='npz', choices=['npz', 'm', 'both'],
    help="Desiered output file format, one of 'npz', 'm' or 'both'."
    )

parser.add_argument(
    '-b', '--maximum-batch-size', type=int, action='store', default=2048,
    help="Number of datapoints to be processed at once during training and testing."
    )

parser.add_argument(
    '-n', '--network-conf',
    type=configreader.Configurator,
    default=configreader.Configurator("../conf/default_network_do_not_change.conf"),
    help="Path to the network configuration file."
    )

parser.add_argument(
    '-s', '--seed', type=int, action='store', default=-1,
    help="Random seed initialization."
    )

parser.add_argument(
    '-v', '--verbose', action='count', default=1,
    help="Verbosity level. Use -v, -vv, -vvv."
    )
subparsers = parser.add_subparsers(dest="test_or_train")
test_parser = subparsers.add_parser('test')
test_parser.add_argument(
    'model_path', metavar='MODEL',  type=str,
    help="Path to a previously trained model."
    )

test_parser.add_argument(
    'input_data', metavar='INPUT', type=str,
    help="Path to the input data file."
    )

test_parser.add_argument(
    'output_path', metavar='OUTPUT', type=str, nargs='?', default='./',
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

train_parser.add_argument(
    '-S', '--train-test-split-ratio', type=percent, action='store', default=10.0, metavar='RATIO',
    help="Determines which percentage of the training data should be used for testing. It can be set to 0 percent, in which case the software does not produce any plot of the training."
    )

train_parser.add_argument(
    '-l', '--learning-rate', type=float, action='store', default=1e-3,
    help="Set the learning rate."
    )

train_parser.add_argument(
    '-L', '--loss-type', type=str, action='store', default='mse', choices=['mse', 'rmse', 'sigmoid_ce'],
    help="Choses a loss type. Can be any of 'mse', 'rmse', 'sigmoid_ce'"
    )
# todo custom function once more

train_parser.add_argument(
    '-e', '--number-of-epochs', type=positive, action='store', default='-1', metavar='N',
    help="Determines the amount of times the algorithm will see each of the training examples. If this flag is not set, the algorithm trains until convergence."
    )

train_parser.add_argument(
    '-c', '--train-test-compute-time-ratio', type=percent, action='store', default=50, metavar='RATIO',
    help="Determines which amount of time should be spent training the network, and which amount of time should be spent testing. The more time is spent testing, the better the plot will look."
    )

train_parser.add_argument(
    '-r', '--resume-training', type=str, action='store', default="", metavar='PATH',
    help="Path to a pretrained model. Resumes the training from the last checkpoint."
    )

args = parser.parse_args()
