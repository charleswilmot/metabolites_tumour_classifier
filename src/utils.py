"""General utility functions"""

import json
import os
import logging
import datetime


# class Params():
#     """Class that loads hyperparameters from a json file.
#
#     Example:
#     ```
#     params = Params(json_path)
#     print(params.learning_rate)
#     params.learning_rate = 0.5  # change the value of learning_rate in params
#     ```
#     """
#
#     def __init__(self, json_path):
#         # type: (object) -> object
#         self.update(json_path)
#
#     def save(self, json_path):
#         """Saves parameters to json file"""
#         with open(json_path, 'w') as f:
#             json.dump(self.__dict__, f, indent=4)
#
#     def update(self, json_path, model_key=None):
#         """Loads parameters from json file. if specify a modelkey, only load the params under thta modelkey"""
#         with open(json_path) as f:
#             params = json.load(f)
#             if not model_key:
#                 self.__dict__.update(params)
#             else:
#                 model_params = params["model"][model_key]
#                 self.__dict__.update(model_params)
#
#     @property
#     def dict(self):
#         """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
#         return self.__dict__

class Params():
    """Class that loads hyperparameters from a json file.
    https://github.com/cs230-stanford/cs230-code-examples/blob/master/tensorflow/vision/model/utils.py
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """
    def __init__(self):
        # type: (object) -> object
        # self.update(json_path)
        pass
    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path, mode=None):
        """Loads parameters from json file. if specify a modelkey, only load the params under thta modelkey"""
        with open(json_path) as f:
            dicts = json.load(f)

            if mode == "train_or_test":
                self.__dict__.update(dicts)
                self.train_or_test = dicts["train_or_test"]
                # general_params = dicts["train_or_test"]["general"]
                # general_params = dicts["general"]
                exp_params = dicts[self.train_or_test]
                # self.__dict__.update(general_params)
                self.__dict__.update(exp_params)
            else:
                model_params = dicts["model"][mode]
                self.__dict__.update(model_params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__

def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v[0]) for k, v in d.items()}
        json.dump(d, f, indent=4)


def load_all_params(exp_json_dir, model_json_dir):
    """
    Load exp. params and model params given the dirs
    :param args:
    :return:
    """
    ## Load experiment parameters and model parameters
    assert os.path.isfile(exp_json_dir), "No json configuration file found at {}".format(exp_json_dir)
    args = Params()
    args.update(exp_json_dir, mode="train_or_test")

    # load model specific parameters
    assert os.path.isfile(model_json_dir), "No json file found at {}".format(model_json_dir)
    args.update(model_json_dir, mode=args.model_name)

    return args


def generate_output_path(args):
    if args.data_mode == "mnist" or args.data_mode == "MNIST":
        args.num_classes = 10
        args.width = 28
        args.height = 28
        args.data_source = "mnist"
        args.data_dim = "2d"
    elif args.data_mode == "metabolite" or args.data_mode == "metabolites":
        args.num_classes = 2
        args.width = 1
        args.height = 288
        args.data_source = os.path.basename(args.input_data).split("_")[-1].split(".")[0]
        args.data_dim = "1d"
        # TODO, cluster and param.json all give this parameter

    if args.certain_dir is not None:
        args.if_from_certain = True
    args.if_save_certain = True if args.if_single_runs else False

    if args.new_folder: #is not None
        args.output_path = os.path.join(args.output_root, args.new_folder+"-{}".format(args.model_name))
    if args.restore_from is None:  # and args.output_path is None:  #cluster.py
        time_str = '{0:%Y-%m-%dT%H-%M-%S-}'.format(datetime.datetime.now())
        args.postfix = "100rns-" + args.train_or_test if args.if_single_runs else args.train_or_test
        args.output_path = os.path.join(args.output_path,
                                        "{}-{}-{}x{}-factor-{}-from-{}-certain{}-theta-{}-s{}-{}".format(
                                            time_str, args.model_name, args.aug_method, args.aug_folds,
                                            args.aug_scale, args.data_source,
                                            args.if_from_certain, args.theta_thr,
                                            args.randseed, args.postfix))
    elif args.restore_from is not None and args.resume_training:  # restore a model
        args.train_or_test = "train"
        args.output_path = os.path.dirname(args.restore_from) + "-on-{}-{}".format(args.data_source, "resume_train")
        args.postfix = "-resume_train"
    elif args.restore_from is not None and not args.resume_training:
        time_str = '{0:%Y%m%dT%H%M%S}'.format(datetime.datetime.now())
        args.train_or_test = "test"
        args.output_path = os.path.dirname(args.restore_from) + "-{}-on-{}-{}".format(time_str, args.data_source, "test")
        args.if_from_certain = False
        args.if_save_certain = False
        args.postfix = "-test"
        args.test_ratio = 1
    args.model_save_dir = os.path.join(args.output_path, "network")
    print("output dir: ", args.output_path)

    return args

