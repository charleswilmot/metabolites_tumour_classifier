import time
import os
import numpy as np
import json
import datetime
from dataio import make_output_dir

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


    def __init__(self, json_path):
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
            if not mode:
                self.__dict__.update(dicts)
            elif mode == "train" or mode == "test":
                # general_params = dicts["train_or_test"]["general"]
                general_params = dicts["general"]
                exp_params = dicts[mode]
                self.__dict__.update(general_params)
                self.__dict__.update(exp_params)
            else:
                model_params = dicts["model"][mode]
                self.__dict__.update(model_params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__

# def generate_experiment_path_str(aug_method=None, aug_scale=None, aug_folds=None,
#                                  description=None, input_data=None,
#                                  theta_thr=0.99, rand_seed=129, if_single_runs=False,
#                                  certain_dir=None):
#     date = time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime())
#     # aug_method = default_aug_method if aug_method is None else aug_method
#     # aug_folds = default_folds if aug_folds is None else aug_folds
#     # input_data = default_source if input_data is None else input_data
#     # theta_thr = default_theta if theta_thr is None else theta_thr
#     if input_data is not None:
#         cv_set_id = os.path.basename(input_data).split("_")[-1].split(".")[0]
#     else:
#         cv_set_id = "mnist"
#     print("cluster.py if_single_runs", if_single_runs)
#     if_from_certain = 1 if certain_dir is not None else 0
#     description = description if description else "train"
#     description = "100rns-" + description if if_single_runs else description
#     output_path = os.path.join(EXPERIMENT_DIR_ROOT, default_model_name, "{}-{}-{}x{}-factor-{}-from-{}-certain{}-theta-{}-s{}-{}".format(date, default_model_name, aug_method, aug_folds, aug_scale, cv_set_id, if_from_certain, theta_thr, rand_seed, description))
#     # experiment_dir = EXPERIMENT_DIR_ROOT + "{}_exp0.776_{}x{}_factor_{}_from-epoch_{}_{}".format(date, aug_method, aug_folds, aug_scale, from_epoch, description)
#     print("end of generate_experiment_path_str: cluster.py if_single_runs", if_single_runs)
#     return output_path


def make_output_dir(output_path, sub_folders=["CAMs"]):
    if os.path.isdir(output_path):
        raise FileExistsError("Output path already exists.")
    else:
        os.makedirs(output_path)
        model_save_dir = os.path.join(output_path, "network")
        os.makedirs(model_save_dir)
        for sub in sub_folders:
            os.makedirs(os.path.join(output_path, sub))
        # copy and save all the files
        copy_save_all_files(model_save_dir)


def copy_save_all_files(model_save_dir):
    """
    Copy and save all files related to model directory
    :param model_save_dir:
    :return:
    """
    src_dir = '../src'
    save_dir = os.path.join(model_save_dir, 'src')
    if not os.path.exists(save_dir):  # if subfolder doesn't exist, should make the directory and then save file.
        os.makedirs(save_dir)
    req_extentions = ['py', 'json']
    for filename in os.listdir(src_dir):
        exten = filename.split('.')[-1]
        if exten in req_extentions:
            src_file_name = os.path.join(src_dir, filename)
            target_file_name = os.path.join(save_dir, filename)
            with open(src_file_name, 'r') as file_src:
                with open(target_file_name, 'w') as file_dst:
                    for line in file_src:
                        file_dst.write(line)
    print('Done WithCopy File!')

#
# class ClusterQueue:
#     def __init__(self, **kwargs):
#         # generate a path for the results + mkdir
#         # TODO
#         self.output_path = generate_experiment_path_str(
#             aug_method=kwargs["aug_method"] if "aug_method" in kwargs else None,
#             aug_scale=kwargs["aug_scale"] if "aug_scale" in kwargs else None,
#             aug_folds=kwargs["aug_folds"] if "aug_folds" in kwargs else None,
#             input_data=kwargs["input_data"] if "input_data" in kwargs else None,
#             theta_thr=kwargs["theta_thr"] if "theta_thr" in kwargs else None,
#             rand_seed=kwargs["randseed"] if "randseed" in kwargs else 129,
#             if_single_runs=kwargs["if_single_runs"] if "if_single_runs" in kwargs else False,
#             certain_dir=kwargs["certain_dir"] if "certain_dir" in kwargs else None,
#             description=kwargs["description"] if "description" in kwargs else None)
#
#         make_output_dir(self.output_path, sub_folders=["AUCs", "CAMs", 'CAMs/mean', "wrong_examples", "certains"])
#

def overwrite_params(args, **kwargs):
    """
    overwrite some parameters for SLURM jobs
    :param args:
    :return:
    """
    time_str = '{0:%Y-%m-%dT%H-%M-%S-}'.format(datetime.datetime.now())

    if args.data_mode == "mnist" or args.data_mode == "MNIST":
        args.num_classes = 10
        args.width = 28
        args.height = 28
        args.data_source = "mnist"
    elif args.data_mode == "metabolite" or args.data_mode == "metabolites":
        args.num_classes = 2
        args.width = 1
        args.height = 288
        args.data_source = os.path.basename(args.input_data).split("_")[-1].split(".")[0]
        # TODO, cluster and param.json all give this parameter

    args.aug_method = kwargs["aug_method"] if "aug_method" in kwargs else "none"
    args.aug_scale = kwargs["aug_scale"] if "aug_scale" in kwargs else 0
    args.aug_folds = kwargs["aug_folds"] if "aug_folds" in kwargs else 0
    args.input_data = kwargs["input_data"] if "input_data" in kwargs else None
    args.theta_thr = kwargs["theta_thr"] if "theta_thr" in kwargs else None
    args.rand_seed = kwargs["randseed"] if "randseed" in kwargs else 129
    args.if_single_runs = kwargs["if_single_runs"] if "if_single_runs" in kwargs else False
    args.certain_dir = kwargs["certain_dir"] if "certain_dir" in kwargs else None
    new_folder = kwargs["new_folder"]+"-{}".format(args.model_name) if "new_folder" in kwargs else None

    if new_folder: #is not None
        args.output_path = os.path.join(args.output_root, new_folder)

    print("Not run from cluster.py params.input data dir: ", args.input_data)
    if args.restore_from is None:  # and args.output_path is None:  #cluster.py
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
        args.train_or_test = "test"
        args.output_path = os.path.dirname(args.restore_from) + "-on-{}-{}".format(args.data_source, "test")
        args.if_from_certain = False
        args.if_save_certain = False
        args.postfix = "-test"
        args.test_ratio = 1




    # params.resplit_data = args.resplit_data
    # params.restore_from = args.restore_from
    # params.train_or_test = args.train_or_test
    # params.resume_training = (params.restore_from != None)
    # params.if_single_runs = False
    # print("argument.py, params.if_single_runs: ", params.if_single_runs)
    args.model_save_dir = os.path.join(args.output_path, "network")
    print("output dir: ", args.output_path)

    make_output_dir(args.output_path, sub_folders=["AUCs", "CAMs", 'CAMs/mean', "wrong_examples", "certains"])

    args.save(os.path.join(args.output_path, "network", "parameters.json"))


if __name__ == "__main__":

    default_train_or_test = "train"
    ## Load experiment parameters and model parameters
    default_exp_json_dir = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/src/exp_parameters.json"
    default_model_json_dir = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/src/model_parameters.json"
    #load default
    assert os.path.isfile(default_exp_json_dir), "No json configuration file found at {}".format(default_exp_json_dir)
    args = Params(default_exp_json_dir)
    args.update(default_exp_json_dir, mode=default_train_or_test)

    # load model specific parameters
    assert os.path.isfile(default_model_json_dir), "No json file found at {}".format(default_model_json_dir)
    args.update(default_model_json_dir, mode=args.model_name)

    data_source_dirs = [
        # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data0.mat",
        # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data9.mat",
        # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data2.mat",
        # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data7.mat",
        # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data4.mat",
        "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data5.mat",
        # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data6.mat",
        "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data3.mat",
        # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data8.mat",
        "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data1.mat"
    ]

    # overwrite part of the parameters given for training with cluster.py
    config_files = []
    for dd in data_source_dirs:
        for method in ["same_mean", "ops_mean","both_mean"]:  #
            for fold in [1, 3, 5, 9]:  #, 3, 5, 7, 9
                for scale in [0.05, 0.2, 0.35, 0.5]:  #, 0.3, 0.5
                    overwrite_params(args,
                                     input_data=dd, #data dir
                                     certain_dir=None,
                                     aug_method=method,
                                     aug_scale=scale,
                                     aug_folds=fold,
                                     theta_thr=1,
                                     randseed=45,
                                     if_single_runs=False,
                                     from_clusterpy=True,
                                     new_folder="randomDA")

