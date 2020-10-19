import time
import os
import utils
import json
from dataio import make_output_dir, save_command_line
import shutil
import numpy as np


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
            self.__dict__.update(dicts)

            if mode == "train" or mode == "test":
                # general_params = dicts["train_or_test"]["general"]
                # general_params = dicts["general"]
                exp_params = dicts[mode]
                # self.__dict__.update(general_params)
                self.__dict__.update(exp_params)
            else:
                model_params = dicts["model"][mode]
                self.__dict__.update(model_params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__

def save_model_json(dest="./", source="./", file_name="model_parameters.json"):
    s = os.path.join(source, file_name)
    d = os.path.join(dest, file_name)
    shutil.copy2(s, d)
    return d


def overwrite_params(args, cfg_dirs, **kwargs):
    """
    overwrite some parameters for SLURM jobs
    :param args:
    :return:
    """
    args.aug_method = kwargs["aug_method"] if "aug_method" in kwargs else "none"
    args.aug_scale = kwargs["aug_scale"] if "aug_scale" in kwargs else 0
    args.aug_folds = kwargs["aug_folds"] if "aug_folds" in kwargs else 0
    args.input_data = kwargs["input_data"] if "input_data" in kwargs else None
    args.theta_thr = kwargs["theta_thr"] if "theta_thr" in kwargs else None
    args.rand_seed = kwargs["randseed"] if "randseed" in kwargs else 129
    args.if_single_runs = kwargs["if_single_runs"] if "if_single_runs" in kwargs else False
    args.if_save_certain = True if args.if_single_runs else False
    args.restore_from = kwargs["restore_from"] if "restore_from" in kwargs else None
    args.certain_dir = kwargs["certain_dir"] if "certain_dir" in kwargs else None
    args.from_clusterpy = kwargs["from_clusterpy"] if "from_clusterpy" in kwargs else False
    if args.certain_dir is not None:
        args.if_from_certain = True

    # GET output_path
    args = utils.generate_output_path(args)

    assert os.path.isfile(default_exp_json_dir), "No json configuration file found at {}".format(default_exp_json_dir)
    new_exp_json_fn = os.path.join(args.output_path, "network", "exp_parameters.json")

    # make dirs
    make_output_dir(args, sub_folders=["AUCs", "CAMs", 'CAMs/mean', "wrong_examples", "certains"])
    save_command_line(args.model_save_dir)

    new_model_son_fn = save_model_json(dest=os.path.join(args.output_path, "network"), source=default_json_dir, file_name="model_parameters.json")
    args.save(os.path.join(args.output_path, "network", "exp_parameters.json"))
    time.sleep(1)

    cfg_dirs.append([args.output_path, new_exp_json_fn, new_model_son_fn])
    return cfg_dirs

######################################################################################################################
default_json_dir = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/src"
default_train_or_test = "train"
## Load experiment parameters and model parameters
default_exp_json_dir = os.path.join(default_json_dir, "exp_parameters.json")
default_model_json_dir = os.path.join(default_json_dir, "model_parameters.json")

args = utils.load_all_params(default_exp_json_dir, default_model_json_dir)

data_source_dirs = [
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data5.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data3.mat",
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data1.mat"
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data9.mat",
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data7.mat",
]
certain_dirs = [
    "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-MLP/2020-10-14T14-29-17--MLP-nonex0-factor-0-from-data5-certainFalse-theta-0-s989-100rns-train",
    "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-MLP/2020-10-14T14-29-18--MLP-nonex0-factor-0-from-data3-certainFalse-theta-0-s989-100rns-train",
    # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-MLP/2020-10-14T14-29-19--MLP-nonex0-factor-0-from-data1-certainFalse-theta-0-s989-100rns-train/network"
    # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-MLP/2020-10-14T14-30-52--MLP-nonex0-factor-0-from-data9-certainFalse-theta-0-s989-100rns-train",
    # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-MLP/2020-10-14T14-30-53--MLP-nonex0-factor-0-from-data7-certainFalse-theta-0-s989-100rns-train"
]
# overwrite part of the parameters given for training with cluster.py


config_dirs = []
seed = np.random.randint(9999)
mode = "training"
if mode == "single_runs":
    ## 100 single-epoch runs
    args.new_folder = "testtesttest"
    for dd in data_source_dirs:
        config_dirs = overwrite_params(args, config_dirs,
                                       input_data=dd,  # data dir
                                       certain_dir=None,
                                       aug_method="none",
                                       aug_scale=0,
                                       aug_folds=0,
                                       theta_thr=0,
                                       randseed=seed,
                                       if_single_runs=True,
                                       from_clusterpy=True
                                       )
elif mode == "training":
    experiment = 0
    for theta in [0.75]: #, 0.5, 0.1, 0.3
        for dd, ct_dir in zip(data_source_dirs, certain_dirs):   #
            for method in ["same_mean"]:  #,"both_mean" , "ops_mean",  #
                for fold in [3]:  #1, 3, 5,  9, 5, 9#
                    for scale in [0.2]:  #, 0.05, 0.35, 0.5, ,#
                        # ct_dir = None
                        if ct_dir is not None:
                            theta = theta
                            args.new_folder = "certain-DA-Res7"
                        else:
                            theta = 1
                            args.new_folder = "randomDA"

                        config_dirs = overwrite_params(args, config_dirs,
                                                       input_data=dd,  #data dir
                                                       certain_dir=ct_dir,
                                                       aug_method=method,
                                                       aug_scale=scale,
                                                       aug_folds=fold,
                                                       theta_thr=theta,
                                                       randseed=seed,
                                                       if_single_runs=False,
                                                       from_clusterpy=True)
elif mode == "testing":
    pretrained_dirs = [
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-43-39-randomDA-MLP-same_meanx1-factor-0.05-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-43-40-randomDA-MLP-same_meanx1-factor-0.2-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-43-41-randomDA-MLP-same_meanx1-factor-0.35-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-43-42-randomDA-MLP-same_meanx1-factor-0.5-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-43-43-randomDA-MLP-same_meanx3-factor-0.05-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-43-44-randomDA-MLP-same_meanx3-factor-0.2-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-43-45-randomDA-MLP-same_meanx3-factor-0.35-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-43-47-randomDA-MLP-same_meanx3-factor-0.5-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-43-48-randomDA-MLP-same_meanx5-factor-0.05-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-43-49-randomDA-MLP-same_meanx5-factor-0.2-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-43-50-randomDA-MLP-same_meanx5-factor-0.35-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-43-51-randomDA-MLP-same_meanx5-factor-0.5-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-43-52-randomDA-MLP-same_meanx9-factor-0.05-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-43-53-randomDA-MLP-same_meanx9-factor-0.2-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-43-54-randomDA-MLP-same_meanx9-factor-0.35-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-43-55-randomDA-MLP-same_meanx9-factor-0.5-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-43-57-randomDA-MLP-ops_meanx1-factor-0.05-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-43-58-randomDA-MLP-ops_meanx1-factor-0.2-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-43-59-randomDA-MLP-ops_meanx1-factor-0.35-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-00-randomDA-MLP-ops_meanx1-factor-0.5-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-01-randomDA-MLP-ops_meanx3-factor-0.05-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-02-randomDA-MLP-ops_meanx3-factor-0.2-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-03-randomDA-MLP-ops_meanx3-factor-0.35-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-04-randomDA-MLP-ops_meanx3-factor-0.5-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-05-randomDA-MLP-ops_meanx5-factor-0.05-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-07-randomDA-MLP-ops_meanx5-factor-0.2-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-08-randomDA-MLP-ops_meanx5-factor-0.35-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-09-randomDA-MLP-ops_meanx5-factor-0.5-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-10-randomDA-MLP-ops_meanx9-factor-0.05-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-11-randomDA-MLP-ops_meanx9-factor-0.2-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-12-randomDA-MLP-ops_meanx9-factor-0.35-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-13-randomDA-MLP-ops_meanx9-factor-0.5-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-14-randomDA-MLP-both_meanx1-factor-0.05-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-15-randomDA-MLP-both_meanx1-factor-0.2-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-17-randomDA-MLP-both_meanx1-factor-0.35-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-18-randomDA-MLP-both_meanx1-factor-0.5-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-19-randomDA-MLP-both_meanx3-factor-0.05-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-20-randomDA-MLP-both_meanx3-factor-0.2-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-21-randomDA-MLP-both_meanx3-factor-0.35-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-22-randomDA-MLP-both_meanx3-factor-0.5-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-23-randomDA-MLP-both_meanx5-factor-0.05-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-24-randomDA-MLP-both_meanx5-factor-0.2-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-25-randomDA-MLP-both_meanx5-factor-0.35-from-data5-certain0-theta-1-s129-train/network",
       # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-26-randomDA-MLP-both_meanx5-factor-0.5-from-data5-certain0-theta-1-s129-train/network",
       "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-18-randomDA-MLP-both_meanx1-factor-0.5-from-data5-certain0-theta-1-s129-train/network",
       "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-17-randomDA-MLP-both_meanx1-factor-0.35-from-data5-certain0-theta-1-s129-train/network",
       "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-15-randomDA-MLP-both_meanx1-factor-0.2-from-data5-certain0-theta-1-s129-train/network",
       "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/randomDA-MLP2/2020-10-12T23-44-14-randomDA-MLP-both_meanx1-factor-0.05-from-data5-certain0-theta-1-s129-train/network"
    ]
    src_data = os.path.basename(os.path.dirname(pretrained_dirs[0])).split("-")[-6]
    data_source_dirs = [
        "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_val_{}.mat".format(src_data)
    ]
    for dd, res_from in zip(data_source_dirs * len(pretrained_dirs), pretrained_dirs):  #
        config_dirs = overwrite_params(args, config_dirs,
                                       input_data=dd,  # data dir
                                       certain_dir=None,
                                       randseed=seed,
                                       restore_from=res_from,
                                       if_single_runs=False,
                                       from_clusterpy=True)

print("ok")
