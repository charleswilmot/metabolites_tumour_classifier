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
    if args.distill_old:  #only exit in the old method
        args.from_epoch = kwargs["from_epoch"] if "from_epoch" in kwargs else 0
    args.input_data = kwargs["input_data"] if "input_data" in kwargs else None
    args.theta_thr = kwargs["theta_thr"] if "theta_thr" in kwargs else None
    args.rand_seed = kwargs["rand_seed"] if "rand_seed" in kwargs else 129
    args.if_single_runs = kwargs["if_single_runs"] if "if_single_runs" in kwargs else False
    # args.if_save_certain = True if args.if_single_runs else False
    args.restore_from = kwargs["restore_from"] if "restore_from" in kwargs else None
    args.certain_dir = kwargs["certain_dir"] if "certain_dir" in kwargs else None
    args.from_clusterpy = kwargs["from_clusterpy"] if "from_clusterpy" in kwargs else False
    if args.certain_dir is not None:
        args.if_from_certain = True
    else:
        args.if_from_certain = False

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
default_json_dir = "./"
default_output_root= "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review"
# default_json_dir = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/src"
default_train_or_test = "train"
## Load experiment parameters and model parameters
default_exp_json_dir = os.path.join(default_json_dir, "exp_parameters.json")
default_model_json_dir = os.path.join(default_json_dir, "model_parameters.json")

args = utils.load_all_params(default_exp_json_dir, default_model_json_dir)

data_source_dirs = [
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data5.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data9.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data8.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data7.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data4.mat",
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data3.mat",
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data6.mat",
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data2.mat",
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data1.mat",
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data0.mat",
]

# overwrite part of the parameters given for training with cluster.py


config_dirs = []
seed = np.random.randint(9999)
mode = "aug_training"


if mode == "single_runs":
    ## 100 single-epoch runs
    args.new_folder = "100-single-epoch-runs-5-fold-data"
    for dd in data_source_dirs:
        config_dirs = overwrite_params(args, config_dirs,
                                       input_data=dd,  # data dir
                                       certain_dir=None,
                                       aug_method="none",
                                       aug_scale=0,
                                       aug_folds=0,
                                       theta_thr=0,
                                       rand_seed=989,
                                       if_single_runs=True,
                                       from_clusterpy=True
                                       )
elif mode == "aug_training":
    aug_method = "random"
    if aug_method == "random":
        certain_dirs = [None] * len(data_source_dirs)
        theta = 1
        args.new_folder = "2-randomDA"
    elif aug_method == "certain":  #1,2,3,5,7,9
        certain_dirs = [
            # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T22-07-37--Res_ECG_CAM-nonex0-factor-0-from-data5-certainFalse-theta-0-s989-100rns-train",
            # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T22-07-38--Res_ECG_CAM-nonex0-factor-0-from-data3-certainFalse-theta-0-s989-100rns-train"
            "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T22-07-39--Res_ECG_CAM-nonex0-factor-0-from-data1-certainFalse-theta-0-s989-100rns-train",
            # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T23-08-12--Res_ECG_CAM-nonex0-factor-0-from-data9-certainFalse-theta-0-s989-100rns-train",
            # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T23-08-13--Res_ECG_CAM-nonex0-factor-0-from-data7-certainFalse-theta-0-s989-100rns-train",
            ## save for later
            # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-12-18T13-50-31--Res_ECG_CAM-nonex0-factor-0-from-data6-certainFalse-theta-0-s899-100rns-train",
            # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-12-18T13-50-32--Res_ECG_CAM-nonex0-factor-0-from-data8-certainFalse-theta-0-s899-100rns-train",
            # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-12-18T13-51-29--Res_ECG_CAM-nonex0-factor-0-from-data2-certainFalse-theta-0-s899-100rns-train",
            # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-12-18T13-51-30--Res_ECG_CAM-nonex0-factor-0-from-data4-certainFalse-theta-0-s899-100rns-train",
            ]
        src_data = [os.path.basename(certain).split("-")[-7] for certain in certain_dirs]
        data_source_dirs = [
            "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_{}.mat".format(
                src_dir) for src_dir in src_data]
        theta = 0.5
        args.new_folder = "3-certain-DA"
    for dd, certain in zip(data_source_dirs, certain_dirs):   #, certain_dirs)
        for method in ["same_mean"]:  #, "both_mean", "ops_mean", "noise" #
            for fold in [9]:  #, 5, 3, 1 , 7, 10, 9,  #
                for scale in [0.5]:  #, 0.35, 0.2, 0.05  3, 0.5#
                    config_dirs = overwrite_params(args, config_dirs,
                                                   input_data=dd,  #data dir
                                                   certain_dir=certain,
                                                   aug_method=method,
                                                   aug_scale=scale,
                                                   aug_folds=fold,
                                                   theta_thr=theta,
                                                   rand_seed=seed,
                                                   if_single_runs=False,
                                                   from_clusterpy=True)
elif mode == "pure_training":  # without DA, without distillation
    args.if_save_certain = False
    for dd in data_source_dirs:   #
        config_dirs = overwrite_params(args, config_dirs,
                                       input_data=dd,  #data dir
                                       certain_dir=None,
                                       aug_method=None,
                                       aug_scale=0,
                                       aug_folds=0,
                                       theta_thr=1,
                                       rand_seed=seed,
                                       if_single_runs=False,
                                       from_clusterpy=True)
        
        
        
elif mode == "testing":
    pretrained_dirs = [
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2-randomDA-MLP-data5-len24/2021-01-08T14-55-12--MLP-same_meanx9-factor-0.5-from-data5-certainFalse-theta-1-s9006-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2-randomDA-MLP-data5-len24/2021-01-08T14-55-14--MLP-same_meanx9-factor-0.35-from-data5-certainFalse-theta-1-s9006-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2-randomDA-MLP-data5-len24/2021-01-08T14-55-15--MLP-same_meanx5-factor-0.5-from-data5-certainFalse-theta-1-s9006-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2-randomDA-MLP-data5-len24/2021-01-08T14-55-16--MLP-same_meanx5-factor-0.35-from-data5-certainFalse-theta-1-s9006-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2-randomDA-MLP-data5-len24/2021-01-08T14-55-17--MLP-same_meanx3-factor-0.5-from-data5-certainFalse-theta-1-s9006-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2-randomDA-MLP-data5-len24/2021-01-08T14-55-18--MLP-same_meanx3-factor-0.35-from-data5-certainFalse-theta-1-s9006-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2-randomDA-MLP-data5-len24/2021-01-08T14-55-19--MLP-same_meanx1-factor-0.5-from-data5-certainFalse-theta-1-s9006-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2-randomDA-MLP-data5-len24/2021-01-08T14-55-21--MLP-same_meanx1-factor-0.35-from-data5-certainFalse-theta-1-s9006-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2-randomDA-MLP-data5-len24/2021-01-08T14-55-22--MLP-both_meanx9-factor-0.5-from-data5-certainFalse-theta-1-s9006-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2-randomDA-MLP-data5-len24/2021-01-08T14-55-23--MLP-both_meanx9-factor-0.35-from-data5-certainFalse-theta-1-s9006-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2-randomDA-MLP-data5-len24/2021-01-08T14-55-24--MLP-both_meanx5-factor-0.5-from-data5-certainFalse-theta-1-s9006-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2-randomDA-MLP-data5-len24/2021-01-08T14-55-25--MLP-both_meanx5-factor-0.35-from-data5-certainFalse-theta-1-s9006-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2-randomDA-MLP-data5-len24/2021-01-08T14-55-26--MLP-both_meanx3-factor-0.5-from-data5-certainFalse-theta-1-s9006-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2-randomDA-MLP-data5-len24/2021-01-08T14-55-27--MLP-both_meanx3-factor-0.35-from-data5-certainFalse-theta-1-s9006-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2-randomDA-MLP-data5-len24/2021-01-08T14-55-29--MLP-both_meanx1-factor-0.5-from-data5-certainFalse-theta-1-s9006-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2-randomDA-MLP-data5-len24/2021-01-08T14-55-30--MLP-both_meanx1-factor-0.35-from-data5-certainFalse-theta-1-s9006-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-53-13--RNN-both_meanx9-factor-0.05-from-data5-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-53-14--RNN-both_meanx9-factor-0.2-from-data5-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-53-15--RNN-both_meanx9-factor-0.35-from-data5-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-53-16--RNN-both_meanx9-factor-0.5-from-data5-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-17--RNN-same_meanx1-factor-0.2-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-18--RNN-same_meanx1-factor-0.35-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-19--RNN-same_meanx1-factor-0.5-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-20--RNN-same_meanx3-factor-0.05-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-21--RNN-same_meanx3-factor-0.2-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-23--RNN-same_meanx3-factor-0.35-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-24--RNN-same_meanx3-factor-0.5-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-25--RNN-same_meanx5-factor-0.05-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-26--RNN-same_meanx5-factor-0.2-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-27--RNN-same_meanx5-factor-0.35-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-28--RNN-same_meanx5-factor-0.5-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-29--RNN-same_meanx9-factor-0.05-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-31--RNN-same_meanx9-factor-0.2-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-32--RNN-same_meanx9-factor-0.35-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-33--RNN-same_meanx9-factor-0.5-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-34--RNN-both_meanx1-factor-0.05-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-35--RNN-both_meanx1-factor-0.2-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-36--RNN-both_meanx1-factor-0.35-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-38--RNN-both_meanx1-factor-0.5-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-39--RNN-both_meanx3-factor-0.05-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-40--RNN-both_meanx3-factor-0.2-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-41--RNN-both_meanx3-factor-0.35-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-42--RNN-both_meanx3-factor-0.5-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-43--RNN-both_meanx5-factor-0.05-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-44--RNN-both_meanx5-factor-0.2-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-46--RNN-both_meanx5-factor-0.35-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-47--RNN-both_meanx5-factor-0.5-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-48--RNN-both_meanx9-factor-0.05-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-49--RNN-both_meanx9-factor-0.2-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-50--RNN-both_meanx9-factor-0.35-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-51--RNN-both_meanx9-factor-0.5-from-data1-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-52--RNN-same_meanx1-factor-0.05-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-54--RNN-same_meanx1-factor-0.2-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-55--RNN-same_meanx1-factor-0.35-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-56--RNN-same_meanx1-factor-0.5-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-57--RNN-same_meanx3-factor-0.05-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-58--RNN-same_meanx3-factor-0.2-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-50-59--RNN-same_meanx3-factor-0.35-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-00--RNN-same_meanx3-factor-0.5-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-01--RNN-same_meanx5-factor-0.05-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-03--RNN-same_meanx5-factor-0.2-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-04--RNN-same_meanx5-factor-0.35-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-05--RNN-same_meanx5-factor-0.5-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-06--RNN-same_meanx9-factor-0.05-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-07--RNN-same_meanx9-factor-0.2-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-08--RNN-same_meanx9-factor-0.35-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-09--RNN-same_meanx9-factor-0.5-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-10--RNN-both_meanx1-factor-0.05-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-12--RNN-both_meanx1-factor-0.2-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-13--RNN-both_meanx1-factor-0.35-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-14--RNN-both_meanx1-factor-0.5-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-15--RNN-both_meanx3-factor-0.05-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-16--RNN-both_meanx3-factor-0.2-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-17--RNN-both_meanx3-factor-0.35-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-18--RNN-both_meanx3-factor-0.5-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-19--RNN-both_meanx5-factor-0.05-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-21--RNN-both_meanx5-factor-0.2-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-22--RNN-both_meanx5-factor-0.35-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-23--RNN-both_meanx5-factor-0.5-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-24--RNN-both_meanx9-factor-0.05-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-25--RNN-both_meanx9-factor-0.2-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-26--RNN-both_meanx9-factor-0.35-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-27--RNN-both_meanx9-factor-0.5-from-data9-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-29--RNN-same_meanx1-factor-0.05-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-30--RNN-same_meanx1-factor-0.2-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-31--RNN-same_meanx1-factor-0.35-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-32--RNN-same_meanx1-factor-0.5-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-33--RNN-same_meanx3-factor-0.05-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-34--RNN-same_meanx3-factor-0.2-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-35--RNN-same_meanx3-factor-0.35-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-36--RNN-same_meanx3-factor-0.5-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-38--RNN-same_meanx5-factor-0.05-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-39--RNN-same_meanx5-factor-0.2-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-40--RNN-same_meanx5-factor-0.35-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-41--RNN-same_meanx5-factor-0.5-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-42--RNN-same_meanx9-factor-0.05-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-43--RNN-same_meanx9-factor-0.2-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-45--RNN-same_meanx9-factor-0.35-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-46--RNN-same_meanx9-factor-0.5-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-47--RNN-both_meanx1-factor-0.05-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-48--RNN-both_meanx1-factor-0.2-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-49--RNN-both_meanx1-factor-0.35-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-50--RNN-both_meanx1-factor-0.5-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-51--RNN-both_meanx3-factor-0.05-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-52--RNN-both_meanx3-factor-0.2-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-54--RNN-both_meanx3-factor-0.35-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-55--RNN-both_meanx3-factor-0.5-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-56--RNN-both_meanx5-factor-0.05-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-57--RNN-both_meanx5-factor-0.2-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-58--RNN-both_meanx5-factor-0.35-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-51-59--RNN-both_meanx5-factor-0.5-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-00--RNN-both_meanx9-factor-0.05-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-02--RNN-both_meanx9-factor-0.2-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-03--RNN-both_meanx9-factor-0.35-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-04--RNN-both_meanx9-factor-0.5-from-data7-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-05--RNN-same_meanx1-factor-0.05-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-06--RNN-same_meanx1-factor-0.2-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-07--RNN-same_meanx1-factor-0.35-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-08--RNN-same_meanx1-factor-0.5-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-09--RNN-same_meanx3-factor-0.05-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-11--RNN-same_meanx3-factor-0.2-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-12--RNN-same_meanx3-factor-0.35-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-13--RNN-same_meanx3-factor-0.5-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-14--RNN-same_meanx5-factor-0.05-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-15--RNN-same_meanx5-factor-0.2-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-16--RNN-same_meanx5-factor-0.35-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-17--RNN-same_meanx5-factor-0.5-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-18--RNN-same_meanx9-factor-0.05-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-20--RNN-same_meanx9-factor-0.2-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-21--RNN-same_meanx9-factor-0.35-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-22--RNN-same_meanx9-factor-0.5-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-23--RNN-both_meanx1-factor-0.05-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-24--RNN-both_meanx1-factor-0.2-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-25--RNN-both_meanx1-factor-0.35-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-26--RNN-both_meanx1-factor-0.5-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-27--RNN-both_meanx3-factor-0.05-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-29--RNN-both_meanx3-factor-0.2-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-30--RNN-both_meanx3-factor-0.35-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-31--RNN-both_meanx3-factor-0.5-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-32--RNN-both_meanx5-factor-0.05-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-33--RNN-both_meanx5-factor-0.2-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-34--RNN-both_meanx5-factor-0.35-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-35--RNN-both_meanx5-factor-0.5-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-37--RNN-both_meanx9-factor-0.05-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-38--RNN-both_meanx9-factor-0.2-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-39--RNN-both_meanx9-factor-0.35-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-40--RNN-both_meanx9-factor-0.5-from-data3-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-41--RNN-same_meanx1-factor-0.05-from-data5-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-42--RNN-same_meanx1-factor-0.2-from-data5-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-43--RNN-same_meanx1-factor-0.35-from-data5-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-45--RNN-same_meanx1-factor-0.5-from-data5-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-46--RNN-same_meanx3-factor-0.05-from-data5-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-47--RNN-same_meanx3-factor-0.2-from-data5-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-48--RNN-same_meanx3-factor-0.35-from-data5-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-49--RNN-same_meanx3-factor-0.5-from-data5-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-50--RNN-same_meanx5-factor-0.05-from-data5-certainTrue-theta-0.5-s989-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN/2021-01-06T12-52-51--RNN-same_meanx5-factor-0.2-from-data5-certainTrue-theta-0.5-s989-train/network"
    ]
    src_data = [os.path.basename(os.path.dirname(pre_train)).split("-")[-6] for pre_train in pretrained_dirs]
    data_source_dirs = [
        "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_val_{}.mat".format(src_dir) for src_dir in src_data]
    for dd, res_from in zip(data_source_dirs, pretrained_dirs):  #
        args.model_name = os.path.basename(os.path.dirname(res_from)).split("-")[6] if os.path.basename(os.path.dirname(res_from)).split("-")[6] != "Res" else "Res_ECG_CAM"
        
        config_dirs = overwrite_params(args, config_dirs,
                                       input_data=dd,  # data dir
                                       certain_dir=None,
                                       rand_seed=989,
                                       restore_from=res_from,
                                       if_single_runs=False,
                                       from_clusterpy=True)

print("ok")
