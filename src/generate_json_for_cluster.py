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
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data5.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data1.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data2.mat",
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data3.mat",
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data4.mat",
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data6.mat",
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data7.mat"
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data8.mat",
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data9.mat"
]

# overwrite part of the parameters given for training with cluster.py


config_dirs = []
seed = np.random.randint(9999)
mode = "aug_training"
if mode == "single_runs":
    ## 100 single-epoch runs
    args.new_folder = "100-single-epoch-runs"
    for dd in data_source_dirs:
        config_dirs = overwrite_params(args, config_dirs,
                                       input_data=dd,  # data dir
                                       certain_dir=None,
                                       aug_method="none",
                                       aug_scale=0,
                                       aug_folds=0,
                                       theta_thr=0,
                                       rand_seed=789,
                                       if_single_runs=True,
                                       from_clusterpy=True
                                       )
elif mode == "aug_training":
    aug_method = "certain"
    if aug_method == "random":
        certain_dirs = [None] * len(data_source_dirs)
    elif aug_method == "certain":
        certain_dirs = [
            "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T22-07-37--Res_ECG_CAM-nonex0-factor-0-from-data5-certainFalse-theta-0-s989-100rns-train",
            "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T22-07-38--Res_ECG_CAM-nonex0-factor-0-from-data3-certainFalse-theta-0-s989-100rns-train",
            "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T23-08-13--Res_ECG_CAM-nonex0-factor-0-from-data7-certainFalse-theta-0-s989-100rns-train",
        ]
        src_data = [os.path.basename(certain).split("-")[-7] for certain in certain_dirs]
        data_source_dirs = [
            "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_{}.mat".format(
                src_dir) for src_dir in src_data]
    for dd, certain in zip(data_source_dirs, certain_dirs):   #, certain_dirs)
        for method in ["both_mean" , "same_mean"]:  #,"ops_mean" ,  #
            for fold in [5, 9]:  #, 1,, 5  #
                for scale in [0.05, 0.2, 0.3, 0.5]:  # #
                    config_dirs = overwrite_params(args, config_dirs,
                                                   input_data=dd,  #data dir
                                                   certain_dir=certain,
                                                   aug_method=method,
                                                   aug_scale=scale,
                                                   aug_folds=fold,
                                                   theta_thr=1,
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
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-51-52--Inception-both_meanx1-factor-0.05-from-data5-certainFalse-theta-1-s4875-train/network",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-51-53--Inception-both_meanx1-factor-0.2-from-data5-certainFalse-theta-1-s4875-train/network",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-51-54--Inception-both_meanx1-factor-0.3-from-data5-certainFalse-theta-1-s4875-train/network"
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-51-55--Inception-both_meanx1-factor-0.5-from-data5-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-52-10--Inception-same_meanx1-factor-0.05-from-data5-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-52-11--Inception-same_meanx1-factor-0.2-from-data5-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-52-12--Inception-same_meanx1-factor-0.3-from-data5-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-52-13--Inception-same_meanx1-factor-0.5-from-data5-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-52-27--Inception-both_meanx1-factor-0.05-from-data1-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-52-29--Inception-both_meanx1-factor-0.2-from-data1-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-52-30--Inception-both_meanx1-factor-0.3-from-data1-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-52-31--Inception-both_meanx1-factor-0.5-from-data1-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-52-45--Inception-same_meanx1-factor-0.05-from-data1-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-52-46--Inception-same_meanx1-factor-0.2-from-data1-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-52-47--Inception-same_meanx1-factor-0.3-from-data1-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-52-49--Inception-same_meanx1-factor-0.5-from-data1-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-53-03--Inception-both_meanx1-factor-0.05-from-data2-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-53-04--Inception-both_meanx1-factor-0.2-from-data2-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-53-05--Inception-both_meanx1-factor-0.3-from-data2-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-53-06--Inception-both_meanx1-factor-0.5-from-data2-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-53-21--Inception-same_meanx1-factor-0.05-from-data2-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-53-22--Inception-same_meanx1-factor-0.2-from-data2-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-53-23--Inception-same_meanx1-factor-0.3-from-data2-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-08T22-53-24--Inception-same_meanx1-factor-0.5-from-data2-certainFalse-theta-1-s4875-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-11T00-07-30--Inception-both_meanx1-factor-0.05-from-data5-certainFalse-theta-1-s7614-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-11T00-07-31--Inception-both_meanx1-factor-0.2-from-data5-certainFalse-theta-1-s7614-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-11T00-07-33--Inception-both_meanx1-factor-0.3-from-data5-certainFalse-theta-1-s7614-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-11T00-07-34--Inception-both_meanx1-factor-0.5-from-data5-certainFalse-theta-1-s7614-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-11T00-07-48--Inception-same_meanx1-factor-0.05-from-data5-certainFalse-theta-1-s7614-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-11T00-07-50--Inception-same_meanx1-factor-0.2-from-data5-certainFalse-theta-1-s7614-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-11T00-07-51--Inception-same_meanx1-factor-0.3-from-data5-certainFalse-theta-1-s7614-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-11T00-07-52--Inception-same_meanx1-factor-0.5-from-data5-certainFalse-theta-1-s7614-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-11T00-08-07--Inception-both_meanx1-factor-0.05-from-data1-certainFalse-theta-1-s7614-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-11T00-08-08--Inception-both_meanx1-factor-0.2-from-data1-certainFalse-theta-1-s7614-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-11T00-08-09--Inception-both_meanx1-factor-0.3-from-data1-certainFalse-theta-1-s7614-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-11T00-08-10--Inception-both_meanx1-factor-0.5-from-data1-certainFalse-theta-1-s7614-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-11T13-32-08--Inception-both_meanx3-factor-0.2-from-data5-certainFalse-theta-1-s8113-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-11T13-32-10--Inception-both_meanx3-factor-0.2-from-data1-certainFalse-theta-1-s8113-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-11T13-32-11--Inception-both_meanx3-factor-0.2-from-data2-certainFalse-theta-1-s8113-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Inception/2020-11-16T16-23-48--Inception-same_meanx9-factor-0.2-from-data1-certainFalse-theta-1-s3083-train/network"
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "/network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
        # "network",
    ]
    src_data = [os.path.basename(os.path.dirname(pre_train)).split("-")[-6] for pre_train in pretrained_dirs]
    data_source_dirs = [
        "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_val_{}.mat".format(src_dir) for src_dir in src_data]

    for dd, res_from in zip(data_source_dirs, pretrained_dirs):  #
        config_dirs = overwrite_params(args, config_dirs,
                                       input_data=dd,  # data dir
                                       certain_dir=None,
                                       rand_seed=988,
                                       restore_from=res_from,
                                       if_single_runs=False,
                                       from_clusterpy=True)

print("ok")
