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
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data9.mat",
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data8.mat",
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data7.mat",
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data6.mat",
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data4.mat",
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data3.mat",
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data2.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data1.mat",
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
            "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T22-07-39--Res_ECG_CAM-nonex0-factor-0-from-data1-certainFalse-theta-0-s989-100rns-train",
            "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-12-18T13-51-29--Res_ECG_CAM-nonex0-factor-0-from-data2-certainFalse-theta-0-s899-100rns-train",
            "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T22-07-38--Res_ECG_CAM-nonex0-factor-0-from-data3-certainFalse-theta-0-s989-100rns-train"
            "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-12-18T13-51-30--Res_ECG_CAM-nonex0-factor-0-from-data4-certainFalse-theta-0-s899-100rns-train",
            # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T22-07-37--Res_ECG_CAM-nonex0-factor-0-from-data5-certainFalse-theta-0-s989-100rns-train",
            # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T23-08-12--Res_ECG_CAM-nonex0-factor-0-from-data9-certainFalse-theta-0-s989-100rns-train",
            # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T23-08-13--Res_ECG_CAM-nonex0-factor-0-from-data7-certainFalse-theta-0-s989-100rns-train",
            ## save for later
            # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-12-18T13-50-31--Res_ECG_CAM-nonex0-factor-0-from-data6-certainFalse-theta-0-s899-100rns-train",
            # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-12-18T13-50-32--Res_ECG_CAM-nonex0-factor-0-from-data8-certainFalse-theta-0-s899-100rns-train",
            ]
        src_data = [os.path.basename(certain).split("-")[-7] for certain in certain_dirs]
        data_source_dirs = [
            "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_{}.mat".format(
                src_dir) for src_dir in src_data]
        theta = 0.5
        args.new_folder = "3-certain-DA"
    for dd, certain in zip(data_source_dirs, certain_dirs):   #, certain_dirs)
        for method in ["same_mean", "both_mean", "ops_mean"]:  #, "noise" #
            for fold in [9, 5]:  #, 3, 1 , 7, 10, 9,  #
                for scale in [0.5, 0.35]:  #, 0.2, 0.05  3, 0.5#
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
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-30--MLP-same_meanx3-factor-0.2-from-data6-certainFalse-theta-1-s7984-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-31--MLP-same_meanx3-factor-0.05-from-data6-certainFalse-theta-1-s7984-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-33--MLP-same_meanx1-factor-0.5-from-data6-certainFalse-theta-1-s7984-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-34--MLP-same_meanx1-factor-0.35-from-data6-certainFalse-theta-1-s7984-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-35--MLP-same_meanx1-factor-0.2-from-data6-certainFalse-theta-1-s7984-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-36--MLP-same_meanx1-factor-0.05-from-data6-certainFalse-theta-1-s7984-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-37--MLP-both_meanx9-factor-0.5-from-data6-certainFalse-theta-1-s7984-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-38--MLP-both_meanx9-factor-0.35-from-data6-certainFalse-theta-1-s7984-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-40--MLP-both_meanx9-factor-0.2-from-data6-certainFalse-theta-1-s7984-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-41--MLP-both_meanx9-factor-0.05-from-data6-certainFalse-theta-1-s7984-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-42--MLP-both_meanx5-factor-0.5-from-data6-certainFalse-theta-1-s7984-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-43--MLP-both_meanx5-factor-0.35-from-data6-certainFalse-theta-1-s7984-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-44--MLP-both_meanx5-factor-0.2-from-data6-certainFalse-theta-1-s7984-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-45--MLP-both_meanx5-factor-0.05-from-data6-certainFalse-theta-1-s7984-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-47--MLP-both_meanx3-factor-0.5-from-data6-certainFalse-theta-1-s7984-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-48--MLP-both_meanx3-factor-0.35-from-data6-certainFalse-theta-1-s7984-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-49--MLP-both_meanx3-factor-0.2-from-data6-certainFalse-theta-1-s7984-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-50--MLP-both_meanx3-factor-0.05-from-data6-certainFalse-theta-1-s7984-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-51--MLP-both_meanx1-factor-0.5-from-data6-certainFalse-theta-1-s7984-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T23-38-04--MLP-same_meanx9-factor-0.5-from-data9-certainFalse-theta-1-s3494-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T23-38-06--MLP-same_meanx5-factor-0.5-from-data9-certainFalse-theta-1-s3494-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T23-40-09--MLP-same_meanx9-factor-0.5-from-data9-certainFalse-theta-1-s6121-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T23-40-10--MLP-same_meanx5-factor-0.5-from-data9-certainFalse-theta-1-s6121-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T23-40-24--MLP-noisex0-factor-0-from-data5-certainFalse-theta-1-s798-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T23-50-22--MLP-noisex0-factor-0-from-data5-certainFalse-theta-1-s798-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-03-46--MLP-same_meanx9-factor-0.5-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-03-47--MLP-same_meanx9-factor-0.35-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-03-48--MLP-same_meanx9-factor-0.2-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-03-49--MLP-same_meanx9-factor-0.05-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-03-50--MLP-same_meanx5-factor-0.5-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-03-51--MLP-same_meanx5-factor-0.35-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-03-52--MLP-same_meanx5-factor-0.2-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-03-54--MLP-same_meanx5-factor-0.05-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-03-55--MLP-same_meanx3-factor-0.5-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-03-56--MLP-same_meanx3-factor-0.35-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-03-57--MLP-same_meanx3-factor-0.2-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-03-58--MLP-same_meanx3-factor-0.05-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-03-59--MLP-same_meanx1-factor-0.5-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-00--MLP-same_meanx1-factor-0.35-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-02--MLP-same_meanx1-factor-0.2-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-03--MLP-same_meanx1-factor-0.05-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-04--MLP-both_meanx9-factor-0.5-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-05--MLP-both_meanx9-factor-0.35-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-06--MLP-both_meanx9-factor-0.2-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-07--MLP-both_meanx9-factor-0.05-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-08--MLP-both_meanx5-factor-0.5-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-10--MLP-both_meanx5-factor-0.35-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-11--MLP-both_meanx5-factor-0.2-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-12--MLP-both_meanx5-factor-0.05-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-13--MLP-both_meanx3-factor-0.5-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-14--MLP-both_meanx3-factor-0.35-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-15--MLP-both_meanx3-factor-0.2-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-17--MLP-both_meanx3-factor-0.05-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-18--MLP-both_meanx1-factor-0.5-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-19--MLP-both_meanx1-factor-0.35-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-20--MLP-both_meanx1-factor-0.2-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-21--MLP-both_meanx1-factor-0.05-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-22--MLP-ops_meanx9-factor-0.5-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-23--MLP-ops_meanx9-factor-0.35-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-25--MLP-ops_meanx9-factor-0.2-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-26--MLP-ops_meanx9-factor-0.05-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-27--MLP-ops_meanx5-factor-0.5-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-28--MLP-ops_meanx5-factor-0.35-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-29--MLP-ops_meanx5-factor-0.2-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-30--MLP-ops_meanx5-factor-0.05-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-32--MLP-ops_meanx3-factor-0.5-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-33--MLP-ops_meanx3-factor-0.35-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-34--MLP-ops_meanx3-factor-0.2-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-35--MLP-ops_meanx3-factor-0.05-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-36--MLP-ops_meanx1-factor-0.5-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-37--MLP-ops_meanx1-factor-0.35-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-38--MLP-ops_meanx1-factor-0.2-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T00-04-40--MLP-ops_meanx1-factor-0.05-from-data4-certainFalse-theta-1-s462-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T14-54-48--MLP-same_meanx9-factor-0.5-from-data9-certainFalse-theta-1-s142-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T14-54-49--MLP-same_meanx9-factor-0.5-from-data8-certainFalse-theta-1-s142-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T14-54-50--MLP-same_meanx9-factor-0.5-from-data7-certainFalse-theta-1-s142-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T14-54-51--MLP-same_meanx9-factor-0.5-from-data4-certainFalse-theta-1-s142-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T15-24-39--MLP-noisex0-factor-0-from-data5-certainFalse-theta-1-s798-train/network",
  "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-09T15-29-14--MLP-noisex0-factor-0-from-data5-certainFalse-theta-1-s798-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-14--MLP-ops_meanx3-factor-0.5-from-data8-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-15--MLP-ops_meanx3-factor-0.35-from-data8-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-16--MLP-ops_meanx3-factor-0.2-from-data8-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-17--MLP-ops_meanx3-factor-0.05-from-data8-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-18--MLP-ops_meanx1-factor-0.5-from-data8-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-19--MLP-ops_meanx1-factor-0.35-from-data8-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-20--MLP-ops_meanx1-factor-0.2-from-data8-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-22--MLP-ops_meanx1-factor-0.05-from-data8-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-23--MLP-same_meanx9-factor-0.5-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-24--MLP-same_meanx9-factor-0.35-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-25--MLP-same_meanx9-factor-0.2-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-26--MLP-same_meanx9-factor-0.05-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-27--MLP-same_meanx5-factor-0.5-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-29--MLP-same_meanx5-factor-0.35-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-30--MLP-same_meanx5-factor-0.2-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-31--MLP-same_meanx5-factor-0.05-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-32--MLP-same_meanx3-factor-0.5-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-33--MLP-same_meanx3-factor-0.35-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-34--MLP-same_meanx3-factor-0.2-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-36--MLP-same_meanx3-factor-0.05-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-37--MLP-same_meanx1-factor-0.5-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-38--MLP-same_meanx1-factor-0.35-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-39--MLP-same_meanx1-factor-0.2-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-40--MLP-same_meanx1-factor-0.05-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-41--MLP-both_meanx9-factor-0.5-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-43--MLP-both_meanx9-factor-0.35-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-44--MLP-both_meanx9-factor-0.2-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-45--MLP-both_meanx9-factor-0.05-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-46--MLP-both_meanx5-factor-0.5-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-47--MLP-both_meanx5-factor-0.35-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-48--MLP-both_meanx5-factor-0.2-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-50--MLP-both_meanx5-factor-0.05-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-51--MLP-both_meanx3-factor-0.5-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-52--MLP-both_meanx3-factor-0.35-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-53--MLP-both_meanx3-factor-0.2-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-54--MLP-both_meanx3-factor-0.05-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-55--MLP-both_meanx1-factor-0.5-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-57--MLP-both_meanx1-factor-0.35-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-58--MLP-both_meanx1-factor-0.2-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T21-59-59--MLP-both_meanx1-factor-0.05-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-00--MLP-ops_meanx9-factor-0.5-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-01--MLP-ops_meanx9-factor-0.35-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-02--MLP-ops_meanx9-factor-0.2-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-04--MLP-ops_meanx9-factor-0.05-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-05--MLP-ops_meanx5-factor-0.5-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-06--MLP-ops_meanx5-factor-0.35-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-07--MLP-ops_meanx5-factor-0.2-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-08--MLP-ops_meanx5-factor-0.05-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-09--MLP-ops_meanx3-factor-0.5-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-11--MLP-ops_meanx3-factor-0.35-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-12--MLP-ops_meanx3-factor-0.2-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-13--MLP-ops_meanx3-factor-0.05-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-14--MLP-ops_meanx1-factor-0.5-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-15--MLP-ops_meanx1-factor-0.35-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-16--MLP-ops_meanx1-factor-0.2-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-18--MLP-ops_meanx1-factor-0.05-from-data7-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-19--MLP-same_meanx9-factor-0.5-from-data6-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-20--MLP-same_meanx9-factor-0.35-from-data6-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-21--MLP-same_meanx9-factor-0.2-from-data6-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-22--MLP-same_meanx9-factor-0.05-from-data6-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-23--MLP-same_meanx5-factor-0.5-from-data6-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-24--MLP-same_meanx5-factor-0.35-from-data6-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-26--MLP-same_meanx5-factor-0.2-from-data6-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-27--MLP-same_meanx5-factor-0.05-from-data6-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-28--MLP-same_meanx3-factor-0.5-from-data6-certainFalse-theta-1-s7984-train/network",
  # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-29--MLP-same_meanx3-factor-0.35-from-data6-certainFalse-theta-1-s7984-train/network"
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
