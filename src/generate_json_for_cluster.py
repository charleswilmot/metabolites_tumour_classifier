import time
import os
import utils
# import json
from dataio import make_output_dir, save_command_line
import shutil
import fnmatch
import numpy as np


def find_folderes(directory, pattern='*.csv'):
    folders = []
    for root, dirnames, filenames in os.walk(directory):
        for subfolder in fnmatch.filter(dirnames, pattern):
            folders.append(os.path.join(root, subfolder))

    return folders

def save_model_yaml_files(dest="./", source="./", file_name="model_parameters.yaml"):
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

    assert os.path.isfile(default_exp_json_dir), "No yaml configuration file found at {}".format(default_exp_json_dir)
    new_exp_json_fn = os.path.join(args.output_path, "network", "exp_parameters.yaml")

    # make dirs
    make_output_dir(args, sub_folders=["AUCs", "CAMs", 'CAMs/mean', "wrong_examples", "certains"])
    save_command_line(args.model_save_dir)

    new_model_son_fn = save_model_yaml_files(dest=os.path.join(args.output_path, "network"), source=default_json_dir, file_name="model_parameters.yaml")
    args.save(os.path.join(args.output_path, "network", "exp_parameters.yaml"))
    time.sleep(1)

    cfg_dirs.append([args.output_path, new_exp_json_fn, new_model_son_fn])
    
    return cfg_dirs


certain_model_files = {
    "Res_ECG_CAM": [
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T22-07-37--Res_ECG_CAM-nonex0-factor-0-from-data5-certainFalse-theta-0-s989-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T22-07-38--Res_ECG_CAM-nonex0-factor-0-from-data3-certainFalse-theta-0-s989-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T22-07-39--Res_ECG_CAM-nonex0-factor-0-from-data1-certainFalse-theta-0-s989-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T23-08-12--Res_ECG_CAM-nonex0-factor-0-from-data9-certainFalse-theta-0-s989-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-10-14T23-08-13--Res_ECG_CAM-nonex0-factor-0-from-data7-certainFalse-theta-0-s989-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-12-18T13-50-31--Res_ECG_CAM-nonex0-factor-0-from-data6-certainFalse-theta-0-s899-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-12-18T13-50-32--Res_ECG_CAM-nonex0-factor-0-from-data8-certainFalse-theta-0-s899-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-12-18T13-51-29--Res_ECG_CAM-nonex0-factor-0-from-data2-certainFalse-theta-0-s899-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Res_ECG_CAM/2020-12-18T13-51-30--Res_ECG_CAM-nonex0-factor-0-from-data4-certainFalse-theta-0-s899-100rns-train"],
    "MLP":[
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-MLP/2020-11-18T19-07-11--MLP-nonex0-factor-0-from-data5-certainFalse-theta-0-s789-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-MLP/2020-10-14T14-29-18--MLP-nonex0-factor-0-from-data3-certainFalse-theta-0-s989-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-MLP/2020-10-14T14-30-52--MLP-nonex0-factor-0-from-data9-certainFalse-theta-0-s989-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-MLP/2020-10-14T14-30-53--MLP-nonex0-factor-0-from-data7-certainFalse-theta-0-s989-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-MLP/2020-11-18T19-07-12--MLP-nonex0-factor-0-from-data1-certainFalse-theta-0-s789-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-MLP/2020-11-18T19-07-14--MLP-nonex0-factor-0-from-data2-certainFalse-theta-0-s789-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-MLP/2021-01-12T11-03-06--MLP-nonex0-factor-0-from-data8-certainFalse-theta-0-s989-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-MLP/2021-01-12T11-03-07--MLP-nonex0-factor-0-from-data6-certainFalse-theta-0-s989-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-MLP/2021-01-12T11-03-09--MLP-nonex0-factor-0-from-data4-certainFalse-theta-0-s989-100rns-train"],
    "RNN": [
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-RNN/2020-10-27T19-36-39--RNN-nonex0-factor-0-from-data3-certainFalse-theta-0-s789-100rns-train",     "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-RNN/2020-10-27T19-36-40--RNN-nonex0-factor-0-from-data4-certainFalse-theta-0-s789-100rns-train",     "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-RNN/2020-10-27T19-36-41--RNN-nonex0-factor-0-from-data6-certainFalse-theta-0-s789-100rns-train",     "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-RNN/2020-10-27T19-36-43--RNN-nonex0-factor-0-from-data7-certainFalse-theta-0-s789-100rns-train",     "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-RNN/2020-10-27T19-36-44--RNN-nonex0-factor-0-from-data8-certainFalse-theta-0-s789-100rns-train",     "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-RNN/2020-10-27T19-36-45--RNN-nonex0-factor-0-from-data9-certainFalse-theta-0-s789-100rns-train",     "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-RNN/2020-11-19T12-15-01--RNN-nonex0-factor-0-from-data5-certainFalse-theta-0-s789-100rns-train",     "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-RNN/2020-11-19T12-15-02--RNN-nonex0-factor-0-from-data1-certainFalse-theta-0-s789-100rns-train",     "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-RNN/2020-11-19T12-15-04--RNN-nonex0-factor-0-from-data2-certainFalse-theta-0-s789-100rns-train"],
    "Inception": [
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2020-11-18T10-23-19--Inception-nonex0-factor-0-from-data5-certainFalse-theta-0-s789-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2020-11-18T10-23-20--Inception-nonex0-factor-0-from-data1-certainFalse-theta-0-s789-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2020-11-18T10-23-21--Inception-nonex0-factor-0-from-data2-certainFalse-theta-0-s789-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2021-01-12T11-47-50--Inception-nonex0-factor-0-from-data9-certainFalse-theta-0-s989-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2021-01-12T11-47-51--Inception-nonex0-factor-0-from-data8-certainFalse-theta-0-s989-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2021-01-12T11-47-52--Inception-nonex0-factor-0-from-data7-certainFalse-theta-0-s989-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2021-01-12T11-47-54--Inception-nonex0-factor-0-from-data6-certainFalse-theta-0-s989-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2021-01-12T11-47-55--Inception-nonex0-factor-0-from-data4-certainFalse-theta-0-s989-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2021-01-12T11-47-56--Inception-nonex0-factor-0-from-data3-certainFalse-theta-0-s989-100rns-train"]
}

######################################################################################################################
default_json_dir = "./"
default_output_root= "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review"
# default_json_dir = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/src"
default_train_or_test = "train"
## Load experiment parameters and model parameters
default_exp_json_dir = os.path.join(default_json_dir, "exp_parameters.yaml")
default_model_json_dir = os.path.join(default_json_dir, "model_parameters.yaml")

args = utils.load_all_params_yaml(default_exp_json_dir, default_model_json_dir)


data_source_dirs = [
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data5.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data9.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data8.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data7.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data6.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data4.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data3.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data2.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data1.mat",
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data0.mat",
]

# overwrite part of the parameters given for training with cluster.py


config_dirs = []
seed = np.random.randint(9999)
mode = "test"


if mode == "single_runs":
    ## 100 single-epoch runs
    args.new_folder = "100-single-epoch-runs-mnist"
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
    aug_method = "certain"
    
    for model_name in ["Res_ECG_CAM", "RNN"]:#"MLP", "Inception",
        args.model_name = model_name
        if aug_method == "random":
            certain_dirs = [None] * len(data_source_dirs)
            theta = 1
            args.new_folder = "2-randomDA-new"
        elif aug_method == "certain":  #1,2,3,5,7,9
            # certain_dirs = [
            #     "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-RNN/2020-10-27T19-36-39--RNN-nonex0-factor-0-from-data3-certainFalse-theta-0-s789-100rns-train",
            #     "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-RNN/2020-10-27T19-36-40--RNN-nonex0-factor-0-from-data4-certainFalse-theta-0-s789-100rns-train",
            #     "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-RNN/2020-10-27T19-36-41--RNN-nonex0-factor-0-from-data6-certainFalse-theta-0-s789-100rns-train",
            #     "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-RNN/2020-10-27T19-36-43--RNN-nonex0-factor-0-from-data7-certainFalse-theta-0-s789-100rns-train",
            #     "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-RNN/2020-10-27T19-36-44--RNN-nonex0-factor-0-from-data8-certainFalse-theta-0-s789-100rns-train",
            #     "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-RNN/2020-10-27T19-36-45--RNN-nonex0-factor-0-from-data9-certainFalse-theta-0-s789-100rns-train",
            #      ]
            certain_dirs = certain_model_files[args.model_name]
            src_data_dirs = [os.path.basename(certain).split("-")[-7] for certain in certain_dirs]
            data_source_dirs = [
                "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_{}.mat".format(
                    src_dir) for src_dir in src_data_dirs]
            theta = 0.5
            args.new_folder = "3-certain-DA-with-distillation"
        for dd, certain in zip(data_source_dirs, certain_dirs):   #, certain_dirs)
            for method in ["both_mean"]:  #, "both_mean", "same_mean"， "ops_mean", "noise" #
                for fold in [3]:  #, 5, 3,  , 7, 10, 9,  #
                    for scale in [0.5]:  # 0.35, 0.2, 3, 0.5#
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
        
        
elif mode == "testing_given_the_whole_folder":

    data_dirs = [
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-with-distillation-Inception"
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-new-MLP",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-new-Res_ECG_CAM",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-new-RNN"
    ]
    for jj in data_dirs:
        test_dirs = find_folderes(jj, pattern="*-train")
        
        pretrained_dirs = [os.path.join(pre_train, "network") for pre_train in test_dirs]
        src_data_dirs = [os.path.basename(os.path.dirname(pre_train)).split("-")[-6] for pre_train in pretrained_dirs]
        data_source_dirs = [
            "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_val_{}.mat".format(src_dir) for src_dir in src_data_dirs]
        for dd, res_from in zip(data_source_dirs, pretrained_dirs):  #
            args.model_name = os.path.basename(os.path.dirname(res_from)).split("-")[6] if os.path.basename(os.path.dirname(res_from)).split("-")[6] != "Res" else "Res_ECG_CAM"
            
            config_dirs = overwrite_params(args, config_dirs,
                                           input_data=dd,  # data dir
                                           certain_dir=None,
                                           rand_seed=989,
                                           restore_from=res_from,
                                           if_single_runs=False,
                                           from_clusterpy=True)

elif mode == "testing_given_specific_pretrained":
    pretrained_dirs = [
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-with-distillation-MLP/2021-02-09T14-49-30--MLP-both_meanx3-factor-0.5-from-data5-certainTrue-theta-0.5-s6786-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-with-distillation-MLP/2021-02-09T14-49-31--MLP-both_meanx3-factor-0.5-from-data3-certainTrue-theta-0.5-s6786-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-with-distillation-MLP/2021-02-09T14-49-32--MLP-both_meanx3-factor-0.5-from-data9-certainTrue-theta-0.5-s6786-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-with-distillation-MLP/2021-02-09T14-49-33--MLP-both_meanx3-factor-0.5-from-data7-certainTrue-theta-0.5-s6786-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-with-distillation-MLP/2021-02-09T14-49-34--MLP-both_meanx3-factor-0.5-from-data1-certainTrue-theta-0.5-s6786-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-with-distillation-MLP/2021-02-09T14-49-35--MLP-both_meanx3-factor-0.5-from-data2-certainTrue-theta-0.5-s6786-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-with-distillation-MLP/2021-02-09T14-49-36--MLP-both_meanx3-factor-0.5-from-data8-certainTrue-theta-0.5-s6786-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-with-distillation-MLP/2021-02-09T14-49-37--MLP-both_meanx3-factor-0.5-from-data6-certainTrue-theta-0.5-s6786-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-with-distillation-MLP/2021-02-09T14-49-38--MLP-both_meanx3-factor-0.5-from-data4-certainTrue-theta-0.5-s6786-train/network"
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Res_ECG_CAM/2021-01-10T23-16-15--Res_ECG_CAM-both_meanx1-factor-0.2-from-data4-certainFalse-theta-1-s5809-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Res_ECG_CAM/2021-01-10T23-16-16--Res_ECG_CAM-both_meanx1-factor-0.05-from-data4-certainFalse-theta-1-s5809-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Res_ECG_CAM/2021-01-10T23-16-18--Res_ECG_CAM-ops_meanx9-factor-0.5-from-data4-certainFalse-theta-1-s5809-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Res_ECG_CAM/2021-01-10T23-16-19--Res_ECG_CAM-ops_meanx9-factor-0.35-from-data4-certainFalse-theta-1-s5809-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Res_ECG_CAM/2021-01-10T23-16-20--Res_ECG_CAM-ops_meanx9-factor-0.2-from-data4-certainFalse-theta-1-s5809-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Res_ECG_CAM/2021-01-10T23-16-21--Res_ECG_CAM-ops_meanx9-factor-0.05-from-data4-certainFalse-theta-1-s5809-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Res_ECG_CAM/2021-01-10T23-16-22--Res_ECG_CAM-ops_meanx5-factor-0.5-from-data4-certainFalse-theta-1-s5809-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Res_ECG_CAM/2021-01-10T23-16-23--Res_ECG_CAM-ops_meanx5-factor-0.35-from-data4-certainFalse-theta-1-s5809-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Res_ECG_CAM/2021-01-10T23-16-25--Res_ECG_CAM-ops_meanx5-factor-0.2-from-data4-certainFalse-theta-1-s5809-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Res_ECG_CAM/2021-01-10T23-16-26--Res_ECG_CAM-ops_meanx5-factor-0.05-from-data4-certainFalse-theta-1-s5809-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Res_ECG_CAM/2021-01-10T23-16-27--Res_ECG_CAM-ops_meanx3-factor-0.5-from-data4-certainFalse-theta-1-s5809-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Res_ECG_CAM/2021-01-10T23-16-28--Res_ECG_CAM-ops_meanx3-factor-0.35-from-data4-certainFalse-theta-1-s5809-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Res_ECG_CAM/2021-01-10T23-16-29--Res_ECG_CAM-ops_meanx3-factor-0.2-from-data4-certainFalse-theta-1-s5809-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Res_ECG_CAM/2021-01-10T23-16-30--Res_ECG_CAM-ops_meanx3-factor-0.05-from-data4-certainFalse-theta-1-s5809-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Res_ECG_CAM/2021-01-10T23-16-32--Res_ECG_CAM-ops_meanx1-factor-0.5-from-data4-certainFalse-theta-1-s5809-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Res_ECG_CAM/2021-01-10T23-16-33--Res_ECG_CAM-ops_meanx1-factor-0.35-from-data4-certainFalse-theta-1-s5809-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Res_ECG_CAM/2021-01-10T23-16-34--Res_ECG_CAM-ops_meanx1-factor-0.2-from-data4-certainFalse-theta-1-s5809-train/network",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Res_ECG_CAM/2021-01-10T23-16-35--Res_ECG_CAM-ops_meanx1-factor-0.05-from-data4-certainFalse-theta-1-s5809-train/network",
        # # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP/2021-01-08T22-00-01--MLP-ops_meanx9-factor-0.35-from-data7-certainFalse-theta-1-s7984-train/network",
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
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/9-train-with-MNIST-MLP/2021-02-24T07-09-21--MLP-both_meanx0-factor-0-from-mnist-ctTrue-theta-0.8-s3699-train-trainOnTrue/network"
    ]
    data_dirs = [
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-with-distillation-Inception"
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-new-MLP",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-new-Res_ECG_CAM",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-new-RNN"
    ]
    for jj in data_dirs:
        test_dirs = find_folderes(jj, pattern="*-trainOnTrue")
        
        pretrained_dirs = [os.path.join(pre_train, "network") for pre_train in
                           test_dirs]
        src_data_dirs = [
            os.path.basename(os.path.dirname(pre_train)).split("-")[-6] for
            pre_train in pretrained_dirs]
        data_source_dirs = [
            "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_val_{}.mat".format(
                src_dir) for src_dir in src_data_dirs]
        for dd, res_from in zip(data_source_dirs, pretrained_dirs):  #
            args.model_name = \
            os.path.basename(os.path.dirname(res_from)).split("-")[6] if \
            os.path.basename(os.path.dirname(res_from)).split("-")[
                6] != "Res" else "Res_ECG_CAM"
            
            config_dirs = overwrite_params(args, config_dirs,
                                           input_data=dd,  # data dir
                                           certain_dir=None,
                                           rand_seed=989,
                                           restore_from=res_from,
                                           if_single_runs=False,
                                           from_clusterpy=True)

print("ok")
