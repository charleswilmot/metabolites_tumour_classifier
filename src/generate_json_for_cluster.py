import time
import os
import utils
import json
import datetime
from dataio import make_output_dir, make_output_dir
import shutil

default_json_dir = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/src"

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
#
# def make_output_dir(output_path, sub_folders=["CAMs"]):
#     if os.path.isdir(output_path):
#         raise FileExistsError("Output path already exists.", output_path)
#     else:
#         os.makedirs(output_path)
#         model_save_dir = os.path.join(output_path, "network")
#         os.makedirs(model_save_dir)
#         for sub in sub_folders:
#             os.makedirs(os.path.join(output_path, sub))


# def copy_save_all_files(model_save_dir):
#     """
#     Copy and save all files related to model directory
#     :param model_save_dir:
#     :return:
#     """
#     src_dir = '../src'
#     save_dir = os.path.join(model_save_dir, 'src')
#     if not os.path.exists(save_dir):  # if subfolder doesn't exist, should make the directory and then save file.
#         os.makedirs(save_dir)
#     req_extentions = ['py', 'json']
#     for filename in os.listdir(src_dir):
#         exten = filename.split('.')[-1]
#         if exten in req_extentions:
#             src_file_name = os.path.join(src_dir, filename)
#             target_file_name = os.path.join(save_dir, filename)
#             with open(src_file_name, 'r') as file_src:
#                 with open(target_file_name, 'w') as file_dst:
#                     for line in file_src:
#                         file_dst.write(line)
#     print('Done WithCopy File!')

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
    args.certain_dir = kwargs["certain_dir"] if "certain_dir" in kwargs else None

    # GET output_path
    args = utils.generate_output_path(args, time_str='{0:%Y-%m-%dT%H-%M-%S-}'.format(datetime.datetime.now()))
    make_output_dir(args, sub_folders=["AUCs", "CAMs", 'CAMs/mean', "wrong_examples", "certains"])

    assert os.path.isfile(default_exp_json_dir), "No json configuration file found at {}".format(default_exp_json_dir)

    args.save(os.path.join(args.output_path, "network", "exp_parameters.json"))
    new_model_son_fn = save_model_json(dest=os.path.join(args.output_path, "network"), source=default_json_dir, file_name="model_parameters.json")

    new_exp_json_fn = os.path.join(args.output_path, "network", "exp_parameters.json")
    cfg_dirs.append([args.output_path, new_exp_json_fn, new_model_son_fn])

    time.sleep(1)

    return cfg_dirs


default_train_or_test = "train"
## Load experiment parameters and model parameters
default_exp_json_dir = os.path.join(default_json_dir, "exp_parameters.json")
default_model_json_dir = os.path.join(default_json_dir, "model_parameters.json")

args = utils.load_all_params(default_exp_json_dir, default_model_json_dir)

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
config_dirs = []
for dd in data_source_dirs:
    for method in ["both_mean", "same_mean", "ops_mean"]:  #
        for fold in [1, 3, 5, 9]:  #, 3, 5, 7, 9
            for scale in [0.05, 0.2, 0.35, 0.5]:  #, 0.3, 0.5
                config_dirs = overwrite_params(args, config_dirs,
                                               input_data=dd,  #data dir
                                               certain_dir=None,
                                               aug_method=method,
                                               aug_scale=scale,
                                               aug_folds=fold,
                                               theta_thr=1,
                                               randseed=99,
                                               if_single_runs=False,
                                               from_clusterpy=True)

