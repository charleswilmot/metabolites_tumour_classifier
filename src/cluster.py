import time
import os
import numpy as np
import generate_json_for_cluster as gen_dir

default_aug_method = "same_mean"
default_factor = 0.5
default_folds = 10
default_theta = 0.9
default_model_name = "randomDA-Res_ECG_CAM"
default_source = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data5.mat"
EXPERIMENT_DIR_ROOT = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review"

class ClusterQueue:
    def __init__(self, dirs):
        # generate a path for the results + mkdir
        # TODO
        # self.output_path = generate_experiment_path_str(
        #     aug_method=kwargs["aug_method"] if "aug_method" in kwargs else None,
        #     aug_scale=kwargs["aug_scale"] if "aug_scale" in kwargs else None,
        #     aug_folds=kwargs["aug_folds"] if "aug_folds" in kwargs else None,
        #     input_data=kwargs["input_data"] if "input_data" in kwargs else None,
        #     theta_thr=kwargs["theta_thr"] if "theta_thr" in kwargs else None,
        #     rand_seed=kwargs["randseed"] if "randseed" in kwargs else 129,
        #     if_single_runs=kwargs["if_single_runs"] if "if_single_runs" in kwargs else False,
        #     certain_dir=kwargs["certain_dir"] if "certain_dir" in kwargs else None,
        #     description=kwargs["description"] if "description" in kwargs else None)
        self.output_path, self.exp_json_dir, self.model_json_dir = dirs[0], dirs[1], dirs[2]
        # make_output_dir(self.output_path, sub_folders=["AUCs", "CAMs", 'CAMs/mean', "wrong_examples", "certains"])
        # print("ClusterQueue cluster.py if_single_runs", kwargs["if_single_runs"])

        # output path for the experiment log
        self.cmd_slurm = "sbatch --output {}/%N_%j.log".format(self.output_path)
        # self.cmd_slurm = "sbatch --array"

        # special treatment for the "description" param (for convevience)
        # self.cmd_slurm += " --job-name {}".format(kwargs["description"])
        self.cmd_slurm += " cluster.sh"

        # Creating the flags to be passed to classifier.py
        self.cmd_python = ""
        for k, v in zip(["output_path", "exp_config", "model_config"], [self.output_path, self.exp_json_dir, self.model_json_dir]):
            # _key_to_flag transforms "something_stupid"   into   "--something-stupid"
            flag = self._key_to_flag(k)
            # _to_arg transforms ("--something-stupid", a_value)   into   "--something-stupid a_value"
            arg = self._to_arg(flag, v)
            self.cmd_python += arg
        self.cmd_python += self._to_arg("--output_path", self.output_path)

        self.cmd = self.cmd_slurm + self.cmd_python
        print("#########################################################")
        print(self.cmd_slurm, "\n")
        print(self.cmd_python)
        print("##########################################################\n")
        # TODO
        os.system(self.cmd)

        time.sleep(1)

    def _key_to_flag(self, key):
        return "--" + key.replace("_", "_")

    def _to_arg(self, flag, v):
        return " {} {}".format(flag, v)

    def watch_tail(self):
        os.system("watch tail -n 40 \"{}\"".format(self.output_path + "/log/*.log"))


# get the config files with different parameteres
config_dirs = gen_dir.config_dirs

for dirs in config_dirs:
    ClusterQueue(dirs)

# data_source_dirs = [
#     # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data0.mat",
#     # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data9.mat",
#     # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data2.mat",
#     # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data7.mat",
#     # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data4.mat",
#     "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data5.mat",
#     # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data6.mat",
#     "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data3.mat",
#     # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data8.mat",
#     "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data1.mat"
# ]
# certain_dirs = [
#     # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-05T13-54-32-MLP-nonex0-factor-0-from-ep-0-from-lout40-data9-theta-None-s129-100rns-train/certains",
#     # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-05T13-54-31-MLP-nonex0-factor-0-from-ep-0-from-lout40-data7-theta-None-s129-100rns-train/certains",
#     "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-05T13-54-30-MLP-nonex0-factor-0-from-ep-0-from-lout40-data5-theta-None-s129-100rns-train/certains",
#     "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-05T13-54-29-MLP-nonex0-factor-0-from-ep-0-from-lout40-data3-theta-None-s129-100rns-train/certains",
#     "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-05T13-54-28-MLP-nonex0-factor-0-from-ep-0-from-lout40-data1-theta-None-s129-100rns-train/certains"
# ]

# use distilled certain to augment

# for dd in cq = ClusterQueue(
#                             input_data=dd,
#                             certain_dir=None,
#                             aug_method=method,
#                             aug_scale=scale,
#                             aug_folds=fold,
#                             theta_thr=1,
#                             randseed=129,
#                             if_single_runs=False,
#                             from_clusterpy=True)

# few jobs for testing
# for dd in data_source_dirs:
#     for method in ["both_mean"]:  #"same_mean", "ops_mean",
#             for fold in [3]:  #, 3, 5, 7, 9
#                 for scale in [0.35]:  #, 0.3, 0.5
#                     cq = ClusterQueue(
#                             input_data=dd,
#                             certain_dir=None,
#                             aug_method=method,
#                             aug_scale=scale,
#                             aug_folds=fold,
#                             theta_thr=1,
#                             randseed=129,
#                             if_single_runs=False,
#                             from_clusterpy=True)