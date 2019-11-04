import time
import os
import numpy as np

default_aug_method = "ops_mean"
default_factor = 0.2
default_folds = 5
default_aug_scale = 0.3
default_from_epoch = 3
EXPERIMENT_DIR_ROOT = "../results/"


def generate_experiment_path_str(aug_method=None, aug_scale=None, aug_folds=None, description=None, from_epoch=None, restore_from=None):
    date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    aug_method = default_aug_method if aug_method is None else aug_method
    aug_scale = default_aug_scale if aug_scale is None else aug_scale
    aug_folds = default_folds if aug_folds is None else aug_folds
    from_epoch = default_from_epoch if from_epoch is None else from_epoch
    if restore_from:
        restore_from = restore_from
    else:
        raise ValueError('A model dir should be passed into!')
    description = description if description else "test"
    experiment_dir = EXPERIMENT_DIR_ROOT + "{}_exp0.874_{}x{}_factor_{}_from-epoch_{}_{}".format(date, aug_method, aug_folds, aug_scale, from_epoch, description)
    return experiment_dir


def make_output_dir(output_path, sub_folders=["CAMs"]):
    if os.path.isdir(output_path):
        logger.critical("Output path already exists. Please use an other path.")
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


class ClusterQueue:
    def __init__(self, **kwargs):
        # generate a path for the results + mkdir
        # TODO
        self.output_path = generate_experiment_path_str(
            aug_method=kwargs["aug_method"] if "aug_method" in kwargs else None,
            aug_scale=kwargs["aug_scale"] if "aug_scale" in kwargs else None,
            aug_folds=kwargs["aug_folds"] if "aug_folds" in kwargs else None,
            from_epoch=kwargs["from_epoch"] if "from_epoch" in kwargs else None,
            description=kwargs["description"] if "description" in kwargs else None,
            restore_from=kwargs["restore_from"] if "restore_from" in kwargs else None)
        make_output_dir(self.output_path, sub_folders=["AUCs", "CAMs", 'CAMs/mean', "wrong_examples", "certains"])

        # output path for the experiment log
        self.cmd_slurm = "sbatch --output {}/%N_%j.log".format(self.output_path)

        # special treatment for the "description" param (for convevience)
        if "description" in kwargs:
            self.cmd_slurm += " --job-name {}".format(kwargs["description"])
        self.cmd_slurm += " cluster.sh"

        # Creating the flags to be passed to classifier.py
        self.cmd_python = ""
        for k, v in kwargs.items():
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


# run all the experiments with different configurations
# for ep_num in range(1, 11):
#     for augmentation_method in ["same_mean", "ops_mean", "both_mean"]:
#         for factor in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#             for fold in range(1, 11):
#                 cq = ClusterQueue(aug_method=augmentation_method,
#                                   aug_scale=factor,
#                                   aug_folds=fold,
#                                   from_epoch=ep_num)
model_dirs = [
"/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-11-04-10-09-42_exp0.874_same_meanx10_factor_0.05_from-epoch_3_train/network",
"/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-11-04-10-09-43_exp0.874_same_meanx10_factor_0.1_from-epoch_3_train/network",
"/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-11-04-10-09-44_exp0.874_same_meanx10_factor_0.2_from-epoch_3_train/network",
"/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-11-04-10-42-18_exp0.874_same_meanx10_factor_0.3_from-epoch_3_train/network",
"/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-11-04-10-42-20_exp0.874_same_meanx10_factor_0.4_from-epoch_3_train/network",
"/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-11-04-10-42-21_exp0.874_same_meanx10_factor_0.5_from-epoch_3_train/network",
"/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-11-04-10-42-22_exp0.874_same_meanx10_factor_0.6_from-epoch_3_train/network",
"/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-11-04-10-42-23_exp0.874_same_meanx10_factor_0.7_from-epoch_3_train/network",
"/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-11-04-10-42-24_exp0.874_same_meanx10_factor_0.8_from-epoch_3_train/network",
"/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-11-04-10-42-26_exp0.874_same_meanx10_factor_0.9_from-epoch_3_train/network",
"/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-11-04-10-42-27_exp0.874_same_meanx10_factor_0.95_from-epoch_3_train/network"
]

for model in model_dirs:
    cq = ClusterQueue(restore_from=model)
