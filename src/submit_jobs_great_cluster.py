import time
import os
import numpy as np
import generate_json_for_cluster

default_aug_method = "same_mean"
default_factor = 0.5
default_folds = 10
default_theta = 0.9
default_model_name = "randomDA-Res_ECG_CAM"
default_source = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data5.mat"
EXPERIMENT_DIR_ROOT = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review"


config_dirs = generate_json_for_cluster.config_dirs
def generate_experiment_path_str(aug_method=None, aug_scale=None, aug_folds=None,
                                 description=None, input_data=None,
                                 theta_thr=0.99, rand_seed=129, if_single_runs=False,
                                 certain_dir=None):
    date = time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime())
    # aug_method = default_aug_method if aug_method is None else aug_method
    # aug_folds = default_folds if aug_folds is None else aug_folds
    # input_data = default_source if input_data is None else input_data
    # theta_thr = default_theta if theta_thr is None else theta_thr
    if input_data is not None:
        cv_set_id = os.path.basename(input_data).split("_")[-1].split(".")[0]
    else:
        cv_set_id = "mnist"
    print("cluster.py if_single_runs", if_single_runs)
    if_from_certain = 1 if certain_dir is not None else 0
    description = description if description else "train"
    description = "100rns-" + description if if_single_runs else description
    output_path = os.path.join(EXPERIMENT_DIR_ROOT, default_model_name, "{}-{}-{}x{}-factor-{}-from-{}-certain{}-theta-{}-s{}-{}".format(date, default_model_name, aug_method, aug_folds, aug_scale, cv_set_id, if_from_certain, theta_thr, rand_seed, description))
    # experiment_dir = EXPERIMENT_DIR_ROOT + "{}_exp0.776_{}x{}_factor_{}_from-epoch_{}_{}".format(date, aug_method, aug_folds, aug_scale, from_epoch, description)
    print("end of generate_experiment_path_str: cluster.py if_single_runs", if_single_runs)
    return output_path


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


class ClusterQueue:
    def __init__(self, **kwargs):
        # generate a path for the results + mkdir
        # TODO
        self.output_path = generate_experiment_path_str(
            aug_method=kwargs["aug_method"] if "aug_method" in kwargs else None,
            aug_scale=kwargs["aug_scale"] if "aug_scale" in kwargs else None,
            aug_folds=kwargs["aug_folds"] if "aug_folds" in kwargs else None,
            input_data=kwargs["input_data"] if "input_data" in kwargs else None,
            theta_thr=kwargs["theta_thr"] if "theta_thr" in kwargs else None,
            rand_seed=kwargs["randseed"] if "randseed" in kwargs else 129,
            if_single_runs=kwargs["if_single_runs"] if "if_single_runs" in kwargs else False,
            certain_dir=kwargs["certain_dir"] if "certain_dir" in kwargs else None,
            description=kwargs["description"] if "description" in kwargs else None)

        make_output_dir(self.output_path, sub_folders=["AUCs", "CAMs", 'CAMs/mean', "wrong_examples", "certains"])
        print("ClusterQueue cluster.py if_single_runs", kwargs["if_single_runs"])

        # output path for the experiment log
        self.cmd_slurm = "sbatch --output {}/%N_%j.log".format(self.output_path)
        # self.cmd_slurm = "sbatch --array"

        # special treatment for the "description" param (for convevience)
        if "description" in kwargs:
            self.cmd_slurm += " --job-name {}".format(kwargs["description"])
        self.cmd_slurm += " submit_jobs_great_cluster.sh"

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
certain_dirs = [
    # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-05T13-54-32-MLP-nonex0-factor-0-from-ep-0-from-lout40-data9-theta-None-s129-100rns-train/certains",
    # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-05T13-54-31-MLP-nonex0-factor-0-from-ep-0-from-lout40-data7-theta-None-s129-100rns-train/certains",
    "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-05T13-54-30-MLP-nonex0-factor-0-from-ep-0-from-lout40-data5-theta-None-s129-100rns-train/certains",
    "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-05T13-54-29-MLP-nonex0-factor-0-from-ep-0-from-lout40-data3-theta-None-s129-100rns-train/certains",
    "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-05T13-54-28-MLP-nonex0-factor-0-from-ep-0-from-lout40-data1-theta-None-s129-100rns-train/certains"
]

# use distilled certain to augment
for dd in data_source_dirs:
    for method in ["same_mean", "ops_mean","both_mean"]:  #
            for fold in [1, 3, 5, 9]:  #, 3, 5, 7, 9
                for scale in [0.05, 0.2, 0.35, 0.5]:  #, 0.3, 0.5
                    cq = ClusterQueue(
                            input_data=dd,
                            certain_dir=None,
                            aug_method=method,
                            aug_scale=scale,
                            aug_folds=fold,
                            theta_thr=1,
                            randseed=129,
                            if_single_runs=False,
                            from_clusterpy=True)

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
