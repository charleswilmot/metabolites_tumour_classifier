import time
import os
import numpy as np

default_aug_method = "same_mean"
default_factor = 0.5
default_folds = 10
default_theta = 0.999
default_source = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data5.mat"
EXPERIMENT_DIR_ROOT = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review"


def generate_experiment_path_str(aug_method=None, aug_scale=None, aug_folds=None,
                                 description=None, from_epoch=None, input_data=None,
                                 theta_thr=0.99, rand_seed=129):
    date = time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime())
    # aug_method = default_aug_method if aug_method is None else aug_method
    # aug_folds = default_folds if aug_folds is None else aug_folds
    # input_data = default_source if input_data is None else input_data
    # theta_thr = default_theta if theta_thr is None else theta_thr
    if input_data is not None:
        cv_set_id = os.path.basename(input_data).split("_")[-1].split(".")[0]
    else:
        cv_set_id = "TT"
    description = description if description else "train"
    output_path = os.path.join(EXPERIMENT_DIR_ROOT, "{}-{}x{}-factor-{}-from-ep-{}-RNN-from-lout40-{}-theta-{}-s{}-{}".format(date, aug_method, aug_folds, aug_scale, from_epoch, cv_set_id, theta_thr, rand_seed, description))
    # experiment_dir = EXPERIMENT_DIR_ROOT + "{}_exp0.776_{}x{}_factor_{}_from-epoch_{}_{}".format(date, aug_method, aug_folds, aug_scale, from_epoch, description)
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
            from_epoch=kwargs["from_epoch"] if "from_epoch" in kwargs else None,
            input_data=kwargs["input_data"] if "input_data" in kwargs else None,
            theta_thr=kwargs["theta_thr"] if "theta_thr" in kwargs else None,
            rand_seed=kwargs["seed"] if "seed" in kwargs else 129,
            description=kwargs["description"] if "description" in kwargs else None)
        make_output_dir(self.output_path, sub_folders=["AUCs", "CAMs", 'CAMs/mean', "wrong_examples", "certains"])

        # output path for the experiment log
        self.cmd_slurm = "sbatch --output {}/%N_%j.log".format(self.output_path)
        # self.cmd_slurm = "sbatch --array"

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
data_source_dirs = [
    # "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325_DATA.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data0.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data1.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data2.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data3.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data4.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data6.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data7.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data8.mat",
    "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data9.mat"
]

# params during Distillation such as threshold-theta, no Data Augmentation yet
for thr in [0.9]:
    for data_dir in data_source_dirs:
        cq = ClusterQueue(input_data=data_dir,
                          aug_method="None",
                          aug_scale=0,
                          from_epoch=0,
                          aug_folds=0,
                          theta_thr=thr)


#

# params during main classifier training with Data Augmentation yet
# for ep in [5]: #1, 3, , 8, 10
#     for aug_meth in ["same_mean", "both_mean", "ops_mean"]:  #
#         for fd in [1]: #, 3, 9
#             for scale in [0.05, 0.35, 0.5, 0.95]:  #, 0.65, 0.2, 0.8, 0.2, 0.65, 0.8
#                 cq = ClusterQueue(input_data=data_source_dirs[0],
#                                   aug_method="none",
#                                   aug_scale=0,
#                                   from_epoch=0,
#                                   aug_folds=0,
#                                   seed=188)


# 100 single-epoch runs
# random_seeds = np.random.randint(0, 9999, 2)
# for s in random_seeds:
#     cq = ClusterQueue(input_data=data_source_dirs[0],
#                       aug_method="none",
#                       aug_scale=0,
#                       from_epoch=0,
#                       aug_folds=0,
#                       seed=s)