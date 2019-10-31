import time
import os


default_augmentation_method = "ops_mean"
default_factor = 0.2
EXPERIMENT_DIR_ROOT = "results/"


def make_experiment_path(augmentation_method=None, factor=None, description=None):
    date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    augmentation_method = default_augmentation_method if augmentation_method is None else augmentation_method
    factor = default_factor if factor is None else factor
    description = ("__" + description) if description else ""
    experiment_dir = EXPERIMENT_DIR_ROOT + "{}_augmentation_method_{}_factor_{}{}".format(date, augmentation_method, factor, description)
    return experiment_dir


class ClusterQueue:
    def __init__(self, **kwargs):
        # generate a path for the results + mkdir
        # TODO
        self.experiment_path = make_experiment_path(
            augmentation_method=kwargs["augmentation_method"] if "augmentation_method" in kwargs else None,
            factor=kwargs["factor"] if "factor" in kwargs else None,
            from_epoch=kwargs["from_epoch"] if "from_epoch" in kwargs else None,
            description=kwargs["description"])
        os.mkdir(self.experiment_path)

        # output path for the experiment log
        self.cmd_slurm = "sbatch --output {}/%N_%j.log".format(self.experiment_path)

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
        self.cmd_python += self._to_arg("--experiment-path", self.experiment_path)

        self.cmd = self.cmd_slurm + self.cmd_python
        print("\n###############")
        print(self.cmd_slurm)
        print(self.cmd_python)
        print("###############\n")
        # TODO
        # os.system(self.cmd)
        print(self.cmd)
        time.sleep(1)

    def _key_to_flag(self, key):
        return "--" + key.replace("_", "-")

    def _to_arg(self, flag, v):
        return " {} {}".format(flag, v)

    def watch_tail(self):
        os.system("watch tail -n 40 \"{}\"".format(self.experiment_path + "/log/*.log"))


# run all the experiments with different configurations
for ep_num in range(1, 21):
    for augmentation_method in ["mean", "ops_mean", "both"]:
        for factor in range(0, 1, 0.05):
            for fold in range(1, 11):
                cq = ClusterQueue(
                    description="Res_ECG_CNN",
                    augmentation_method=augmentation_method,
                    factor=factor,
                    aug_fold=fold,
                    from_epoch=ep_num)
