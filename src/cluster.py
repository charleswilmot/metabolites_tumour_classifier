import time
import os
import generate_json_for_cluster as gen_dir


class ClusterQueue:
    def __init__(self, dirs):

        self.output_path, self.exp_json_dir, self.model_json_dir = dirs[0], dirs[1], dirs[2]

        # output path for the experiment log
        self.cmd_slurm = "sbatch --output {}/%N_%j.log".format(self.output_path)

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

def _key_to_flag(key):
    return "--" + key.replace("_", "_")

def _to_arg(flag, v):
        return " {} {}".format(flag, v)
############################################################################################3
if __name__ == "__main__":
    config_dirs = gen_dir.config_dirs
    #config_dirs =  [[exp1_path1, exp1_arg2, exp1_arg3],
    # [exp2_path1, exp2_arg2, exp2_arg3],
    # [exp3_path1, exp3_arg2, exp3_arg3], ...]

    # config_dirs = [["exp1_path1", "exp1_arg2", "exp1_arg3"],
    #                ["exp2_path1", "exp2_arg2", "exp2_arg3"],
    #                ["exp3_path1", "exp3_arg2", "exp3_arg3"]]


    # Creating the flags to be passed to classifier.py

    cmds_to_sh = []
    for config_files in config_dirs:   #three arguments
        cmd_python = ""
        for k, v in zip(["output_path", "exp_config", "model_config"], [config_files[0], config_files[1], config_files[2]]):
            # _key_to_flag transforms "something_stupid"   into   "--something-stupid"
            flag = _key_to_flag(k)
            # _to_arg transforms ("--something-stupid", a_value)   into   "--something-stupid a_value"
            arg = _to_arg(flag, v)
            cmd_python += arg
        cmds_to_sh.append(cmd_python)
        # cmds_to_sh.append(cmd_python + " --output {}/%N_%j.log".format(config_files[0]))
        # cmd_python = "" --output {}/%N_%j.log".format(config_files[0])"

    for i in range(len(cmds_to_sh)):
        print("-----------------------------")
        print(cmds_to_sh[i])
        print("-----------------------------")
        
    commands = ''
    for cmds in cmds_to_sh:
        commands += "\"{}\" ".format(cmds)

    os.system("sbatch --output {}/%N_%j.log cluster.sh {}".format(config_files[0], commands))

    # for dirs in config_dirs:
    #     ClusterQueue(dirs)

