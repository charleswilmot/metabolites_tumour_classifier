import time
import os



class ClusterQueue:
    def __init__(self, dirs):

        self.output_path, self.exp_json_dir, self.model_json_dir = dirs[0], dirs[1], dirs[2]
        
        dir_root = os.path.basename(os.path.dirname(self.output_path)).split("-")
        jobname = "".join([dir_root[-1]]+[dir_root[0]])
        
        # output path for the experiment log
        self.cmd_slurm = "sbatch --job-name {} --output {}/%N_%j.log".format(jobname, self.output_path)

        # special treatment for the "description" param (for convevience)
        # self.cmd_slurm += " --job-name {}".format(kwargs["description"])
        self.cmd_slurm += " cluster2.sh"
        # self.cmd_slurm = "python3 classifier.py "

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
    import generate_json_for_cluster as gen_dir
    config_dirs = gen_dir.config_dirs

    # config_dirs = [["exp1_path", "exp1_path1/exp1_config", "exp1_path1/exp1_config"],
    #                ["exp2_path", "exp1_path2/exp2_config", "exp1_path2/exp2_config"],
    #                ["exp3_path", "exp1_path3/exp3_config", "exp1_path3/exp3_config"],
    #                ["exp4_path", "exp1_path4/exp4_config", "exp1_path4/exp4_config"],
    #                ["exp5_path", "exp1_path5/exp5_config", "exp1_path5/exp5_config"]
    #                ]


    # Creating the flags to be passed to classifier.py
    cmds_to_sh = []
    for config_files in config_dirs:   #three arguments
        cmd_python = ""
        for k, v in zip(["output_path", "exp_config", "model_config"], [config_files[0], config_files[1], config_files[2]]):
            # _key_to_flag transforms "something_stupid"   into   "--something-stupid"
            flag = _key_to_flag(k)
            # _to_arg transforms ("--something-stupid", a_value)   into   "--something-stupid a_value"
            arg = _to_arg(flag, v)
            # arg = _to_arg(flag, os.path.join(root_exp, v))
            cmd_python += arg
        cmds_to_sh.append(cmd_python)
        # cmds_to_sh.append(cmd_python + " --output {}/%N_%j.log".format(config_files[0]))
        # cmd_python = "" --output {}/%N_%j.log".format(config_files[0])"


    commands = ''
    for cmds in cmds_to_sh:
        commands += "\"{}\" ".format(cmds)

    active_num_job = 8
    job_submit_mode = "sbatch_array"   #"sbatch_queue" # "srun_jobid" # , , True   #
    
    if job_submit_mode == "sbatch_array":
        # os.system("sbatch --output {}/%N_%j.log cluster_test.sh {}".format(config_files[0], commands))
        # os.system("sbatch --output {}/%N_%j.log --array 0-{}%{} cluster_test.sh {}".format(config_files[0], len(config_dirs), min(5, len(config_dirs)), commands))
        dir_root = os.path.basename(os.path.dirname(config_files[0])).split("-")
        jobname = "".join([dir_root[-1]] + ["-"] + [dir_root[0]])
        os.system("sbatch --job-name={} --mem={} --output {}/%N_%j.log --array 0-{}%{} cluster.sh {}".format(jobname, 7000, config_files[0], len(config_dirs), min(active_num_job, len(config_dirs)), commands))
        
    elif job_submit_mode == "sbatch_queue":
        for dirs in config_dirs:
            ClusterQueue(dirs)
            
    elif job_submit_mode == "srun_jobid":
        import numpy as np
        bash_jobids = [440644, 440645, 440648, 440744]#
        num_jobs_per_bash = np.int(np.ceil(len(config_dirs) / len(bash_jobids)))
        for jj, jobid in enumerate(bash_jobids):
            for config_files in config_dirs[jj*num_jobs_per_bash : min((jj+1)*num_jobs_per_bash, len(config_dirs))]:  # three arguments
    
                dir_root = os.path.basename(
                    os.path.dirname(config_files[0])).split("-")
                jobname = "".join([dir_root[-1]] + ["-"] + [dir_root[0]])
                cmd_python = ""
                for k, v in zip(["output_path", "exp_config", "model_config"],
                                [config_files[0], config_files[1], config_files[2]]):
                    # _key_to_flag transforms "something_stupid"   into   "--something-stupid"
                    flag = _key_to_flag(k)
                    # _to_arg transforms ("--something-stupid", a_value)   into   "--something-stupid a_value"
                    arg = _to_arg(flag, v)
                    # arg = _to_arg(flag, os.path.join(root_exp, v))
                    cmd_python += arg

                os.system("srun --jobid={} --ntasks=1 --job-name={} --error {}/{}.log python3 classifier.py {} > {}/{}.log &".format(jobid, jobname, config_files[0], jobid, cmd_python, config_files[0], jobid))
    
