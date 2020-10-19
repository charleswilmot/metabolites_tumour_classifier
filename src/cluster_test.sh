#!/usr/bin/env bash
#SBATCH --partition sleuths
#SBATCH --job-name metab
#SBATCH --reservation triesch-shared
##SBATCH --nodelist speedboat
#SBATCH --mincpus 2
#SBATCH --gres gpu:1

# parse the output folder from a set of arguments
# i.e. --output_path exp1_path1 --exp_config exp1_path1/exp1_config1 --model_config exp1_path1/exp1_config2
# becomes exp1_path1
# because that's the value after --output_path
function get_output_folder() {
    local config_array=($1)
    local n=0
    for a in ${config_array[@]};
    do
        if [ "$a" == "--output_path" ]; then
            local index=$((n+1))
            echo ${config_array[index]}
            return
        fi
        n=$(($n+1))
    done
}

for i in "$@";
  do
      echo cluster job $i
      output_path="$(get_output_folder "$i")"
#      echo parsed ${output_path}/output.log
      python3 classifier.py $i  > "${output_path}/output.log" 2>&1 &
      # ^ don't wait for the job to finish, but continue the for loop instead
  done


##SBATCH --partition sleuths
##SBATCH --job-name meta
##SBATCH --mem 4G
###SBATCH --reservation triesch-shared
##SBATCH --mincpus 2
##SBATCH --gres gpu:1
#
#srun -u python3 classifier.py test "$@"
