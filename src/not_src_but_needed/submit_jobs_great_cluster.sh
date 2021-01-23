#!/usr/bin/env sh

#SBATCH --partition sleuths
#SBATCH --job-name metab
##SBATCH -t 1-200%20
#SBATCH --mem 9G
#SBATCH --reservation triesch-shared
#SBATCH --nodelist jetski
##SBATCH --nodelist speedboat
#SBATCH --mincpus 2
#SBATCH --gres gpu:1


srun -u python3 classifier.py "$@"


###################################
#! /bin/bash
#SBATCH --partition sleuths
#SBATCH --job-name metab
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
##SBATCH --reservation triesch-shared
##SBATCH --nodelist speedboat
##SBATCH --nodelist jetski
#SBATCH --mem 5GB
#SBATCH --time=1-00:00:00
##SBATCH --mail-type=END
##SBATCH --mail-user=elu@fias.uni-frankfurt.de
##SBATCH --gres gpu:1

# parse the output folder from a set of arguments
# because that's the value after --output_path

config_array=("$@")


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
j=$((SLURM_ARRAY_TASK_ID))
JOB_ID=$((SLURM_JOB_ID))
ARRAY_JOB_ID=$((SLURM_ARRAY_JOB_ID))
ARRAY_TASK_ID=$((SLURM_ARRAY_TASK_ID))

for i in `seq 1 1 1`
  do
    echo "job $j"
    echo "config_array[j]" ${config_array[$i]}
    output_path="$(get_output_folder "${config_array[$i]}")"
    echo "--error $output_path/$JOB_ID-$ARRAY_TASK_ID-${j}.log"
    echo "python classifier.py" ${config_array[$i]}
    echo "output log fn $output_path/$JOB_ID-$ARRAY_TASK_ID-${j}.log"
    srun --error "$output_path/$JOB_ID-$ARRAY_TASK_ID-${j}.log" python3 classifier.py ${config_array[$i]} > "$output_path/$JOB_ID-$ARRAY_TASK_ID-${j}.log" &
      # ^ don't wait for the job to finish, but continue the for loop instead
  done

