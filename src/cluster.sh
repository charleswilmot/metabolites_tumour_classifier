#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=700:00:00
#SBATCH --mem=20GB
#SBATCH --reservation triesch-shared
#SBATCH --partition=sleuths
#SBATCH --output=/home/mernst/git/titan/experiments/005_mnist_rcnn/files/slurm_output/mnist_rcnn_slurm_%j.out
##SBATCH --array=0-5%3
##SBATCH --gres=gpu:1

#config_array=("/home/mernst/git/titan/experiments/005_mnist_rcnn/files/config_files/config9.csv" "/home/mernst/git/titan/experiments/005_mnist_rcnn/files/config_files/config10.csv" "/home/mernst/git/titan/experiments/005_mnist_rcnn/files/config_files/config11.csv" "/home/mernst/git/titan/experiments/005_mnist_rcnn/files/config_files/config12.csv" "/home/mernst/git/titan/experiments/005_mnist_rcnn/files/config_files/config13.csv" "/home/mernst/git/titan/experiments/005_mnist_rcnn/files/config_files/config14.csv")

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
    echo "iteration $i"
    echo "job $j"
    srun python3 classifier.py ${config_array[$j]}
done

############### end ###################
#for i in "$@";
#  do
#    echo "----------------------------------------------------------"
#    echo cluster job $i
#    echo "11111111111111111111111111111111111111111111111111111111111"
#    output_path="$(get_output_folder "$i")"
#    echo "output_path: ${output_path}/$JOB_ID-$ARRAY_TASK_ID-$j.log"
#    echo "22222222222222222222222222222222222222222222222222222222222"
##    echo "python3 classifier.py $i > ${output_path}/$JOB_ID-$ARRAY_JOB_ID-$ARRAY_TASK_ID-${j}.log"
#    srun python3 classifier.py $i > "$output_path/$JOB_ID-$ARRAY_TASK_ID-${j}.log" 2>&1  &
#      # ^ don't wait for the job to finish, but continue the for loop instead
#  done
# --- end of experiment ---

