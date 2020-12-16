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

for i in "$@";
  do
    echo "----------------------------------------------------------"
    echo cluster job $i
    echo "11111111111111111111111111111111111111111111111111111111111"
    output_path="$(get_output_folder "$i")"
    echo "output_path: ${output_path}/$JOB_ID-$ARRAY_TASK_ID-$j.log"
    echo "22222222222222222222222222222222222222222222222222222222222"
#    echo "python3 classifier.py $i > ${output_path}/$JOB_ID-$ARRAY_JOB_ID-$ARRAY_TASK_ID-${j}.log"
    srun python3 classifier.py $i > "$output_path/$JOB_ID-$ARRAY_TASK_ID-${j}.log" 2>&1  &
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
