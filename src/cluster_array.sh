#!/bin/bash
#
#SBATCH --partition=sleuths
#SBATCH --reservation triesch-shared
##SBATCH --nodelist jetski
##SBATCH --nodelist speedboat
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=4
##SBATCH --mem=9GB
##SBATCH --gres=gpu:1


config_array=("$@")

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

