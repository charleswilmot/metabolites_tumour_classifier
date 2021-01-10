#!/usr/bin/env sh

#SBATCH --partition sleuths
#SBATCH --reservation triesch-shared
#SBATCH --nodelist jetski
##SBATCH --nodelist speedboat
#SBATCH --nodes 1
#SBATCH --mem 8GB
#SBATCH --mincpus 2
##SBATCH --gres gpu:1

#j=$((SLURM_ARRAY_TASK_ID))

srun python3 classifier.py "$@"

