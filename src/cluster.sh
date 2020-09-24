#!/usr/bin/env sh

#SBATCH --partition=sleuths
#SBATCH --job-name=metab
#SBATCH --mem=9G
#SBATCH --reservation=triesch-shared
#SBATCH --mincpus=2
#SBATCH --gres=gpu:1
#SBATCH --array=1-5



srun -u python3 classifier.py train "$@"

#echo "$SLURM_ARRAY_TASK_ID"