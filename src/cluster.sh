#!/usr/bin/env sh
#SBATCH --partition sleuths
#SBATCH --mem 4G
#SBATCH --mincpus 2
##SBATCH --gres gpu:1
##SBATCH --reservation triesch-shared

srun -u python3 classifier.py train "$@"
