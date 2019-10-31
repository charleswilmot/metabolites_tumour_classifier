#!/usr/bin/env sh
#SBATCH --partition sleuths
#SBATCH --reservation triesch-shared
#SBATCH --mem 4G
#SBATCH --gres gpu:1

srun -u python3 classifier.py train "$@"
