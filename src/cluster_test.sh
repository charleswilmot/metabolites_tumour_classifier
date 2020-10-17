#!/usr/bin/env sh

#SBATCH --partition sleuths
#SBATCH --job-name meta
#SBATCH --mem 4G
##SBATCH --reservation triesch-shared
#SBATCH --mincpus 2
#SBATCH --gres gpu:1

srun -u python3 classifier.py test "$@"
