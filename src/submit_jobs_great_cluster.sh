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
