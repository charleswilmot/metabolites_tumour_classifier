#!/usr/bin/env sh

#SBATCH --partition sleuths
##SBATCH --nodes 1
##SBATCH --ntasks-per-node 32
##SBATCH --cpus-per-task 1
#SBATCH --mem 20GB
#SBATCH --job-name metab
#SBATCH --reservation triesch-shared
##SBATCH --nodelist jetski
#SBATCH --nodelist speedboat
#SBATCH --mincpus 2
##SBATCH --gres gpu:1

j=$((SLURM_ARRAY_TASK_ID))

for i in "$@";
  do
      echo "$j: $i"
      srun python3 classifier.py $i
  done

