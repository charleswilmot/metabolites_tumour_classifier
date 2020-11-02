#!/usr/bin/env sh

#SBATCH --partition sleuths
##SBATCH --reservation triesch-shared
##SBATCH --nodelist jetski
##SBATCH --nodelist speedboat
##SBATCH --nodes 1
##SBATCH -n 10
##SBATCH --ntasks-per-node 32
##SBATCH --cpus-per-task 1
#SBATCH --mem 9GB
#SBATCH --mincpus 2
#SBATCH --job-name rnn
##SBATCH --gres gpu:1

#j=$((SLURM_ARRAY_TASK_ID))

srun python3 classifier.py "$@"
#for i in "$@";
#  do
#      echo "cluster job $j: $i"
#      srun python3 classifier.py $i
#  done

