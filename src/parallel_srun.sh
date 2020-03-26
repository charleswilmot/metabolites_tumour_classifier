#!/bin/bash

#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --partition=sleuths
#SBATCH -w jetski
#SBATCH -N 1 # 3 nodes
#SBATCH -n 1 # 5 tasks
#SBATCH -c 1 # 1 core per task
#SBATCH --mem=4G
# Execute jobs in parallel
srun -N 1 -n 1 python3 classifier.py train 0 &
srun -N 1 -n 1 python3 classifier.py train 1 &
srun -N 1 -n 1 python3 classifier.py train 2 &