#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=100GB  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 01:00:00  # Job time limit
#SBATCH --mail-type=ALL
#SBATCH -o output-%j.out  # %j = job ID


module load conda/latest
conda activate project_19
python -m eval.test_eval