#!/bin/bash
#SBATCH -p gpu  # Partition
#SBATCH --gpus=1  # Number of GPUs
#SBATCH -c 4  # Number of CPU cores
#SBATCH --mem=98GB  # Requested Memory
#SBATCH -t 0-01:00:00  # Zero day, one hour
#SBATCH --constraint=vram48
#SBATCH --mail-type=ALL
#SBATCH -o evalmetrics.out  # Specify where to save terminal output, %j = job ID will be filled by slurm

module load conda/latest
conda activate 696hw1
python ./combinedeval.py
