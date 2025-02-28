#!/bin/bash

#SBATCH --job-name=wu_1_k256
#SBATCH --output=wu_1_k256.out
#SBATCH --error=wu_1_k256.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=carlsonlab
#SBATCH --mem=64G

source ~/.bashrc
conda activate gpytorch

python wu_1.py \
    --smoke_test \
    --file_path /work/zdc6/spat_temp/data/july2023.csv \
    --output_path /work/zdc6/spat_temp/exp1/k256 \
    --epochs 100 \
    --num_neighbors 256 \
