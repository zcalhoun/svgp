#!/bin/bash

#SBATCH --job-name=k256_1
#SBATCH --output=k256_1.out
#SBATCH --error=k256_1.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=carlsonlab
#SBATCH --mem=64G

source ~/.bashrc
conda activate gpytorch

export PYTHONUNBUFFERED=1

python -u main.py \
    --file_path /datacommons/carlsonlab/zdc6/unc_data/data.csv \
    --k 256 \
    --batch_size 256 \
    --checkpoint_path /cwork/zdc6/unc/k256_1/model/ \
    --output_path /cwork/zdc6/unc/k256_1/ \
    --dataset UNC_periodic \
    --lr 0.01 \
    --epochs 200 \
    --noise_constraint 0.001 \
    --model BaseVNNGP_PeriodicFeatures \
    --time_multiplier 20.0