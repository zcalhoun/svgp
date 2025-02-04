#!/bin/bash

#SBATCH --job-name=k256_3
#SBATCH --output=k256_3.out
#SBATCH --error=k256_3.err
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
    --file_path /cwork/zdc6/unc/data/data.csv \
    --k 256 \
    --batch_size 256 \
    --checkpoint_path /cwork/zdc6/unc/k256_3/model/ \
    --output_path /cwork/zdc6/unc/k256_3/ \
    --dataset UNC_periodic_covariates \
    --lr 0.01 \
    --epochs 200 \
    --noise_constraint 0.001 \
    --model BaseVNNGP_PeriodicFeatures_Covariates \
    --time_multiplier 20.0