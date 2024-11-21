#!/bin/bash

#SBATCH --job-name=k64_6
#SBATCH --output=k64_6.out
#SBATCH --error=k64_6.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=carlsonlab
#SBATCH --mem=64G

source ~/.bashrc
conda activate gpytorch

python -u main.py \
    --file_path /datacommons/carlsonlab/zdc6/unc_data/data.csv \
    --k 64 \
    --batch_size 1024 \
    --checkpoint_path /cwork/zdc6/unc/k64_6/model/ \
    --output_path /cwork/zdc6/unc/k64_6/ \
    --dataset UNC_periodic \
    --lr 0.01 \
    --epochs 200 \
    --noise_constraint 0.01 \
    --model BaseVNNGP_PeriodicFeatures \
    --time_multiplier 10.0