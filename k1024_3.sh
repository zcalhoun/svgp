#!/bin/bash

#SBATCH --job-name=k1024_3
#SBATCH --output=k1024_3.out
#SBATCH --error=k1024_3.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=carlsonlab
#SBATCH --mem=64G

source ~/.bashrc
conda activate gpytorch

python main.py \
    --file_path /datacommons/carlsonlab/zdc6/unc_data/data.csv \
    --k 1024 \
    --training_batch_size 16 \
    --checkpoint_path /cwork/zdc6/unc/k1024_3/model/ \
    --output_path /cwork/zdc6/unc/k1024_3/ \
    --dataset UNC_periodic \
    --lr 0.001 \
    --epochs 100 \
    --noise_constraint 0.01 \
    --model BaseVNNGP_PeriodicFeatures \
    --time_multiplier 100.0