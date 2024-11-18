#!/bin/bash

#SBATCH --job-name=unc_fit_2
#SBATCH --output=unc_fit_2.out
#SBATCH --error=unc_fit_2.err
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
    --k 64 \
    --training_batch_size 1024 \
    --checkpoint_path /cwork/zdc6/unc/k64_tb1024_periodic_features/model/ \
    --output_path /cwork/zdc6/unc/k64_tb1024_periodic_features/ \
    --dataset UNC_periodic \
    --lr 0.01 \
    --epochs 100 \
    --noise_constraint 0.1 \
    --model BaseVNNGP_PeriodicFeatures \