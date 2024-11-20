#!/bin/bash

#SBATCH --job-name=k1024_4
#SBATCH --output=k1024_4.out
#SBATCH --error=k1024_4.err
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
    --batch_size 16 \
    --checkpoint_path /cwork/zdc6/unc/k1024_4/model/ \
    --output_path /cwork/zdc6/unc/k1024_4/ \
    --dataset UNC_periodic_time \
    --lr 0.0001 \
    --epochs 100 \
    --noise_constraint 0.01 \
    --model BaseVNNGP_PeriodicFeatures_time \
    --time_multiplier 100.0