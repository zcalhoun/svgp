#!/bin/bash

#SBATCH --job-name=k64_1
#SBATCH --output=k64_1.out
#SBATCH --error=k64_1.err
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
    --batch_size 1024 \
    --checkpoint_path /cwork/zdc6/unc/k64_1/model/ \
    --output_path /cwork/zdc6/unc/k64_1/ \
    --dataset UNC_periodic_time \
    --lr 0.01 \
    --epochs 100 \
    --noise_constraint 0.1 \
    --model BaseVNNGP_PeriodicFeatures_time \
    --time_multiplier 20.0