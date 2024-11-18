#!/bin/bash

#SBATCH --job-name=unc_fit
#SBATCH --output=unc_fit.out
#SBATCH --error=unc_fit.err
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
    --k 512 \
    --training_batch_size 64 \
    --checkpoint_path /cwork/zdc6/unc/k512_tb64/model/ \
    --output_path /cwork/zdc6/unc/k512_tb64/ \
    --lr 0.001