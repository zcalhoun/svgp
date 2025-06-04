#!/bin/bash

#SBATCH --job-name=e10_6
#SBATCH --output=e10_6.out
#SBATCH --error=e10_6.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=carlsonlab
#SBATCH --mem=64G

source gp/bin/activate

python july24_experiments.py \
    --data_directory /work/zdc6/weather_underground/durham/july2024/ \
    --output /work/zdc6/exp10/periodic/1000/6 \
    --num_epochs 40 \
    --name e10_6p \
    --num_inducing_points 1000 \
    --batch_size 512 \
    --loss_function IW-PLL \
    --extra_cols evi t_s_avg \
    --model periodic \
    --lr 0.01