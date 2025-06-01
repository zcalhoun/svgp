#!/bin/bash

#SBATCH --job-name=e10_5_100
#SBATCH --output=e10_5_100.out
#SBATCH --error=e10_5_100.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=carlsonlab
#SBATCH --mem=64G

source gp/bin/activate

python july24_experiments.py \
    --data_directory /work/zdc6/weather_underground/durham/july2024/ \
    --output /work/zdc6/exp10/periodic/100/5 \
    --num_epochs 100 \
    --name e10_5p \
    --num_inducing_points 100 \
    --batch_size 512 \
    --likelihood Student \
    --loss_function IW-PLL \
    --extra_cols evi t_s_avg \
    --model periodic \
    --lr 0.01