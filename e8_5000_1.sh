#!/bin/bash

#SBATCH --job-name=e8_5000_1
#SBATCH --output=e8_5000_1.out
#SBATCH --error=e8_5000_1.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=carlsonlab
#SBATCH --mem=64G

source gp/bin/activate

python svgp_fit.py \
    --data_directory /work/zdc6/weather_underground/durham/combined/ \
    --output /work/zdc6/exp8/5000/1/ \
    --num_epochs 100 \
    --name e8_5000_1 \
    --year 2023 \
    --month 7 \
    --likelihood Student \
    --loss_function PLL \
    --num_inducing_points 5000 \
    --batch_size 4096 \
    --lr 0.1 \
    --inducing_points_method random_train