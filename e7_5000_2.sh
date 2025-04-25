#!/bin/bash

#SBATCH --job-name=e7_5000_2
#SBATCH --output=e7_5000_2.out
#SBATCH --error=e7_5000_2.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=carlsonlab
#SBATCH --mem=64G

source gp/bin/activate

python svgp_fit.py \
    --data_directory /work/zdc6/weather_underground/durham/combined/ \
    --output /work/zdc6/exp7/5000/2 \
    --num_epochs 100 \
    --name e7_5000_2 \
    --year 2023 \
    --month 7 \
    --loss_function PLL \
    --num_inducing_points 5000 \
    --batch_size 4096 \
    --lr 0.1