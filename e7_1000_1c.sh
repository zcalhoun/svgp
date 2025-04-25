#!/bin/bash

#SBATCH --job-name=e7_1000_1c
#SBATCH --output=e7_1000_1c.out
#SBATCH --error=e7_1000_1c.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=carlsonlab
#SBATCH --mem=64G

source gp/bin/activate

python svgp_fit.py \
    --data_directory /work/zdc6/weather_underground/durham/combined/ \
    --output /work/zdc6/exp7/1000/1c/ \
    --num_epochs 100 \
    --name e7_1000_1c \
    --year 2023 \
    --month 7 \
    --likelihood Student \
    --num_inducing_points 1000 \
    --batch_size 512 \
    --lr 0.01