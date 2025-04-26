#!/bin/bash

#SBATCH --job-name=e9_1000_1
#SBATCH --output=e9_1000_1.out
#SBATCH --error=e9_1000_1.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=carlsonlab
#SBATCH --mem=64G

source gp/bin/activate

python svgp_fit.py \
    --data_directory /work/zdc6/weather_underground/durham/combined/ \
    --output /work/zdc6/exp9/1000/1/ \
    --num_epochs 200 \
    --name e9_1000_1 \
    --year 2023 \
    --month 7 \
    --likelihood Student \
    --num_inducing_points 1000 \
    --batch_size 512 \
    --lr 0.01