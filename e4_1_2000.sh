#!/bin/bash

#SBATCH --job-name=e4_1_2000
#SBATCH --output=e4_1_2000.out
#SBATCH --error=e4_1_2000.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=carlsonlab
#SBATCH --mem=64G

source gp/bin/activate

python svgp_fit.py \
    --data_directory /work/zdc6/weather_underground/durham/combined/ \
    --output /work/zdc6/exp4/1/2000 \
    --num_epochs 100 \
    --num_inducing_points 2000 \
    --name e4_1_2000 \
    --batch_size 4096 \
    --year 2023 \
    --month 7 \
    --lr 0.1
