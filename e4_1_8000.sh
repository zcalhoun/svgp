#!/bin/bash

#SBATCH --job-name=e4_1_8000
#SBATCH --output=e4_1_8000.out
#SBATCH --error=e4_1_8000.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=carlsonlab
#SBATCH --mem=64G

source gp/bin/activate

python svgp_fit.py \
    --data_directory /work/zdc6/weather_underground/durham/combined/ \
    --output /work/zdc6/exp4/1/8000 \
    --num_epochs 100 \
    --num_inducing_points 8000 \
    --name e4_1_8000 \
    --batch_size 256 \
    --year 2023 \
    --month 7 \
    --lr 0.01
