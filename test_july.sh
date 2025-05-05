#!/bin/bash

#SBATCH --job-name=e10_base
#SBATCH --output=e10_base.out
#SBATCH --error=e10_base.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=carlsonlab
#SBATCH --mem=64G

source gp/bin/activate

python july24_experiments.py \
    --data_directory /work/zdc6/weather_underground/durham/july2024/ \
    --output /work/zdc6/exp10/test/ \
    --num_epochs 20 \
    --name e10_test \
    --num_inducing_points 500 \
    --batch_size 1024 \
    --lr 0.01