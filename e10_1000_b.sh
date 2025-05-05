#!/bin/bash

#SBATCH --job-name=e10_base1000
#SBATCH --output=e10_base1000.out
#SBATCH --error=e10_base1000.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=carlsonlab
#SBATCH --mem=64G

source gp/bin/activate

python july24_experiments.py \
    --data_directory /work/zdc6/weather_underground/durham/july2024/ \
    --output /work/zdc6/exp10/base/1000 \
    --num_epochs 200 \
    --name e10_base \
    --num_inducing_points 1000 \
    --batch_size 512 \
    --lr 0.1