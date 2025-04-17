#!/bin/bash

#SBATCH --job-name=e3_1_64
#SBATCH --output=e3_1_64.out
#SBATCH --error=e3_1_64.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=carlsonlab
#SBATCH --mem=64G

source gp/bin/activate

python vnngp_fit.py \
    --data_directory /work/zdc6/weather_underground/durham/combined/ \
    --output /work/zdc6/exp3/1/64 \
    --num_epochs 100 \
    --n_neighbors 64 \
    --name e3_1_64 \
    --year 2023 \
    --month 7