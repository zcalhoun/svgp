#!/bin/bash

#SBATCH --job-name=e5_2
#SBATCH --output=e5_2.out
#SBATCH --error=e5_2.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=carlsonlab
#SBATCH --mem=64G

source gp/bin/activate

python vnngp_fit.py \
    --data_directory /work/zdc6/weather_underground/durham/combined/ \
    --output /work/zdc6/exp5/2/ \
    --num_epochs 100 \
    --n_neighbors 64 \
    --name e5_2 \
    --year 2023 \
    --month 7 \
    --likelihood Student