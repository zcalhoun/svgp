#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=test_e2.out
#SBATCH --error=test_e2.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=carlsonlab
#SBATCH --mem=64G

source gp/bin/activate

python vnngp_fit.py \
    --data_directory /work/zdc6/weather_underground/durham/combined/ \
    --output /work/zdc6/exp2/test/ \
    --num_epochs 20 \
    --num_neighbors 64 \
    --year 2023 \
    --month 7 \
    --name TEST \
    