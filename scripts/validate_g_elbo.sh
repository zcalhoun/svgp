#!/bin/bash

#SBATCH --job-name=g-elbo
#SBATCH --array=66-69
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=./logs_g_elbo/output_%a.txt
#SBATCH --error=./logs_g_elbo/output_%a.err
#SBATCH --time=00:10:00

source gp/bin/activate

python main.py \
    -i /work-old/zdc6/weather_underground/durham/combined/ \
    -o /work-old/zdc6/temp_g_elbo/ \
    --num_epochs 10 \
    --variable tempAvg \
    --likelihood Gaussian \
    --loss ELBO
