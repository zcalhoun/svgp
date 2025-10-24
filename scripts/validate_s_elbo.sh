#!/bin/bash

#SBATCH --job-name=s_elbo
#SBATCH --array=0-71
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=./logs_s_elbo/output_%a.txt
#SBATCH --error=./logs_s_elbo/output_%a.err
#SBATCH --time=00:10:00

source gp/bin/activate

python main.py \
    -i /work/zdc6/weather_underground/durham/combined/ \
    -o /work/zdc6/temp_s_elbo/ \
    --num_epochs 10 \
    --variable tempAvg \
    --likelihood Student \
    --loss ELBO
