#!/bin/bash

#SBATCH --job-name=main
#SBATCH --array=0
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=./logs/output_%a.txt
#SBATCH --error=./logs/output_%a.err

source gp/bin/activate

python main.py \
    -i /work/zdc6/weather_underground/durham/combined/ \
    -o /work/zdc6/temperature_all/ \
    --num_epochs 10 \
    --variable tempAvg \
    --likelihood Student \
    --loss PLL
