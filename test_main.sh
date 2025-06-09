#!/bin/bash

#SBATCH --job-name=main
#SBATCH --array=0
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=./logs/output_%a.txt


source gp/bin/activate

python main.py \
    -i /work/zdc6/weather_underground/durham/combined/ \
    -o /work/zdc6/test_main/ \
    --num_epochs 2 \
    --variable tempAvg \
    --likelihood Student \
    --loss PLL
