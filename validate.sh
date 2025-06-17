#!/bin/bash

#SBATCH --job-name=main
#SBATCH --array=5,13,14,15,16,19,24,27,28,31,37,39
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=./logs/output_%a.txt
#SBATCH --error=./logs/output_%a.err
#SBATCH --time=00:10:00

source gp/bin/activate

python main.py \
    -i /work/zdc6/weather_underground/durham/combined/ \
    -o /work/zdc6/temp_stud_pll/ \
    --num_epochs 10 \
    --variable tempAvg \
    --likelihood Student \
    --loss PLL
