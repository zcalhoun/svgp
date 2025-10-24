#!/bin/bash

#SBATCH --job-name=main
#SBATCH --array=0-71
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=./logs/output_%a.txt
#SBATCH --error=./logs/output_%a.err
#SBATCH --time=00:30:00

source gp/bin/activate

python main.py \
    -i /work-old/zdc6/weather_underground/durham/combined/ \
    -o /work-old/zdc6/dewpoint_preds/ \
    --num_epochs 10 \
    --variable dewptAvg \
    --likelihood Student \
    --loss PLL \
    --upper_alpha 0.95 \
    --lower_alpha 0.05 \
    --train_size 1.0 \
    --ref_data /work-old/zdc6/era5_tabular/
