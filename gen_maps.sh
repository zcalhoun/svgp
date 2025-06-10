#!/bin/bash

#SBATCH --job-name=main
#SBATCH --array=0
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=./logs/map_%a.txt
#SBATCH --error=./logs/map_%a.err

source gp/bin/activate

python main.py \
    -i /work/zdc6/weather_underground/durham/combined/ \
    -o /work/zdc6/temp_preds/ \
    --num_epochs 10 \
    --variable tempAvg \
    --likelihood Student \
    --loss PLL \
    --train_size 1.0 \
    --ref_data /work/zdc6/era5_tabular/
