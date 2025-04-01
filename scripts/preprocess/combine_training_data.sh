#!/bin/bash

#SBATCH --job-name=combine_training_data
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./combine.txt
#SBATCH --error=./combine.err

source ~/.bashrc
conda activate era

python combine_training_data.py \
    --era_dir /work/zdc6/weather_underground/durham/era_by_month/ \
    --wu_file /work/zdc6/weather_underground/durham/wu.csv \
    --nlcd_file /work/zdc6/weather_underground/durham/nlcd/pca_train_354m_buffer.csv \
    --destination_dir /work/zdc6/weather_underground/durham/combined/ \