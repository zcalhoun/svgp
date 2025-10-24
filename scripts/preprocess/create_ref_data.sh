#!/bin/bash

#SBATCH --job-name=create_ref_data
#SBATCH --array=0-71         # Adjust this based on number of files - 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/ref_%a.txt
#SBATCH --mem=64GB

source ~/.bashrc
conda activate era

python create_ref_data.py \
    --input /work-old/zdc6/era5/ \
    --output /work-old/zdc6/era5_tabular/ \
    --nlcd /work-old/zdc6/weather_underground/durham/nlcd/pca_test_500m_buffer.csv