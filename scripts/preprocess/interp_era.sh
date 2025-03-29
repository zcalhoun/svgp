#!/bin/bash

#SBATCH --job-name=aggregate_era
#SBATCH --array=0-2         # Adjust this based on number of files - 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/output_%A_%a.txt

source ~/.bashrc
conda activate era

python test_interp_era.py \
    --input /work/zdc6/era5/ \
    --output /work/zdc6/weather_underground/durham/era_by_month/ \
    --station_list /work/zdc6/weather_underground/durham/stations.csv \
    --test