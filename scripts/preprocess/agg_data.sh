#!/bin/bash

#SBATCH --job-name=agg_data
#SBATCH --mem=64G
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./agg.out
#SBATCH --error=./agg.err

source ~/.bashrc
conda activate svgp


python aggregate_station_data.py \
    --input /hpc/group/carlsonlab/weather_underground/station_data/ \
    --output /work/zdc6/weather_underground/durham/wu.csv \
    --start_date 2019-01-01 \
    --end_date 2024-12-31 \
    --station_list /work/zdc6/weather_underground/durham/stations.csv \