#!/bin/bash

#SBATCH --job-name=hdd_cdd
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./cdd_hdd.txt
#SBATCH --error=./cdd_hdd.err

source ~/.bashrc
conda activate era

python calc_hdd_cdd.py \
    --ref_data ref_data = "/work/zdc6/temp_preds/" \
    --output /work/zdc6/weather_underground/durham/ 