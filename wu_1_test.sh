#!/bin/bash

# source ~/.bashrc
# conda activate gp

python wu_1.py \
    --smoke_test \
    --file_path ./data/wu_simple/july2023.csv \
    --output_path ./test/wu_1/ \
    --log_level DEBUG \
    --epochs 10 \
    --num_neighbors 32 \
