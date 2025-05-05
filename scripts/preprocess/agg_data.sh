#!/bin/bash

python aggregate_station_data.py \
    --input /work/zdc6/weather_underground/weather_underground/station_data/ \
    --output /work/zdc6/weather_underground/durham/wu2024.csv \
    --start_date 2024-01-01 \
    --end_date 2024-12-31 \
    --station_list /work/zdc6/weather_underground/durham/stations.csv \