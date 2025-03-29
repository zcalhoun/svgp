#!/bin/bash

python aggregate_station_data.py \
    --input /work/zdc6/weather_underground/weather_underground/station_data/ \
    --output /work/zdc6/weather_underground/durham/wu.csv \
    --start_date 2019-01-01 \
    --end_date 2023-12-01 \
    --station_list /work/zdc6/weather_underground/durham/stations.csv \
    --test