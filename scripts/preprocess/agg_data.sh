#!/bin/bash

python aggregate_station_data.py \
    --input /work/zdc6/weather_underground/weather_underground/station_data/ \
    --output /work/zdc6/weather_underground/july142024.csv \
    --start_date 2024-07-14 \
    --end_date 2024-07-14 \
    --station_list /work/zdc6/weather_underground/nc_stations.csv \