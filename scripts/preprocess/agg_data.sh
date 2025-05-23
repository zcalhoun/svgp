#!/bin/bash

python aggregate_station_data.py \
    --input /work/zdc6/weather_underground/weather_underground/station_data/ \
    --output /work/zdc6/weather_underground/july2021_nc.csv \
    --start_date 2021-07-01 \
    --end_date 2021-07-31 \
    --station_list /work/zdc6/weather_underground/nc_stations.csv \