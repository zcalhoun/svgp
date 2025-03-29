#!/bin/bash

python interp_era.py \
    --input /work/zdc6/era5/ \
    --output /work/zdc6/weather_underground/durham/era_by_station \
    --station_list /work/zdc6/weather_underground/durham/stations.csv \
    --test