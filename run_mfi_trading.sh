#!/bin/bash

python binance_mfi_algo.py --config keys.yaml \
    --symbol HARDUSDT \
    --quantity 100 \
    --dry-run &

sleep 2

python binance_mfi_algo.py --config keys.yaml \
    --symbol VITEUSDT \
    --quantity 500 \
    --dry-run &

sleep 2

python binance_mfi_algo.py --config keys.yaml \
    --symbol IRISUSDT \
    --quantity 1000 \
    --dry-run &

sleep 2

python binance_mfi_algo.py --config keys.yaml \
    --symbol AKROUSDT \
    --quantity 10000 \
    --dry-run &
