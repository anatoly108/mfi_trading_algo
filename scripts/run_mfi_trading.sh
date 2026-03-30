#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

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


python binance_mfi_algo.py --config keys.yaml \
    --symbol FLUXUSDT \
    --quantity 100 \
    --dry-run &


python binance_mfi_algo.py --config keys.yaml \
    --symbol HARDUSDT \
    --quantity 75

python binance_mfi_algo.py --config keys.yaml \
    --symbol QKCUSDT \
    --quantity 1000


python binance_mfi_algo.py --config keys.yaml \
    --symbol FIOUSDT \
    --usdt_amount 10

python binance_mfi_algo.py --config keys.yaml \
    --symbol CYBERUSDT \
    --usdt_amount 10

python binance_mfi_algo.py --config keys.yaml \
    --symbol BURGERUSDT \
    --usdt_amount 10

python binance_mfi_algo.py --config keys.yaml \
    --symbol OOKIUSDT \
    --usdt_amount 10

python mfi_algo.py --config keys.yaml \
    --symbol VICUSDT \
    --usdt_amount 100 \
    --dry-run 
