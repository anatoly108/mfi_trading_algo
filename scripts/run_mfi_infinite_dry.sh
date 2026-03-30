#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

nohup python mfi_infinite_trade.py --usdt_amount 100 --dry-run --exchange binance --symbols BTCUSDT,ETHUSDT,SOLUSDT &
