#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

python mfi_analysis.py --symbols DARUSDT --no_vol_threshold --exchange binance --now 2024_09_08__14_59
