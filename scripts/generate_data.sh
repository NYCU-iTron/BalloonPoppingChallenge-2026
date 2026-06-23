#!/bin/sh

uv run python3 scripts/generate_training_scenarios.py \
    --start 100 \
    --end 200 \
    --overwrite \
    --difficulty-start 0.2 \
    --difficulty-end 0.9 \
    --point-std 0.001 \
    --randomness 0.3 \
    --axis-weights 2 1 0.7 \
    --enable-gust
