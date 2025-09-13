#!/bin/bash

source .venv/bin/activate

python collect_reasoning_trace.py \
    --config=config/qwen_reasoning_trace.yaml