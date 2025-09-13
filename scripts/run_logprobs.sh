#!/bin/bash

source .venv/bin/activate

python collect_logprobs.py --config config/qwen_logprobs.yaml
