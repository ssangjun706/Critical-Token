#!/bin/bash

uv init -p 3.13
uv venv -p 3.13

source .venv/bin/activate

uv pip install torch transformers vllm accelerate matplotlib wandb omegaconf
uv pip install flash-attn --no-build-isolation
uv pip install ipykernel ipywidgets