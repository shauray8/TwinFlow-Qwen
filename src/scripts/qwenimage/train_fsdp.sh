#!/bin/bash

export PYTHONPATH=./
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

CONFIG_PATH=configs/qwenimage_task/qwenimage_full.yaml
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nnodes=1 --nproc-per-node=4 \
  steerers/qwenimage/rlhf_fsdp.py $CONFIG_PATH