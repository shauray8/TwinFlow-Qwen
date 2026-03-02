#!/bin/bash
export PYTHONPATH=./
export TOKENIZERS_PARALLELISM=false

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_IB_DISABLE=0
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2

export PYTORCH_CUDA_ALLOC_CONF=backend:native

export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

CONFIG_PATH=configs/qwenimage_task/qwenimage_full.yaml
torchrun --nnodes=1 --nproc-per-node=4 \
  steerers/qwenimage/rlhf_fsdp.py $CONFIG_PATH
