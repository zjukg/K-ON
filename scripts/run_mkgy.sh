#!/bin/bash
# export TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_LAUNCH_BLOCKING=1 
accelerate launch --gpu_ids 'all' --num_processes 8 --mixed_precision bf16 main.py -c config/MKG-Y.yaml > logs/train_mkgw.log