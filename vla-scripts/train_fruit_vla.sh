#!/bin/bash
# train_fruit_vla.sh — X-VLA LoRA fine-tuning for fruit-clearing task
# Usage: bash vla-scripts/train_fruit_vla.sh [NPZ_DATA_DIR] [NUM_GPUS]
#
# Requirements: 5.1, 5.2, 5.3, 5.4
# domain_id=5 (fruit-clearing task, distinct from LIBERO domain_id=3)
# lora_rank=32, learning_rate=2e-4, batch_size=8, grad_accumulation_steps=4
# max_steps=20000, save_steps=500, image_aug=True

NPZ_DATA_DIR=${1:-"raw_demos/mixed"}   # mixed atomic_ops + full_episodes (3:1)
NUM_GPUS=${2:-4}

torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node $NUM_GPUS \
    vla-scripts/finetune_xvla.py \
    --pretrained_checkpoint "HuggingFaceM4/xvla-7b" \
    --npz_data_dir "$NPZ_DATA_DIR" \
    --domain_id 5 \
    --lora_rank 32 \
    --learning_rate 2e-4 \
    --batch_size 8 \
    --grad_accumulation_steps 4 \
    --max_steps 20000 \
    --save_steps 500 \
    --image_aug True \
    --run_root_dir "runs/fruit_vla" \
    --wandb_project "fruit-clearing-vla" \
    --run_id_note "fruit_clearing_domain5"
