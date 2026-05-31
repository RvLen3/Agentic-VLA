#!/bin/bash
# train_fruit_vla.sh — X-VLA fine-tuning for the fruit-clearing task
# Usage: bash vla-scripts/train_fruit_vla.sh [NPZ_DATA_DIR] [NUM_GPUS] [FINETUNE_MODE]
#
# FINETUNE_MODE ∈ {soft_prompt, lora, full}  (default: lora — official recipe)
#   soft_prompt — freeze backbone, train only the embodiment soft prompts
#                 (smallest; tends to underfit a brand-new robot)
#   lora        — OFFICIAL X-VLA recipe: LoRA on all-linear + fully train the
#                 Action Expert (action_encoder/decoder) + soft prompts
#   full        — train everything (VLM layers at 1/10 LR)
#
# Small-data friendly: holds out 15% of EPISODES for validation + early stopping.
# Checkpoints:
#   runs/.../pth/XVLA_<mode>_step<step>_loss<loss>.pth   (periodic archival)
#   runs/.../pth/XVLA_<mode>_best.pth                    (lowest val loss)
#   runs/.../pth/XVLA_<mode>_final.pth                   (last step)
#   runs/...  (full HF model dir, exported at the end → AutoModel.from_pretrained)
#
# Model downloads are cached under ../.cache (set inside finetune_xvla.py).

NPZ_DATA_DIR=${1:-"raw_demos_left_third"}
NUM_GPUS=${2:-1}
FINETUNE_MODE=${3:-"lora"}

torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node $NUM_GPUS \
    vla-scripts/finetune_xvla.py \
    --pretrained_checkpoint "2toINF/X-VLA-Pt" \
    --npz_data_dir "$NPZ_DATA_DIR" \
    --finetune_mode "$FINETUNE_MODE" \
    --domain_id 3 \
    --num_actions 30 \
    --use_wrist_image True \
    --frame_stride 3 \
    --min_chunk_motion 0.01 \
    --learning_rate 3e-5 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --train_action_expert True \
    --freeze_steps 500 \
    --max_grad_norm 1.0 \
    --use_lr_schedule True \
    --warmup_steps 1000 \
    --use_cosine_decay True \
    --min_lr_ratio 0.1 \
    --batch_size 32 \
    --grad_accumulation_steps 1 \
    --num_workers 8 \
    --cache_episodes 6 \
    --max_steps 15000 \
    --save_steps 500 \
    --val_ratio 0.15 \
    --val_every_steps 500 \
    --early_stop_patience 10 \
    --run_root_dir "runs/fruit_vla" \
    --wandb_project "fruit-clearing-vla" \
    --run_id_note "fruit_clearing"
