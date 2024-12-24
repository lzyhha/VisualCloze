#!/usr/bin/env sh

train_data_root='configs/data/16view.yaml'

batch_size=512
lr=2e-4
precision=bf16
image_size=1024
snr_type=lognorm

exp_name=32_16view_bs${batch_size}_lr${lr}_${precision}_${image_size}px_snr${snr_type}_noshift
mkdir -p results/"$exp_name"

# unset NCCL_IB_HCA
#export TOKENIZERS_PARALLELISM=false

python -u train_16view.py \
    --master_port 18181 \
    --global_bs 512 \
    --micro_bs 16 \
    --data_path ${train_data_root} \
    --results_dir results/${exp_name} \
    --lr ${lr} --grad_clip 2.0 \
    --data_parallel fsdp \
    --max_steps 1000000 \
    --ckpt_every 500 --log_every 10 \
    --precision ${precision} --grad_precision fp32 --qk_norm \
    --image_size ${image_size} \
    --global_seed 20240826 \
    --num_workers 8 \
    --cache_data_on_disk \
    --snr_type ${snr_type} \
    --checkpointing \
    --no_shift \
    2>&1 | tee -a results/"$exp_name"/output.log
