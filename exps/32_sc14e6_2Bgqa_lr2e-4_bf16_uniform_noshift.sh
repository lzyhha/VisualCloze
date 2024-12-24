#!/usr/bin/env sh

train_data_root='configs/data/sc14e6.yaml'

model=NextDiT_2B_GQA_patch2
lr=2e-4
precision=bf16
snr_type=uniform

exp_name=32_sc14e6_${model}_lr${lr}_${precision}_snr${snr_type}_noshift
mkdir -p results/"$exp_name"

# unset NCCL_IB_HCA
#export TOKENIZERS_PARALLELISM=false

python -u train.py \
    --master_port 18181 \
    --global_bsz_512 512 \
    --micro_bsz_512 16 \
    --model ${model} \
    --data_path ${train_data_root} \
    --results_dir results/${exp_name} \
    --lr ${lr} --grad_clip 2.0 \
    --data_parallel fsdp \
    --max_steps 1000000 \
    --ckpt_every 1000 --log_every 10 \
    --precision ${precision} --grad_precision fp32 --qk_norm \
    --global_seed 20240826 \
    --num_workers 4 \
    --cache_data_on_disk \
    --snr_type ${snr_type} \
    --checkpointing \
    --no_shift \
    2>&1 | tee -a results/"$exp_name"/output.log
