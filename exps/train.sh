#!/bin/bash

model_name=flux-dev-fill-lora
train_data_root='configs/data/visualcloze.yaml'
gpu_num=8
node_num=1
batch_size=16
micro_batch_size=2
lr=1e-4
precision=bf16
lora_rank=256
snr_type=lognorm
training_type="lora"
grid_resolution=384
exp_name=visualcloze_${node_num}x${gpu_num}_bs${batch_size}_mbs${micro_batch_size}_rank${lora_rank}_lr${lr}_${grid_resolution}
results_dir=./output/${exp_name}
mkdir -p ${results_dir}

torchrun --nproc_per_node=${gpu_num} --nnodes=${node_num} --master_port 29339 train.py \
--global_bs ${batch_size} \
--micro_bs ${micro_batch_size} \
--data_path ${train_data_root} \
--results_dir ${results_dir} \
--lr ${lr} \
--grad_clip 2.0 \
--grid_resolution ${grid_resolution} \
--lora_rank ${lora_rank} \
--data_parallel fsdp \
--max_steps 1000000 \
--ckpt_every 5000 \
--log_every 1 \
--precision ${precision} \
--grad_precision fp32 \
--global_seed 20240826 \
--num_workers 4 \
--snr_type ${snr_type} \
--training_type ${training_type} \
--debug \
--load_t5 \
--load_clip \
--model_name ${model_name} \
--checkpointing
