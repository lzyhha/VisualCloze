#!/bin/bash

export WANDB_API_KEY="0e1d81e87ba810ad562909026788da60d0b0e17b"
export HF_TOKEN="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"

gpu_num=8
node_num=1

model_name=flux-dev-fill-lora
train_data_root='configs/data/subject200k.yaml'
batch_size=64
micro_batch_size=1
lr=1e-4
precision=bf16
lora_rank=256
snr_type=lognorm
training_type="lora"
total_resolution=-1
grid_resolution=384
results_dir=/mnt/hwfile/alpha_vl/duruoyi/in_context_results_v6/
# exp_name=debug
exp_name=nxn_grid_fill_task-instruction_${node_num}x${gpu_num}_bs${batch_size}_mbs${micro_batch_size}_rank${lora_rank}_lr${lr}_res${total_resolution}_${grid_resolution}_object200k
# exp_name=nxn_grid_fill_${node_num}x${gpu_num}_mbs${micro_batch_size}_${training_type}_lr${lr}_1k_object200k-wo-depth
mkdir -p ${results_dir}"/"$exp_name

# unset NCCL_IB_HCA
#export TOKENIZERS_PARALLELISM=false

# p=Gvlab-S1-32
p=Omnilab
# p=lumina



srun -p ${p} --async --gres=gpu:${gpu_num} --cpus-per-task=64 -n${node_num} --ntasks-per-node=1 --quotatype=spot --job-name=in-${gpu_num}-${micro_batch_size}-fill \
torchrun --nproc_per_node=${gpu_num} --nnodes=${node_num} --master_port 29339 train.py \
    --master_port 18181 \
    --global_bs ${batch_size} \
    --micro_bs ${micro_batch_size} \
    --data_path ${train_data_root} \
    --results_dir ${results_dir}/${exp_name} \
    --lr ${lr} --grad_clip 2.0 \
    --total_resolution ${total_resolution} \
    --grid_resolution ${grid_resolution} \
    --lora_rank ${lora_rank} \
    --data_parallel fsdp \
    --max_steps 1000000 \
    --ckpt_every 1000 --log_every 1 \
    --precision ${precision} --grad_precision fp32 \
    --global_seed 20240826 \
    --num_workers 4 \
    --cache_data_on_disk \
    --snr_type ${snr_type} \
    --training_type ${training_type} \
    --debug \
    --load_t5 \
    --load_clip \
    --model_name ${model_name} \
    --checkpointing \
    # --masking_loss \
    # --checkpointing \
    # --use_model_ema 
    
# srun -p ${p} --async --gres=gpu:1 --cpus-per-task=4 -n1 --ntasks-per-node=1 --quotatype=spot --job-name=in-test \
# python -u train.py \
#     --master_port 18181 \
#     --global_bs ${batch_size} \
#     --micro_bs ${micro_batch_size} \
#     --data_path ${train_data_root} \
#     --results_dir ${results_dir}/${exp_name} \
#     --lr ${lr} --grad_clip 2.0 \
#     --resolution ${resolution} \
#     --lora_rank ${lora_rank} \
#     --data_parallel fsdp \
#     --max_steps 1000000 \
#     --ckpt_every 1000 --log_every 1 \
#     --precision ${precision} --grad_precision fp32 \
#     --global_seed 20240826 \
#     --num_workers 1 \
#     --cache_data_on_disk \
#     --snr_type ${snr_type} \
#     --training_type ${training_type} \
#     --checkpointing \
#     --debug \
#     --load_t5 \
#     --load_clip \
#     --model_name ${model_name} \
#     # --use_model_ema 