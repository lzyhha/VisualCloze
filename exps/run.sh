#!/bin/bash

export WANDB_API_KEY="0e1d81e87ba810ad562909026788da60d0b0e17b"
export HF_TOKEN="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"

gpu_num=1

model_name=flux-dev-lora
train_data_root='configs/data/qinqi.yaml'
batch_size=4
micro_batch_size=4
lr=1e-4
precision=bf16
lora_rank=128
snr_type=lognorm
results_dir=/mnt/hwfile/alpha_vl/duruoyi/in_context_results/
exp_name=record_loss_lognorm
mkdir -p ${results_dir}"/"$exp_name

# unset NCCL_IB_HCA
#export TOKENIZERS_PARALLELISM=false

# p=Gvlab-S1-32
p=Omnilab
# p=lumina



# srun -p ${p} --async --gres=gpu:${gpu_num} --cpus-per-task=32 -n1 --ntasks-per-node=1 --quotatype=reserved --job-name=in-${gpu_num}-${micro_batch_size}-fill \
# torchrun --nproc_per_node=${gpu_num} --nnodes=1 --master_port 29339 train.py \
#     --master_port 18181 \
#     --global_bs ${batch_size} \
#     --micro_bs ${micro_batch_size} \
#     --data_path ${train_data_root} \
#     --results_dir ${results_dir}/${exp_name} \
#     --lr ${lr} --grad_clip 2.0 \
#     --lora_rank ${lora_rank} \
#     --data_parallel fsdp \
#     --max_steps 1000000 \
#     --ckpt_every 1000 --log_every 1 \
#     --precision ${precision} --grad_precision fp32 \
#     --global_seed 20240826 \
#     --num_workers 1 \
#     --cache_data_on_disk \
#     --snr_type ${snr_type} \
#     --checkpointing \
#     --debug \
#     --load_t5 \
#     --load_clip \
#     --model_name ${model_name}
    
srun -p ${p} --async --gres=gpu:1 --cpus-per-task=4 -n1 --ntasks-per-node=1 --quotatype=spot --job-name=in-test \
python -u train_record.py \
    --master_port 18181 \
    --global_bs ${batch_size} \
    --micro_bs ${micro_batch_size} \
    --data_path ${train_data_root} \
    --results_dir ${results_dir}/${exp_name} \
    --lr ${lr} --grad_clip 2.0 \
    --data_parallel fsdp \
    --max_steps 1000000 \
    --ckpt_every 1000 --log_every 1 \
    --precision ${precision} --grad_precision fp32 \
    --global_seed 20240826 \
    --num_workers 1 \
    --cache_data_on_disk \
    --snr_type ${snr_type} \
    --checkpointing \
    --debug \
    --load_t5 \
    --load_clip \
    --model_name ${model_name}