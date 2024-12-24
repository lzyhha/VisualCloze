#!/usr/bin/env sh

export HF_TOKEN="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"
export HF_HOME="/data/huggingface"


train_data_root='configs/data/2M.yaml'
res="1024:1024x1024"
t=1
txt_cfg_list="20.0,19.0,18.0,17.0,16.0,15.0,14.0,13.0,12.0,11.0,10.0,9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0,0.0"
seed=25
steps=30
solver=euler
train_steps=0008000
model_dir=/data/DiffEntropy/flux-dedisill/results/1024_lr_1e-4_bsz_32_huber_lognorm_gt/checkpoints/${train_steps}
cap_dir=validation_data.json
out_dir=samples/inter_seed${seed}
root_dir=/goosedata/images

CUDA_VISIBLE_DEVICES=5 python -u infer.py --ckpt ${model_dir} \
--data_path ${train_data_root} \
--image_save_path ${out_dir} \
--solver ${solver} --num_sampling_steps ${steps} \
--caption_path ${cap_dir} \
--seed ${seed} \
--resolution ${res} \
--time_shifting_factor ${t} \
--txt_cfg_list ${txt_cfg_list} \
--batch_size 1 \
--cache_data_on_disk \
--debug \
