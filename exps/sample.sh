#!/usr/bin/env sh

guidance_scale=30.0
seed=0
steps=30
solver=euler
train_steps=0010000
lora_rank=256
resolution=384
model_name=flux-dev-fill-lora

exp_name=visualcloze_1x8_bs16_mbs2_rank256_lr1e-4_384
model_path=output/${exp_name}/checkpoints/${train_steps}/consolidated.00-of-01.pth
data_path=dataset/test/data.json
output_path=output/${exp_name}/samples

python -u sample.py --model_path ${model_path} \
--image_save_path ${output_path} \
--solver ${solver} --num_sampling_steps ${steps} \
--data_path ${data_path} \
--seed ${seed} \
--guidance_scale ${guidance_scale} \
--batch_size 1 \
--model_name ${model_name} \
--lora_rank ${lora_rank} \
--resolution ${resolution}