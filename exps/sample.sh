#!/usr/bin/env sh

export HF_TOKEN="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"

# Lumina-Next supports any resolution (up to 2K)
res="768:1024x512"
t=1
guidance_scale=4.0
seed=25
steps=30
solver=euler
train_steps=0005000
model_name=flux-dev-lora
model_dir=/mnt/hwfile/alpha_vl/duruoyi/in_context_results/2x1_grid_8_mbs8_rank128_uniform_lige/checkpoints/${train_steps}
cap_dir=lige_test.json
out_dir=samples/2x1_grid_n8_mbs4_cfg${guidance_scale}_steps${steps}_seed${seed}_ckpt${train_steps}_uniform_lige

p=lumina
# p=Gvlab-S1-32
# p=Omnilab

srun -p ${p} --async --gres=gpu:1 --cpus-per-task=4 -n1 --ntasks-per-node=1 --quotatype=spot --job-name=in-sample \
python -u sample.py --ckpt ${model_dir} \
--image_save_path ${out_dir} \
--solver ${solver} --num_sampling_steps ${steps} \
--caption_path ${cap_dir} \
--seed ${seed} \
--resolution ${res} \
--time_shifting_factor ${t} \
--guidance_scale ${guidance_scale} \
--batch_size 1 \
--model_name ${model_name} \
# --do_classifier_free_guidance \
# --debug \