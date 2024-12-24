#!/usr/bin/env sh

export HF_TOKEN="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"

# Lumina-Next supports any resolution (up to 2K)
res="1024"
# res="1536"
# res="2048"
t=1
guidance_scale=30.0
seed=25
steps=30
solver=euler
train_steps=0019000
lora_rank=256
grid_resolution=384
total_resolution=-1
model_name=flux-dev-fill-lora
training_type="lora"
# exp_name=nxn_grid_fill_task-instruction_1x8_mbs8_${training_type}_lr1e-4_1k_object200k-wo-depth
# exp_name=nxn_grid_fill_task-instruction_1x8_bs8_mbs1_rank256_lr1e-4_res1024_512_object200k-wo-depth
# nxn_grid_fill_task-instruction_1x8_mbs8_rank256_lr1e-4_object200k-filtered-wo-depth
# nxn_grid_fill_task-instruction_1x8_mbs8_rank256_lr1e-4_fixed-res_object200k-wo-depth

# exp_name=nxn_grid_fill_task-instruction_1x8_mbs8_rank256_lr1e-4_1k_object200k-wo-depth # 18000
# exp_name=nxn_grid_fill_task-instruction_1x8_mbs8_rank256_lr1e-4_fixed-res_object200k-wo-depth # 10000
# exp_name=nxn_grid_fill_1x8_mbs4_full_model_lr1e-4_1k_object200k-wo-depth # 13000
# exp_name=nxn_grid_fill_1x8_mbs4_rank256_lr1e-4_1k_object200k-wo-depth # 12000

# exp_name=nxn_grid_fill_task-instruction_1x8_bs8_mbs1_rank256_lr1e-4_res1024_512_object200k-wo-depth # 113000
# exp_name=nxn_grid_fill_task-instruction_1x8_bs64_mbs1_rank256_lr1e-4_res-1_512_object200k-wo-depth # 15000
# exp_name=nxn_grid_fill_task-instruction_1x8_bs64_mbs1_rank256_lr1e-4_res1024_512_object200k-wo-depth # 27000

# exp_name=nxn_grid_fill_task-instruction_1x8_bs64_mbs1_rank256_lr1e-4_res1024_512_object200k-wo-depth # 47000
# exp_name=nxn_grid_fill_task-instruction_1x8_bs64_mbs1_rank256_lr1e-4_res-1_512_object200k-wo-depth # 15000
exp_name=nxn_grid_fill_task-instruction_1x8_bs64_mbs1_rank256_lr1e-4_res-1_384_object200k # 19000

model_dir=/mnt/hwfile/alpha_vl/duruoyi/in_context_results_v6/${exp_name}/checkpoints/${train_steps}
cap_dir=/mnt/hwfile/alpha_vl/duruoyi/subjects200k/metadata_test_v6.json
out_dir=samples_test_v6/${exp_name}_cfg${guidance_scale}_shift${t}_steps${steps}_seed${seed}_ckpt${train_steps}

# cap_dir=/mnt/hwfile/alpha_vl/lizhongyu/in_context_control_images/ood_task/easydrawingguides/easydrawingguides.jsonl
# cap_dir=/mnt/hwfile/alpha_vl/lizhongyu/in_context_control_images/ood_task/9x9_face/9x9_face_bottom.jsonl
# out_dir=samples_test_v6_ood/${exp_name}_cfg${guidance_scale}_shift${t}_steps${steps}_seed${seed}_ckpt${train_steps}_9x9_face_bottom_target_w-content-prompt
# cap_dir=/mnt/hwfile/alpha_vl/lizhongyu/in_context_control_images/ood_task/Yangguang-obj_images_16view/Yangguang-obj_images_16view.jsonl
# out_dir=samples_test_v6_ood/${exp_name}_cfg${guidance_scale}_shift${t}_steps${steps}_seed${seed}_ckpt${train_steps}_16view_target_w-content-prompt
# cap_dir=/mnt/hwfile/alpha_vl/lizhongyu/in_context_control_images/ood_task/iteractive_editing/subjects200k-00000-of-00022.jsonl
# out_dir=samples_test_v6_ood/${exp_name}_cfg${guidance_scale}_shift${t}_steps${steps}_seed${seed}_ckpt${train_steps}_iteractive_editing_111_target_wo-content-prompt
# cap_dir=/mnt/hwfile/alpha_vl/lizhongyu/in_context_control_images/ood_task/iclight/iclight.jsonl
# out_dir=samples_test_v6_ood/${exp_name}_cfg${guidance_scale}_shift${t}_steps${steps}_seed${seed}_ckpt${train_steps}_iclight_target_wo-content-prompt_order2

# p=lumina
# p=Gvlab-S1-32
p=Omnilab

srun -p ${p} --async --gres=gpu:1 --cpus-per-task=4 -n1 --ntasks-per-node=1 --quotatype=spot --job-name=in-sample \
python -u sample_fill.py --ckpt ${model_dir} \
--image_save_path ${out_dir} \
--solver ${solver} --num_sampling_steps ${steps} \
--caption_path ${cap_dir} \
--seed ${seed} \
--resolution ${res} \
--time_shifting_factor ${t} \
--guidance_scale ${guidance_scale} \
--batch_size 1 \
--model_name ${model_name} \
--lora_rank ${lora_rank} \
--grid_resolution ${grid_resolution} \
--total_resolution ${total_resolution} \
# --ema \
# --debug \
# --do_classifier_free_guidance \
# --debug \
