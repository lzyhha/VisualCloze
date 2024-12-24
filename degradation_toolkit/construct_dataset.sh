#!/usr/bin/env sh

#SBATCH -p Gvlab-S1-32
#SBATCH --gres=gpu:0
#SBATCH -n 8
#SBATCH --ntasks-per-node 8
#SBATCH --output slurm_output/%j.out
#SBATCH --error slurm_output/%j.err
#SBATCH --quotatype spot
#SBATCH --job-name data10
#SBATCH --requeue
#SBATCH --open-mode=append

source ~/.zshrc
proxy_off
conda activate backbone

for i in {0..10}
do 
    srun python dataset/construct_dataset.py --chunk_id $i --base_path s_hdd_new:s3://lwf_v1/OmniLV_Data/test
    sleep 2s
done