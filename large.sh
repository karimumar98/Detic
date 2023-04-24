#!/bin/bash

#SBATCH --nodes=1
#SBATCJ --ntaks=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus=4
#SBATCH --mem-per-cpu=32g 
#SBATCH --gres=gpumem:16g
#SBATCH --time=06:00:00
#SBATCH -o "slurm-output/large-%j.out"

export RUN_NAME="Detic_convnet_large_2b_run_1"

gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "counted $gpu_count GPUS"

nvidia-smi

nvidia-smi -L

module load gcc/8.2.0 python_gpu/3.8.5 eth_proxy cuda/11.7.0 eth_proxy
source /cluster/project/zhang/umarka/clip_detector/prod/.venv/bin/activate

python train_net.py --num-gpus $gpu_count --config-file configs/convnet_large_2b.yaml --resume

: '
python train_net.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_CXT21k_640b32_4x_ft4x_max-size

''
