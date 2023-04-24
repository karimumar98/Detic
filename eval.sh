#!/bin/bash

#SBATCH --nodes=1
#SBATCJ --ntaks=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus=4
#SBATCH --mem-per-cpu=32g 
#SBATCH --gres=gpumem:16g
#SBATCH --time=00:30:00
#SBATCH -o "slurm-output/large_clip_35k.out"

#export RUN_NAME="Detic_convnet_base_laion_400m_run_1"

gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "counted $gpu_count GPUS"

nvidia-smi

nvidia-smi -L

## Load the needed modules
module load gcc/8.2.0 python_gpu/3.8.5 eth_proxy cuda/11.7.0 eth_proxy
## Activate the virtualenv created earlier
source /cluster/project/zhang/umarka/clip_detector/prod/.venv/bin/activate


#python train_net.py --num-gpus $gpu_count --config-file configs/convnet_large_2b.yaml --eval-only MODEL.WEIGHTS /cluster/project/zhang/umarka/clip_detector/dev/Detic/output/convnext_large_d_laion2b_s26b_b102k_augreg_big_zs/model_0004999.pth

python train_net.py --num-gpus $gpu_count --config-file configs/convnet_large_2b.yaml --eval-only MODEL.WEIGHTS output/convnext_large_d_laion2b_s26b_b102k_augreg_eval/model_0034999.pth

#python train_net.py --num-gpus $gpu_count --config-file configs/convnet_large_2b.yaml --eval-only MODEL.WEIGHTS output/convnext_large_d_laion2b_s26b_b102k_augreg/model_final.pth
: '
python train_net.py --num-gpus 1 --config-file configs/convnet_base_400m.yaml --resume

python train_net.py --num-gpus 1 --config-file configs/clip_base_lvis.yaml

python train_net.py --num-gpus 1 --config-file configs/Detic_LbaseCCcapimg_CLIP_R5021k_640b64_4x_ft4x_max-size.yaml
python train_net.py --num-gpus 4 --config-file configs/Detic_LbaseCCcapimg_CLIP_R5021k_640b64_4x_ft4x_max-size.yaml --eval-only MODEL.WEIGHTS output/base_fb_in22k_ft_in1k_384/model_0009999.pth

'
