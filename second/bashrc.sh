#!/bin/env bash
# python create_data.py nuscenes_data_prep --data_path=/store01/shared/NuScenes/NuScenes-Training  --version="v1.0-trainval" --max_sweeps=10
# python create_data.py nuscenes_data_prep --root_path=/store01/shared/NuScenes/NuScenes-Training --dataset-name "v1.0-trainval  --version="v1.0-trainval" --max_sweeps=10
# python create_data.py nuscenes_data_prep --root_path /store01/shared/NuScenes/NuScenes-Training --dataset-name "v1.0-trainval"  --version "v1.0-trainval" --max_sweeps 10
# python pytorch/train.py evaluate --config_path=configs/pointpillars/ped_cycle/xyres_16.proto --model_dir=../new_ped_model --pickle_result=False
# python create_data.py nuscenes_data_prep --root-path=/store01/shared/NuScenes --version "mini" --dataset-name "trainval" --max_sweeps 10

# export PYTHONPATH=/home/jim/3d/second.nu.ki

# export LD_LIBRARY_PATH=/home/jim/anaconda3/envs/pillars/lib:/lib64
#export PYTHONPATH=/home/jim/3d/second.pytorch:/home/jim/3d/second.pytorch/SparseConvNet

export PYTHONPATH=/home/jim/3d/second.nu.ki:/home/jim/3d

export LD_LIBRARY_PATH=/home/jim/anaconda3/envs/pillars/lib:/home/jim/anaconda3/lib:/usr/lib64:/usr/local/lib64:/lib64
export PATH=/usr/local/cuda-10.1/bin:/opt/rh/devtoolset-7/root/usr/bin:/home/jim/anaconda3/envs/pillars/bin:/home/jim/anaconda3/condabin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin:/var/lib/snapd/snap/bin:/home/jim/.local/bin:/home/jim/bin:/usr/local/cuda/bin
export NUMBAPRO_CUDA_DRIVER=/usr/lib64/libcuda.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
# export PATH=$PATH:/usr/local/cuda/bin

