#!/bin/bash
set -e

export ENVIRONMENTNAME=mmdeploy_tensorrt_v2

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/tis697/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/tis697/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/tis697/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/tis697/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
export PATH=$HOME/miniconda3/bin:$PATH
# <<< conda initialize <<<

# Load necessary modules
module load gcc/9.2.0
module load cuda/11.7

# Create and activate a conda environment
echo "Creating and activating the conda environment..."
if [ -d "/n/groups/datta/tim_sainburg/conda_envs/$ENVIRONMENTNAME" ]; then
    echo "Conda environment already exists. Skipping creation."
else
    conda create --prefix /n/groups/datta/tim_sainburg/conda_envs/$ENVIRONMENTNAME python=3.8 -y
fi

source activate /n/groups/datta/tim_sainburg/conda_envs/$ENVIRONMENTNAME
conda activate /n/groups/datta/tim_sainburg/conda_envs/$ENVIRONMENTNAME

conda info --envs

echo "Installing CUDA and cuDNN..."
# Install cuda & cudnn
conda install -y nvidia/label/cuda-12.1.1::cuda-toolkit

echo "Installing PyTorch and related packages..."
# Install pytorch
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

echo "Installing TensorRT..."
# Install tensorrt
pip install nvidia-pyindex
pip install --upgrade nvidia-tensorrt
pip install pycuda

echo "Installing MMCV and related packages..."
# Install mmcv
pip install -U openmim
mim install -y mmengine
mim install -y "mmcv>=2.0.0rc2"
mim install -y mmdet
mim install -y "mmpose>=1.1.0"

echo "Installing MMDeploy..."
# Install mmdeploy
# 1. install MMDeploy model converter
pip install mmdeploy==1.3.1

# 2. install MMDeploy sdk inference
# you can install one to install according to whether you need gpu inference
# 2.1 support onnxruntime
pip install mmdeploy-runtime==1.3.1
# 2.2 support onnxruntime-gpu, tensorrt
pip install mmdeploy-runtime-gpu==1.3.1

echo "MMDeploy setup complete."