#!/bin/bash

export ENVIRONMENTNAME=mmdeploy_tensorrt

# Load necessary modules
module load gcc/9.2.0
module load cuda/11.7

# Create and activate a conda environment
conda create --prefix /n/groups/datta/tim_sainburg/conda_envs/$ENVIRONMENTNAME python=3.8 -y
conda activate /n/groups/datta/tim_sainburg/conda_envs/$ENVIRONMENTNAME

# Install PyTorch for GPU with CUDA 11.7
conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install MMCV
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc2"

# Install MMDeploy and its inference engine
# Model converter
pip install mmdeploy==1.3.1

# Choose one depending on your target inference need (ONNX Runtime or GPU/TensorRT/ONNX Runtime)
pip install mmdeploy-runtime==1.3.1  # for ONNX Runtime
pip install mmdeploy-runtime-gpu==1.3.1  # for TensorRT and ONNX Runtime GPU
pip install opencv-python

# Install Inference Engines

# Set up TensorRT
export TENSORRT_DIR=/n/groups/datta/tim_sainburg/tensorrt/TensorRT-8.2.3.0
export LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:${LD_LIBRARY_PATH}
export PATH=${TENSORRT_DIR}/bin:${PATH}

# Install TensorRT Python bindings
pip install ${TENSORRT_DIR}/python/tensorrt-8.2.3.0-cp38-none-linux_x86_64.whl
pip install pycuda

# Install TensorRT Python bindings
#pip install ${HOME}/usr/lib/python3.8/dist-packages/tensorrt-8.2.3.0-cp38-none-linux_x86_64.whl
#pip install pycuda

# Install ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.8.1
export LD_LIBRARY_PATH=${ONNXRUNTIME_DIR}/lib:${LD_LIBRARY_PATH}

# Install ONNX Runtime GPU
pip install onnxruntime-gpu==1.8.1
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-gpu-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-gpu-1.8.1.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-gpu-1.8.1
export LD_LIBRARY_PATH=${ONNXRUNTIME_DIR}/lib:${LD_LIBRARY_PATH}

echo "MMDeploy setup complete."