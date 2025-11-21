# CUDA 11.8 Setup Guide

## 是否需要重新安装CUDA?

**简短回答**: 如果您已经有CUDA 11.8，不需要重新安装。如果您有其他版本的CUDA，建议安装CUDA 11.8以确保最佳兼容性。

## 检查当前CUDA版本

```bash
# 检查CUDA编译器版本
nvcc --version

# 检查GPU驱动和CUDA运行时版本
nvidia-smi
```

## CUDA版本兼容性说明

本项目使用 **TensorFlow 2.13**，它与以下CUDA版本兼容：
- **推荐**: CUDA 11.8 + cuDNN 8.6
- **也兼容**: CUDA 11.2 - 11.8

### 您的CUDA版本决策表

| 当前CUDA版本 | 是否需要重装 | 说明 |
|------------|------------|------|
| 无CUDA | ✅ 需要安装 | 必须安装CUDA 11.8 |
| CUDA 11.8 | ❌ 不需要 | 完美兼容，直接使用 |
| CUDA 11.2-11.7 | ⚠️ 建议升级 | 可能工作，但建议升级到11.8 |
| CUDA 11.x (x>8) | ⚠️ 可能需要 | TensorFlow 2.13不支持 |
| CUDA 12.x | ⚠️ 需要安装11.8 | TensorFlow 2.13不支持CUDA 12 |
| CUDA 10.x或更早 | ✅ 需要安装 | 太旧，必须升级 |

## 安装CUDA 11.8

### Ubuntu/Debian

```bash
# 下载CUDA 11.8 Toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# 安装
sudo sh cuda_11.8.0_520.61.05_linux.run

# 添加到PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证安装
nvcc --version
```

### CentOS/RHEL

```bash
# 下载CUDA 11.8 Toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# 安装
sudo sh cuda_11.8.0_520.61.05_linux.run

# 添加到PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Windows

1. 下载CUDA 11.8安装程序:
   https://developer.nvidia.com/cuda-11-8-0-download-archive

2. 选择Windows版本并下载exe安装程序

3. 运行安装程序，选择"Express安装"

4. 安装完成后重启电脑

5. 验证安装:
   ```cmd
   nvcc --version
   ```

## 安装cuDNN 8.6

cuDNN是深度学习必需的库。

### 下载cuDNN

1. 访问: https://developer.nvidia.com/cudnn
2. 注册/登录NVIDIA Developer账号
3. 下载 **cuDNN 8.6 for CUDA 11.x**

### Linux安装

```bash
# 假设下载的是tar.gz文件
tar -xzvf cudnn-linux-x86_64-8.6.x.x_cuda11-archive.tar.xz

# 复制文件到CUDA目录
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda-11.8/include
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda-11.8/lib64
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*
```

### Windows安装

1. 解压下载的zip文件
2. 复制以下文件到CUDA安装目录:
   - 从`bin`文件夹复制到`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
   - 从`include`文件夹复制到`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include`
   - 从`lib`文件夹复制到`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64`

## 验证安装

### 验证CUDA

```bash
# 检查CUDA版本
nvcc --version

# 应该显示类似:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2022 NVIDIA Corporation
# Built on Wed_Sep_21_10:33:58_PDT_2022
# Cuda compilation tools, release 11.8, V11.8.89
```

### 验证TensorFlow GPU支持

```python
import tensorflow as tf

# 检查TensorFlow版本
print(f"TensorFlow版本: {tf.__version__}")

# 检查GPU是否可用
gpus = tf.config.list_physical_devices('GPU')
print(f"可用的GPU数量: {len(gpus)}")

if len(gpus) > 0:
    print("GPU设备列表:")
    for gpu in gpus:
        print(f"  {gpu}")
    print("✅ GPU配置成功!")
else:
    print("❌ 未检测到GPU")
```

## 多CUDA版本共存

如果您需要保留其他CUDA版本，可以这样做：

```bash
# CUDA会安装到版本特定的目录
# /usr/local/cuda-11.8/
# /usr/local/cuda-12.0/
# 等等

# 切换CUDA版本只需更改软链接
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-11.8 /usr/local/cuda

# 或者使用环境变量
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## 常见问题

### Q1: nvidia-smi显示CUDA 12.x，但我需要11.8？

**A**: `nvidia-smi`显示的是GPU驱动支持的最高CUDA版本，不是实际安装的版本。您可以同时安装CUDA 11.8，只要驱动版本足够新即可。

```bash
# 检查实际安装的CUDA版本
nvcc --version  # 这个才是编译器版本
```

### Q2: TensorFlow找不到GPU？

**A**: 按顺序检查：

1. 验证GPU驱动: `nvidia-smi`
2. 验证CUDA安装: `nvcc --version`
3. 验证cuDNN安装: `ls /usr/local/cuda-11.8/lib64/libcudnn*`
4. 重装TensorFlow: `pip uninstall tensorflow && pip install tensorflow==2.13.0`

### Q3: 出现"libcudnn.so.8: cannot open shared object file"错误？

**A**: cuDNN未正确安装或未在库路径中。

```bash
# 检查cuDNN
ls /usr/local/cuda-11.8/lib64/libcudnn*

# 添加到库路径
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# 永久添加
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Q4: 我可以使用CPU运行吗？

**A**: 可以！使用`--device cpu`参数运行脚本。但GPU会快50-100倍。

```bash
python batch_process_teeth.py \
    --input_dir ./test_images \
    --yolo_weights ./yolov8x.pt \
    --device cpu
```

## Docker方案（推荐用于复杂环境）

如果CUDA安装太复杂，可以使用Docker：

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# 使用
# docker build -t teeth-segmentation .
# docker run --gpus all -v $(pwd):/workspace teeth-segmentation
```

## 总结

1. **检查当前CUDA**: `nvcc --version` 和 `nvidia-smi`
2. **如果已有CUDA 11.8**: 不需要重装，直接使用
3. **如果是其他版本**: 建议安装CUDA 11.8以确保兼容性
4. **安装cuDNN 8.6**: 必需，用于深度学习
5. **验证安装**: 运行setup_venv.sh脚本会自动检查

## 相关链接

- [CUDA 11.8下载](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- [cuDNN下载](https://developer.nvidia.com/cudnn)
- [TensorFlow GPU支持文档](https://www.tensorflow.org/install/gpu)
- [NVIDIA驱动下载](https://www.nvidia.com/download/index.aspx)
