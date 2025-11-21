#!/bin/bash

# 快速重新安装脚本 - 使用CUDA 11.8兼容的PyTorch

set -e

echo "=========================================="
echo "重新安装依赖 - CUDA 11.8版本"
echo "=========================================="
echo ""

# 激活虚拟环境
if [ -d "venv" ]; then
    echo "激活现有虚拟环境..."
    source venv/bin/activate
else
    echo "创建新的虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
fi

# 升级pip
echo ""
echo "升级pip..."
pip install --upgrade pip

# 先卸载可能已安装的torch（如果有的话）
echo ""
echo "清理现有的PyTorch安装..."
pip uninstall -y torch torchvision torchaudio || true

# 安装依赖
echo ""
echo "安装依赖（使用CUDA 11.8版本）..."
pip install -r requirements.txt

# 验证安装
echo ""
echo "=========================================="
echo "验证安装"
echo "=========================================="

echo ""
echo "1. 检查TensorFlow GPU支持..."
python3 << EOF
import tensorflow as tf
print(f"TensorFlow版本: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"TensorFlow检测到的GPU数量: {len(gpus)}")
if len(gpus) > 0:
    print("✅ TensorFlow GPU配置成功!")
    for gpu in gpus:
        print(f"  {gpu}")
else:
    print("⚠️ TensorFlow未检测到GPU（将使用CPU）")
EOF

echo ""
echo "2. 检查PyTorch CUDA版本..."
python3 << EOF
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print("✅ PyTorch GPU配置成功!")
else:
    print("⚠️ PyTorch未检测到GPU（将使用CPU）")
EOF

echo ""
echo "3. 检查Ultralytics (YOLOv8)..."
python3 << EOF
import ultralytics
print(f"Ultralytics版本: {ultralytics.__version__}")
print("✅ YOLOv8安装成功!")
EOF

echo ""
echo "=========================================="
echo "安装完成!"
echo "=========================================="
echo ""
echo "现在可以运行批量处理："
echo "  python batch_process_teeth.py --input_dir ./test_images --yolo_weights ./yolov8x.pt"
echo ""
