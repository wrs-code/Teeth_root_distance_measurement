#!/bin/bash

# 快速修复依赖问题
# 修复1: TensorFlow GPU支持
# 修复2: typing_extensions版本

set -e

echo "=========================================="
echo "修复依赖问题"
echo "=========================================="
echo ""

# 激活虚拟环境
if [ -d "venv" ]; then
    echo "激活虚拟环境..."
    source venv/bin/activate
else
    echo "错误: 虚拟环境不存在，请先运行 ./setup_venv.sh"
    exit 1
fi

echo ""
echo "修复1: 升级 typing_extensions (修复Ultralytics导入错误)..."
pip install --upgrade "typing_extensions>=4.6.0"

echo ""
echo "修复2: 安装TensorFlow GPU支持库..."
pip install nvidia-cudnn-cu11==8.6.0.163 nvidia-cuda-runtime-cu11==11.8.89

echo ""
echo "=========================================="
echo "验证修复"
echo "=========================================="

echo ""
echo "1. 测试Ultralytics导入..."
python3 << 'EOF'
try:
    import ultralytics
    print(f"✅ Ultralytics版本: {ultralytics.__version__}")
    print("✅ Ultralytics导入成功!")
except Exception as e:
    print(f"❌ Ultralytics导入失败: {e}")
    exit(1)
EOF

echo ""
echo "2. 测试TensorFlow GPU..."
python3 << 'EOF'
import tensorflow as tf
print(f"TensorFlow版本: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"检测到的GPU数量: {len(gpus)}")
if len(gpus) > 0:
    print("✅ TensorFlow GPU配置成功!")
    for gpu in gpus:
        print(f"  {gpu}")
else:
    print("⚠️ TensorFlow未检测到GPU（可能需要重启或检查驱动）")
EOF

echo ""
echo "3. 测试PyTorch CUDA..."
python3 << 'EOF'
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print("✅ PyTorch GPU配置成功!")
EOF

echo ""
echo "=========================================="
echo "修复完成!"
echo "=========================================="
echo ""
echo "如果TensorFlow仍然检测不到GPU，请尝试："
echo "1. 检查LD_LIBRARY_PATH是否包含CUDA库路径"
echo "2. 重启终端或运行: export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:\$LD_LIBRARY_PATH"
echo "3. 验证: ls /usr/local/cuda-11.8/lib64/libcudnn*"
echo ""
echo "现在可以运行批量处理："
echo "  python batch_process_teeth.py --input_dir ./test_images --yolo_weights ./yolov8x.pt"
echo ""
