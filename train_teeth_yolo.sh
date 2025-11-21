#!/bin/bash

# YOLO牙齿检测模型训练脚本
# 严格按照Instance_seg_teeth原始仓库的训练流程
# 参考: Instance_seg_teeth/notebooks/yolov8/yolov8_train.ipynb

set -e  # 遇到错误立即退出

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_green() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_yellow() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_red() {
    echo -e "${RED}✗ $1${NC}"
}

echo "=========================================="
echo "YOLO牙齿检测模型训练"
echo "严格按照原始仓库流程"
echo "=========================================="
echo ""

# 检查虚拟环境
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_red "错误: 请先激活虚拟环境"
    echo "运行: source venv/bin/activate"
    exit 1
fi
print_green "虚拟环境已激活: $VIRTUAL_ENV"
echo ""

# 运行Python训练脚本（严格按照原始notebook流程）
python train_teeth_yolo.py

if [ $? -eq 0 ]; then
    echo ""
    print_green "训练流程完成！"
    echo ""
else
    echo ""
    print_red "训练失败"
    exit 1
fi
