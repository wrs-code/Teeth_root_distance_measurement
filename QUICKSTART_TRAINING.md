# YOLOv8 牙齿检测训练 - 快速开始

## 🚀 5分钟快速开始

### 前提条件

```bash
# 1. 确认环境
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 2. 安装依赖
pip install ultralytics==8.0.28

# 3. 验证数据集
ls Instance_seg_teeth/Dataset/yolo_train_dataset/train/images/ | wc -l
# 应该显示: 894
```

### 开始训练（3种方法）

#### 方法1: 一键训练脚本 ⭐ 推荐

```bash
./train_yolov8.sh
```

#### 方法2: Python脚本

```bash
python train_yolov8_teeth.py --data yolo_training_config.yaml
```

#### 方法3: YOLO CLI

```bash
yolo task=detect mode=train \
    model=yolov8x.pt \
    data=yolo_training_config.yaml \
    epochs=30 \
    batch=10
```

## 📊 训练配置（官方最佳）

| 参数 | 值 | 说明 |
|-----|---|------|
| 模型 | yolov8x | 最大模型，最佳精度 |
| Epochs | 30 | 官方推荐 |
| Batch | 10 | 适配8-16GB显存 |
| Image Size | 640 | 标准大小 |
| 训练时间 | 2-4h | RTX 3090 |
| 预期mAP | 74.9% | 论文报告 |
| 预期AP50 | 94.6% | 论文报告 |

## 🎯 训练完成后

### 模型位置
```
runs/detect/teeth_detection_TIMESTAMP/weights/best.pt
```

### 快速测试
```bash
yolo predict \
    model=runs/detect/teeth_detection_TIMESTAMP/weights/best.pt \
    source=./test_images \
    conf=0.5
```

### 批量处理
```bash
python batch_process_teeth.py \
    --yolo_weights runs/detect/teeth_detection_TIMESTAMP/weights/best.pt \
    --input_dir ./test_images
```

## 🔧 常见问题快速解决

### CUDA Out of Memory
```bash
# 减小batch size
python train_yolov8_teeth.py --data yolo_training_config.yaml --batch 4
```

### 使用较小模型（更快）
```bash
# yolov8l: 更快，精度稍低 (~73% mAP)
python train_yolov8_teeth.py --data yolo_training_config.yaml --model yolov8l.pt
```

### 恢复中断的训练
```bash
python train_yolov8_teeth.py --data yolo_training_config.yaml --resume
```

## 📚 完整文档

详细信息请查看: [TRAINING_GUIDE.md](./TRAINING_GUIDE.md)

## ✅ 训练前检查清单

- [ ] CUDA可用
- [ ] ultralytics==8.0.28已安装
- [ ] 数据集存在 (894张训练图)
- [ ] GPU显存 ≥ 8GB
- [ ] 硬盘空间 ≥ 10GB

就这么简单！🎉
